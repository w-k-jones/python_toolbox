import numpy as np
from numpy import ma
import xarray as xr
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.ndimage import label as label_features
from scipy.signal import convolve
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import local_minima, h_minima, selem, star, ball, watershed
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing, binary_opening, binary_dilation, binary_erosion
import pandas as pd
import cv2 as cv
from .abi_tools import get_abi_IR, get_abi_ref


def constrain_thresholds(data, upper_thresh=np.inf, lower_thresh=-np.inf):
    return np.maximum(np.minimum(data,upper_thresh),lower_thresh)

def mark_thresholds(data, upper_thresh=np.inf, lower_thresh=-np.inf, mask=None):
    markers = np.maximum(2*(data>=upper_thresh), data<=lower_thresh)
    if mask is not None:
        marker *= mask
    return markers

def get_sobel_matrix(ndims):
    sobel_matrix = np.array([-1,0,1])
    for i in range(ndims-1):
        sobel_matrix = np.multiply.outer(np.array([1,2,1]), sobel_matrix)
    return sobel_matrix

def convolve_grad(input_matrix, sobel_matrix):
    output_matrix = np.zeros(input_matrix.shape)
    for i in range(np.product(sobel_matrix.shape)):
        if sobel_matrix.ravel()[i] != 0:
            loc = tuple([slice([0,1][ind<1], [None,-1][ind>1]) for ind in np.unravel_index(i, sobel_matrix.shape)])
            offset_loc = tuple([slice([0,1][ind>1], [None,-1][ind<1]) for ind in np.unravel_index(i, sobel_matrix.shape)])
            output_matrix[loc] += (input_matrix[offset_loc] - input_matrix[loc]) * sobel_matrix.ravel()[i]
    return output_matrix

# Faster 'convolution' function to find the gradient for the sobel case -- uphill only version
def convolve_grad_uphill(input_matrix, sobel_matrix, downhill=False):
    output_matrix = np.zeros(input_matrix.shape)
    for i in range(np.product(sobel_matrix.shape)):
        if sobel_matrix.ravel()[i] != 0:
            loc = tuple([slice([0,1][ind<1], [None,-1][ind>1]) for ind in np.unravel_index(i, sobel_matrix.shape)])
            offset_loc = tuple([slice([0,1][ind>1], [None,-1][ind<1]) for ind in np.unravel_index(i, sobel_matrix.shape)])
            if downhill:
                output_matrix[loc] += np.minimum(input_matrix[offset_loc] - input_matrix[loc],0) * sobel_matrix.ravel()[i]
            else:
                output_matrix[loc] += np.maximum(input_matrix[offset_loc] - input_matrix[loc],0) * sobel_matrix.ravel()[i]
    return output_matrix

# nD Sobel gradient function
def sobel(input_matrix, use_convolve=False):
    ndims = len(input_matrix.shape)
    sobel_matrix = get_sobel_matrix(ndims)
    output_matrix = np.zeros(input_matrix.shape)
    if use_convolve:
        for i in range(ndims):
            if np.sum(np.isfinite(input_matrix)) == input_matrix.size:
                output_matrix += convolve(input_matrix, np.rollaxis(sobel_matrix,i))[[slice(1,-1)]*ndims]**2
            else:
                output_matrix += convolve(input_matrix, np.rollaxis(sobel_matrix,i), method='direct')[[slice(1,-1)]*ndims]**2
    else:
        for i in range(ndims):
            output_matrix += convolve_grad(input_matrix, np.rollaxis(sobel_matrix,i))**2
    output_matrix = output_matrix**0.5
    return output_matrix

# 'Uphill only' sobel operation. Finds only uphill slopes, to avoid masking smaller cloud objects
def uphill_sobel(input_matrix, downhill=False, uphill_positive=False, use_convolve=False, axis=None):
    ndims = len(input_matrix.shape)
    sobel_matrix = get_sobel_matrix(ndims)
    if uphill_positive:
        sobel_matrix = np.abs(sobel_matrix)
    one_arr = np.zeros(sobel_matrix.size)
    one_arr[0] = 1
    output_matrix = np.stack([np.zeros(input_matrix.shape)]*ndims)
    if use_convolve:
        if downhill:
            for i in range(sobel_matrix.size):
                if np.sum(np.isfinite(input_matrix)) == input_matrix.size:
                    temp_matrix = np.minimum(convolve(input_matrix,np.roll(one_arr,i).reshape(sobel_matrix.shape))[[slice(1,-1)]*ndims]-input_matrix, 0)
                else:
                    temp_matrix = np.minimum(convolve(input_matrix,np.roll(one_arr,i).reshape(sobel_matrix.shape), method='direct')[[slice(1,-1)]*ndims]-input_matrix, 0)
                for j in range(ndims):
                    output_matrix[j] += temp_matrix * np.rollaxis(sobel_matrix,j).ravel()[i]
        else:
            for i in range(sobel_matrix.size):
                if np.sum(np.isfinite(input_matrix)) == input_matrix.size:
                    temp_matrix = np.maximum(convolve(input_matrix,np.roll(one_arr,i).reshape(sobel_matrix.shape))[[slice(1,-1)]*ndims]-input_matrix, 0)
                else:
                    temp_matrix = np.maximum(convolve(input_matrix,np.roll(one_arr,i).reshape(sobel_matrix.shape), method='direct')[[slice(1,-1)]*ndims]-input_matrix, 0)
                for j in range(ndims):
                    output_matrix[j] += temp_matrix * np.rollaxis(sobel_matrix,j).ravel()[i]
    else:
        if axis is None:
            for i in range(ndims):
                output_matrix[i] = convolve_grad_uphill(input_matrix, np.rollaxis(sobel_matrix,i), downhill=downhill)
            output_matrix = np.sum(output_matrix**2,0)**0.5
        elif axis not in range(ndims):
            raise ValueError('Axis value is '+str(axis)+' but input data only has '+str(ndims)+' dimensions')
        else:
            output_matrix = convolve_grad_uphill(input_matrix, np.rollaxis(sobel_matrix, axis), downhill=downhill)
    return output_matrix

def get_markers(field_in, upper_thresh=None, lower_thresh=None, mask=None):
    if lower_thresh is not None:
        markers = field_in < lower_thresh
        upper_marker = 2
    else:
        upper_marker = 1
    if upper_thresh is not None:
        if upper_marker == 2:
            markers = np.maximum(2 * (field_in > upper_thresh), markers)
        else:
            markers = field_in > upper_thresh
    elif upper_marker == 1:
        raise ValueError("""Error in get_markers: Atleast one of keywords upper_thresh,
                            lower_thresh must be supplied""")
    if mask is not None:
        markers *= mask

    return markers

def get_object_info(object_slice, files_list, feature_mask, lats, lons, pixel_area, label=True):
    object_info={}
    object_info['files']=files_list[object_slice[0]]
    object_info['slice']=object_slice[1:]
    object_info['duration'] = (object_info['files'][-1][0] - object_info['files'][0][0]).total_seconds()
    y0 = object_slice[1].start
    y1 = object_slice[1].stop-1
    x0 = object_slice[2].start
    x1 = object_slice[2].stop-1
    object_info['UL_corner_latlon'] = [lats[y0,x0],lons[y0,x0]]
    object_info['LL_corner_latlon'] = [lats[y1,x0],lons[y1,x0]]
    object_info['UR_corner_latlon'] = [lats[y0,x1],lons[y0,x1]]
    object_info['LR_corner_latlon'] = [lats[y1,x1],lons[y1,x1]]
    object_info['feature_mask'] = feature_mask[object_slice]==label
    object_info['central_latlon'] = [(np.nanmean(lats[object_info['slice']][mask]),np.nanmean(lons[object_info['slice']][mask])) for mask in object_info['feature_mask']]
    object_info['local_time'] = [f[0]+timedelta(hours=(object_info['central_latlon'][i][1]/15)) for i,f in enumerate(object_info['files'])]
    object_info['pixel_count'] = [np.sum(mask) for mask in object_info['feature_mask']]
    object_info['area'] = [np.sum(pixel_area[object_info['slice']][mask]) for mask in object_info['feature_mask']]
    object_info['timedeltas'] = [(object_info['files'][i][0]-object_info['files'][i-1][0]).total_seconds() for i in range(1, len(object_info['files']))]
    object_info['growth_rate'] = [(object_info['area'][i]-object_info['area'][i-1])/object_info['timedeltas'][i-1] for i in range(1, len(object_info['files']))]
    return object_info

def subsegment(bt, h_level=2.5, min_separation=2, sigma=0.5, BT_thresh=240):
    bt.fill_value = bt.max()
    peaks = np.all((h_minima(gaussian_filter(bt.filled(),sigma),h_level),local_minima(gaussian_filter(bt.filled(),sigma),connectivity=min_separation),bt.filled()<BT_thresh), axis=0)
    eroded_mask = binary_erosion(bt.mask, structure=(np.ones((1,3,3))))
    segments = watershed(bt.filled(),label_features(peaks)[0], mask=np.logical_not(eroded_mask))
    dilated_segments = grey_dilation(segments, structure=np.ones((3,3,3)))
    segments[segments==0] = dilated_segments[segments==0]
    segments = ma.array(segments, mask = bt.mask)
    segments.fill_value = 0
    return segments

def get_central_xy(obj):
    xx,yy = np.meshgrid(np.arange(obj['slice'][1].start,obj['slice'][1].stop),np.arange(obj['slice'][0].start,obj['slice'][0].stop))
    central_x = ma.array(np.stack([xx]*obj['feature_mask'].shape[0]),mask=np.logical_not(obj['feature_mask'])).mean(axis=(1,2)).data
    central_y = ma.array(np.stack([yy]*obj['feature_mask'].shape[0]),mask=np.logical_not(obj['feature_mask'])).mean(axis=(1,2)).data
    return central_x, central_y

def subsegment_object(obj, **kwargs):
    c13_ds = xr.open_mfdataset([f[13] for f in obj['files']], combine='nested', concat_dim='t')
    BT = get_abi_IR(c13_ds[{'y':obj['slice'][0], 'x':obj['slice'][1]}]).compute()
    BTma = BT.to_masked_array()
    BTma.mask = np.logical_not(obj['feature_mask'])
    segment_labels = subsegment(BTma, **kwargs)
    return segment_labels

def recursive_linker(links_list1=None, links_list2=None, label_list1=None, label_list2=None, overlap_list1=None, overlap_list2=None):
    recursive = False
    if links_list1 is None:
        links_list1=[]
    if links_list2 is None:
        links_list2=[]
    if label_list1 is None:
        label_list1=[]
    if label_list2 is None:
        label_list2=[]
    if overlap_list1 is None:
        overlap_list1=[]
    if overlap_list2 is None:
        overlap_list2=[]
    for i in links_list1:
        if i in label_list1:
            loc = label_list1.index(i)
            label = label_list1.pop(loc)
            overlap = overlap_list1.pop(loc)
            for j in overlap:
                if j not in links_list2:
                    links_list2.append(j)
                    recursive = True
    if recursive:
        links_list2, links_list1 = recursive_linker(links_list1=links_list2, links_list2=links_list1, label_list1=label_list2, label_list2=label_list1, overlap_list1=overlap_list2, overlap_list2=overlap_list1)
    return links_list1, links_list2

def link_labels(labels1, labels2):
    overlap_mask = np.logical_and((labels1>0),(labels2>0))
    labels_masked1 = labels1[overlap_mask]
    labels_masked2 = labels2[overlap_mask]
    label_list1 = np.unique(labels_masked1).tolist()
    label_list2 = np.unique(labels_masked2).tolist()
    overlap_list1 = [np.unique(labels_masked2[labels_masked1==label]).tolist() for label in label_list1]
    overlap_list2 = [np.unique(labels_masked1[labels_masked2==label]).tolist() for label in label_list2]
    links_list1 = []
    links_list2 = []
    while len(label_list1)>0:
        temp_links1, temp_links2 = recursive_linker([label_list1[0]], label_list1=label_list1, label_list2=label_list2, overlap_list1=overlap_list1, overlap_list2=overlap_list2)
        links_list1.append(temp_links1)
        links_list2.append(temp_links2)
    return links_list1, links_list2

def ds_to_8bit(ds, vmin=None, vmax=None):
    if vmin is None:
        vmin= ds.min()
    if vmax is None:
        vmax= ds.max()
    ds_out = (ds-vmin)*255/(vmax-vmin)
    return ds_out.astype('uint8')

def optical_flow_track(frame0, frame1, frame0_labels, frame1_labels):
    u,v = get_flow(frame0, frame1)
    u,v = np.rint(u), np.rint(v)
    y,x=np.where(frame0_labels>0)
    y_new = np.minimum(np.maximum(y+v[frame0_labels>0],0),1499).astype('int')
    x_new = np.minimum(np.maximum(x+u[frame0_labels>0],0),2499).astype('int')
    new_frame_labels = np.zeros_like(frame0_labels)
    new_frame_labels[y_new, x_new] = frame0_labels[y,x]
    new_frame_labels = grey_closing(new_frame_labels, size=(3,3)).astype('int')
    flow_links = link_labels(new_frame_labels, frame1_labels)
    flow_links = zip(*flow_links)
    linked_labels = np.stack([frame0_labels, frame1_labels])
    for links in flow_links:
        linked_labels[0][np.any(frame0_labels[...,np.newaxis] == links[0], axis=-1)] = links[0][0]
        linked_labels[1][np.any(frame1_labels[...,np.newaxis] == links[1], axis=-1)] = links[0][0]
    missed_labels = np.unique(np.stack([frame0_labels,frame1_labels])[np.logical_and(np.stack([frame0_labels,frame1_labels])>0,linked_labels==0)])
    if missed_labels.size > 0:
        raise Exception("Labels missed")
    # Have removed overlap proportion for now, will need to look into improving the method
    # overlap_proportion = [np.sum(np.logical_and(
    #     np.any(new_frame_labels[...,np.newaxis] == links[0], axis=-1),
    #     np.any(frame1_labels[...,np.newaxis] == links[1], axis=-1)))/
    #     np.sum(np.any(new_frame_labels[...,np.newaxis] == links[0], axis=-1)) for links in flow_links]
    return linked_labels#, overlap_proportion

def get_flow(frame0, frame1):
    flow = cv.calcOpticalFlowFarneback(ds_to_8bit(frame0).data.compute(),ds_to_8bit(frame1).data.compute(), None, 0.5, 3, 4, 3, 5, 1.2, 0)
    return flow[...,0], flow[...,1]

def interp_flow(da, flow):
    x = np.arange(da.coords['x'].size)
    y = np.arange(da.coords['y'].size)
    xx, yy = np.meshgrid(x, y)
    new_xx, new_yy = xx + flow[...,0], yy + flow[...,1]
    interp_da = xr.apply_ufunc(interpolate.interpn, (y, x), da, (new_yy, new_xx), kwargs={'method':'linear', 'bounds_error':False, 'fill_value':None})
    return interp_da

def get_diff_flow(frame0, frame1):
    vmin = np.minimum(frame0.min(), frame1.min())
    vmax = np.maximum(frame0.max(), frame1.max())
    flow = cv.calcOpticalFlowFarneback(ds_to_8bit(frame0).data.compute(),#, vmin=vmin, vmax=vmax).data.compute(),
                                       ds_to_8bit(frame1).data.compute(),#, vmin=vmin, vmax=vmax).data.compute(),
                                       None, 0.5, 4, 8, 4, 5, 1.1, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
    interp_frame1 = interp_flow(frame1.compute(), flow)
    return interp_frame1 - frame0

def get_object_abi(obj, channel, expand=0):
    x_slice = slice(obj['slice'][1].start-expand, obj['slice'][1].stop+expand)
    y_slice = slice(obj['slice'][0].start-expand, obj['slice'][0].stop+expand)

    with xr.open_mfdataset([f[channel] for f in obj['files']], combine='nested', concat_dim='t') as ds:
        if channel > 6:
            return get_abi_IR(ds[{'x':x_slice, 'y':y_slice}])
        else:
            return get_abi_ref(ds[{'x':x_slice, 'y':y_slice}])

def get_ds_flow(da, direction='forwards', pyr_scale=0.5, levels=5, winsize=15, iterations=4, poly_n=5, poly_sigma=1.2, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN):
    x_flow = xr.zeros_like(da).compute()*np.nan
    y_flow = xr.zeros_like(da).compute()*np.nan
    if direction == 'forwards':
        frame1 = ds_to_8bit(da[0]).data.compute()
        for i in range(1, da.coords['t'].size):
            frame0, frame1 = frame1, ds_to_8bit(da[i]).data.compute()

            flow = xr.apply_ufunc(cv.calcOpticalFlowFarneback, frame0, frame1,
                                   None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            x_flow[i-1], y_flow[i-1] = flow[...,0], flow[...,1]
    elif direction == 'backwards':
        frame1 = ds_to_8bit(da[-1]).data.compute()
        for i in range(da.coords['t'].size-1, 0, -1):
            frame0, frame1 = frame1, ds_to_8bit(da[i-1]).data.compute()

            flow = xr.apply_ufunc(cv.calcOpticalFlowFarneback, frame0, frame1,
                                   None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            x_flow[i], y_flow[i] = flow[...,0], flow[...,1]
    else:
        raise ValueError("""keyword 'direction' only accepts 'forwards' or 'backwards' as acceptable values""")
    return x_flow, y_flow

def interp_ds_flow(da, x_flow, y_flow, direction='forwards'):
    da_flow = xr.zeros_like(da).compute()*np.nan
    if direction == 'forwards':
        for i in range(da.coords['t'].size-1):
            da_flow[i] = interp_flow(da[i+1].compute(), np.stack([x_flow[i], y_flow[i]], axis=-1))
    elif direction == 'backwards':
        for i in range(1,da.coords['t'].size):
            da_flow[i] = interp_flow(da[i-1].compute(), np.stack([x_flow[i], y_flow[i]], axis=-1))
    else:
        raise ValueError("""keyword 'direction' only accepts 'forwards' or 'backwards' as acceptable values""")
    return da_flow

def get_growth_metric(da, x_flow, y_flow, direction='forwards'):
    da_d2 = da.differentiate('x').differentiate('x') + da.differentiate('y').differentiate('y')
    da_flow = interp_ds_flow(da, x_flow, y_flow, direction=direction)
    if direction == 'forwards':
        da_growth = np.maximum(-(da_flow - da) * da_d2/1e10, 0)
    elif direction == 'backwards':
        da_growth = np.maximum((da_flow - da) * da_d2/1e10, 0)
    else:
        raise ValueError("""keyword 'direction' only accepts 'forwards' or 'backwards' as acceptable values""")
    return da_growth

def expand_labels_optflow(labels, x_flow_forwards, y_flow_forwards,
                          x_flow_backwards, y_flow_backwards,
                          max_iter=100, mask=None):
    z,y,x = (range(shape) for shape in labels.shape)
    zzz, yyy, xxx = np.meshgrid(z,y,x, indexing='ij')
    xxx_for = xxx + x_flow_forwards
    yyy_for = yyy + y_flow_forwards
    xxx_back = xxx + x_flow_backwards
    yyy_back = yyy + y_flow_backwards
    iter_labels = labels.copy()
    old_sum = np.sum(iter_labels > 0)
    for i in range(max_iter):
        if mask is None:
            wh_interp = iter_labels == 0
        else:
            wh_interp = np.logical_and(mask, iter_labels==0)
        iter_labels[wh_interp] = interpolate.interpn((z,y,x),
                                                     iter_labels,
                                                    (zzz[wh_interp]+1,
                                                     yyy_for.data[wh_interp],
                                                     xxx_for.data[wh_interp]),
                                                    method='nearest',
                                                    bounds_error=False,
                                                    fill_value=0)
        if mask is None:
            wh_interp = iter_labels == 0
        else:
            wh_interp = np.logical_and(mask, iter_labels==0)
        iter_labels[wh_interp] = interpolate.interpn((z,y,x),
                                                     iter_labels,
                                                     (zzz[wh_interp]-1,
                                                      yyy_back.data[wh_interp],
                                                      xxx_back.data[wh_interp]),
                                                     method='nearest',
                                                     bounds_error=False,
                                                     fill_value=0)
        new_sum = np.sum(iter_labels>0)
        print(new_sum)
        if new_sum == old_sum:
            break
        old_sum = new_sum

    return iter_labels

def flow_gradient_watershed(field, markers, mask=None, flow=None, max_iter=100, max_no_progress=10, expand_mask=True):
    import numpy as np
    from scipy import interpolate
    import scipy.ndimage as ndi
    from skimage.feature import peak_local_max
    import warnings
    shape = [np.arange(shape) for shape in field.shape]
    grids = np.stack(np.meshgrid(*shape, indexing='ij'),-1)
    grads = np.stack(np.gradient(-field),-1)
    if flow is not None:
        if grids.shape[-1] == 2:
            interp_flow_for = interpolate.interpn(shape, field,
                                              np.stack([grids[1:,...,0],
                                                        np.maximum(np.minimum(
                                                            grids[1:,...,1:].squeeze()+flow[:-1],
                                                            (np.array(field.shape)-1)[1:]), 0)],-1),
                                              method='linear')
            interp_flow_back = interpolate.interpn(shape, field,
                                              np.stack([grids[:-1,...,0],
                                                        np.maximum(np.minimum(
                                                            grids[:-1,...,1:].squeeze()-flow[1:],
                                                            (np.array(field.shape)-1)[1:]), 0)],-1),
                                              method='linear')
        else:
            interp_flow_for = interpolate.interpn(shape, field,
                                              np.concatenate([grids[1:,...,0][...,np.newaxis],
                                                        np.maximum(np.minimum(
                                                            grids[1:,...,1:].squeeze()+flow[:-1],
                                                            (np.array(field.shape)-1)[1:]), 0)],-1),
                                              method='linear')
            interp_flow_back = interpolate.interpn(shape, field,
                                              np.concatenate([grids[:-1,...,0][...,np.newaxis],
                                                        np.maximum(np.minimum(
                                                            grids[:-1,...,1:].squeeze()-flow[1:],
                                                            (np.array(field.shape)-1)[1:]), 0)],-1),
                                              method='linear')
        # Replace the first dimension gradients with the new flow gradients
        grads[1:-1,...,0] = -(interp_flow_for[1:]-interp_flow_back[:-1])/2
        grads[0,...,0] = -(interp_flow_for[0]-field[0])
        grads[-1,...,0] = (interp_flow_back[-1]-field[-1])
    grads_mag = (np.sum(grads**2,-1)**0.5)
    wh_mag = grads_mag!=0
    new_grads = grads.copy()
    new_grads[wh_mag] /= grads_mag[wh_mag][...,np.newaxis]
    pos = grids.astype(float)+new_grads
    if flow is not None:
        if pos.shape[-1] == 2:
            pos[...,1] += flow*new_grads[...,0]
        else:
            pos[...,1:] += flow*new_grads[...,0][...,np.newaxis]
    pos = np.maximum(np.minimum(pos, (np.array(field.shape)-1)), 0)

    local_min = peak_local_max(-field, indices=False, exclude_border=False)
    max_markers = np.nanmax(markers)
    new_markers = ndi.label(np.logical_or(local_min, np.logical_not(wh_mag))*(markers==0), structure=np.ones([3]*len(field.shape)))[0]
    new_markers[new_markers>0]+=max_markers
    fill_markers = markers+new_markers
    if mask is not None:
        fill_markers[np.logical_not(mask)] = -1
    fill = fill_markers.copy()
    wh = fill==0
    n_to_fill = np.sum(wh)
    counter = 0
    for step in range(1, max_iter+1):
        fill[wh] = interpolate.interpn(shape, fill, pos[wh], method='nearest')
        wh = fill==0
        if np.sum(wh) == n_to_fill:
            counter += 1
            if counter >= max_no_progress:
                warnings.warn('Reached maximum iterations without progress. Remaining unfilled pixels = '+str(n_to_fill))
                break
        else:
            counter = 0
            n_to_fill = np.sum(wh)
        if np.sum(wh) == 0:
            break
        else:
            new_grads = np.stack([interpolate.interpn(shape, grads[...,i], pos[wh], method='linear')
                                  for i in range(grads.shape[-1])], -1)
            new_grads_mag = (np.sum(new_grads**2,-1)**0.5)
            wh_mag = new_grads_mag!=0
            new_grads[wh_mag] /= new_grads_mag[wh_mag][...,np.newaxis]
            if flow is not None:
                new_flow = interpolate.interpn(shape, flow, pos[wh], method='linear')
                if new_grads.shape[-1] == 2:
                    new_grads[...,1] += new_flow * new_grads[...,0]
                else:
                    new_grads[...,1:] += new_flow * new_grads[...,0][...,np.newaxis]
            pos[wh] += new_grads
            pos = np.maximum(np.minimum(pos, (np.array(field.shape)-1)), 0)
    else:
        warnings.warn('Reached maximum iterations without completion. Remaining unfilled pixels = '+str(n_to_fill))

    for i in np.unique(fill[fill>max_markers]):
        wh_i = fill==i
        edges = ndi.morphology.binary_dilation(wh_i)*(fill!=i)
        if expand_mask:
            wh = edges*(fill!=0)
        else:
            wh = edges*(fill>0)
        if np.any(wh):
            min_label = fill[wh][np.argmin(field[wh])]
        else:
            min_label = 0
        fill[wh_i] = min_label
    print(step)
    return np.maximum(fill,0)

# def flow_neighbour(field, flow=None, structure=None):
#     if structure=None:
#         structure=np.array()
#
#
#     return
#
# def flow_neighbour_watershed(field, markers, mask=None, flow=None, max_iter=100, max_no_progress=10, expand_mask=True):
