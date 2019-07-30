import numpy as np
from numpy import ma
import xarray as xr
from datetime import datetime, timedelta
from scipy.ndimage import label as label_features
from scipy.signal import convolve
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import local_minima, h_minima, selem, star, ball, watershed
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing, binary_opening, binary_dilation, binary_erosion
import pandas as pd
import cv2 as cv
from .abi_tools import get_abi_IR


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
def uphill_sobel(input_matrix, downhill=False, uphill_positive=False, use_convolve=False):
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
        for i in range(ndims):
            output_matrix[i] = convolve_grad_uphill(input_matrix, np.rollaxis(sobel_matrix,i), downhill=downhill)
    output_matrix = np.sum(output_matrix**2,0)**0.5
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

def ds_to_8bit(ds):
    ds_out = (ds-ds.min())*255/(ds.max()-ds.min())
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
