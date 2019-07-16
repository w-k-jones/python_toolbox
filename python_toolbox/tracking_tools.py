import numpy as np
from numpy import ma
import xarray as xr
from datetime import datetime, timedelta
from scipy.ndimage import label
from scipy.signal import convolve
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import local_minima, h_minima, selem, star, ball, watershed
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing, binary_opening, binary_dilation
import pandas as pd


def constrain_thresholds(data, upper_thresh=np.inf, lower_thresh=-np.inf):
    return np.maximum(np.minimum(data,upper_thresh),lower_thresh)

def mark_thresholds(data, upper_thresh=np.inf, lower_thresh=-np.inf, mask=None):
    markers = np.maximum(2*(data>=upper_thresh), data<=lower_thresh)
    if mask is not None:
        marker *= mask
    return markers

def convolve_grad(input_matrix, sobel_matrix):
    output_matrix = np.zeros(input_matrix.shape)
    for i in range(np.product(sobel_matrix.shape)):
        if sobel_matrix.ravel()[i] != 0:
            loc = tuple([slice([0,1][ind<1], [None,-1][ind>1]) for ind in np.unravel_index(i, sobel_matrix.shape)])
            offset_loc = tuple([slice([0,1][ind>1], [None,-1][ind<1]) for ind in np.unravel_index(i, sobel_matrix.shape)])
            output_matrix[loc] += (input_matrix[offset_loc] - input_matrix[offset_loc]) * sobel_matrix.ravel()[i]
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

def subsegment(bt, h_level=2.5, min_separation=2):
    bt.fill_value = bt.max()
    peaks = np.all((h_minima(gaussian_filter(bt.filled(),0.5),h_level),local_minima(gaussian_filter(bt.filled(),0.5),connectivity=min_separation),bt.filled()<240), axis=0)
    segments = watershed(bt.filled(),label_features(peaks)[0], mask=np.logical_not(bt.mask))
    return segments

def get_central_xy(obj):
    xx,yy = np.meshgrid(np.arange(obj['slice'][1].start,obj['slice'][1].stop),np.arange(obj['slice'][0].start,obj['slice'][0].stop))
    central_x = ma.array(np.stack([xx]*obj['feature_mask'].shape[0]),mask=np.logical_not(obj['feature_mask'])).mean(axis=(1,2)).data
    central_y = ma.array(np.stack([yy]*obj['feature_mask'].shape[0]),mask=np.logical_not(obj['feature_mask'])).mean(axis=(1,2)).data
    return central_x, central_y

def subsegment_object(obj):
    c13_ds = xr.open_mfdataset([f[13] for f in obj['files']], concat_dim='t')
    BT = get_abi_IR(c13_ds[{'y':obj['slice'][0], 'x':obj['slice'][1]}]).compute()
    BTma = BT.to_masked_array()
    BTma.mask = np.logical_not(obj['feature_mask'])
    segment_labels = subsegment(BTma)
    return segment_labels
