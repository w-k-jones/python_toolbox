import cv2 as cv
import numpy as np
from numpy import ma
import xarray as xr
from scipy import interpolate
from python_toolbox import dataset_tools
from scipy import ndimage as ndi
import pdb

def ds_to_8bit(ds, vmin=None, vmax=None):
    if vmin is None:
        vmin= ds.min()
    if vmax is None:
        vmax= ds.max()
    ds_out = (ds-vmin)*255/(vmax-vmin)
    return ds_out.astype('uint8')

def get_ds_flow(da, direction='forwards', pyr_scale=0.5, levels=5, winsize=15, iterations=4, poly_n=5, poly_sigma=1.2, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN, dtype=None):
    if dtype == None:
        x_flow = xr.zeros_like(da).compute()*np.nan
        y_flow = xr.zeros_like(da).compute()*np.nan
    else:
        x_flow = xr.zeros_like(da, dtype=dtype).compute()*np.nan
        y_flow = xr.zeros_like(da, dtype=dtype).compute()*np.nan
    if direction == 'forwards':
        frame1 = ds_to_8bit(da[0]).compute().data
        for i in range(1, da.coords['t'].size):
            frame0, frame1 = frame1, ds_to_8bit(da[i]).compute().data

            flow = xr.apply_ufunc(cv.calcOpticalFlowFarneback, frame0, frame1,
                                   None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            x_flow[i-1], y_flow[i-1] = flow[...,0], flow[...,1]
    elif direction == 'backwards':
        frame1 = ds_to_8bit(da[-1]).compute().data
        for i in range(da.coords['t'].size-1, 0, -1):
            frame0, frame1 = frame1, ds_to_8bit(da[i-1]).compute().data

            flow = xr.apply_ufunc(cv.calcOpticalFlowFarneback, frame0, frame1,
                                   None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            x_flow[i], y_flow[i] = flow[...,0], flow[...,1]
    else:
        raise ValueError("""keyword 'direction' only accepts 'forwards' or 'backwards' as acceptable values""")
    return x_flow, y_flow

def get_abi_multispectral_flow(da_list, channels=None, direction='forwards', pyr_scale=0.5, levels=5, winsize=15, iterations=4, poly_n=5, poly_sigma=1.2, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN):
    if channels == None:
        channels = range(1,len(da_list)+1)
    x_flow_list, y_flow_list = [], []
    for i, da in enumerate(da_list):
        x_flow, y_flow = get_ds_flow(da, direction, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        if channels[i] in [1,3,5]:
            x_flow_list.append(dataset_tools.ds_area_func(np.mean, x_flow, 2, dims=('x','y'), chop=True))
            y_flow_list.append(dataset_tools.ds_area_func(np.mean, y_flow, 2, dims=('x','y'), chop=True))
        elif channels[i] == 2:
            x_flow_list.append(dataset_tools.ds_area_func(np.mean, x_flow, 4, dims=('x','y'), chop=True))
            y_flow_list.append(dataset_tools.ds_area_func(np.mean, y_flow, 4, dims=('x','y'), chop=True))
        else:
            x_flow_list.append(x_flow)
            y_flow_list.append(y_flow)
    dataset_tools.match_coords(x_flow_list)
    dataset_tools.match_coords(y_flow_list)
    x_flow_stack = np.stack(x_flow_list)
    y_flow_stack = np.stack(y_flow_list)
    flow_weights = (x_flow_stack**2 + y_flow_stack**2)**0.5
    x_flow_average = xr.DataArray(np.average(x_flow_stack, axis=0, weights=flow_weights),
                                  dims=['t','y','x'],
                                  coords={'t':x_flow_list[0].t, 'y':x_flow_list[0].y, 'x':x_flow_list[0].x})
    y_flow_average = xr.DataArray(np.average(y_flow_stack, axis=0, weights=flow_weights),
                                  dims=['t','y','x'],
                                  coords={'t':x_flow_list[0].t, 'y':x_flow_list[0].y, 'x':x_flow_list[0].x})
    return x_flow_average, y_flow_average

class Flow_Func(object):
    def __init__(self, flow_x_for, flow_x_back, flow_y_for, flow_y_back):
        self.flow_x_for = flow_x_for
        self.flow_y_for = flow_y_for
        self.flow_x_back = flow_x_back
        self.flow_y_back = flow_y_back
        self.shape = flow_x_for.shape

    def __getitem__(self, items):
        """
        return a subset of the flow vectors
        """
        return Flow_Func(self.flow_x_for[items], self.flow_x_back[items],
                         self.flow_y_for[items], self.flow_y_back[items])

    def __call__(self, t):
        """
        parabolic interpolation of the flow vectors
        """
        return (0.5*t*(t+1)*self.flow_x_for + 0.5*t*(t-1)*self.flow_x_back,
                0.5*t*(t+1)*self.flow_y_for + 0.5*t*(t-1)*self.flow_y_back)


def get_flow_func(field, replace_missing=False, post_iter=0, **kwargs):
    flow_forward = get_ds_flow(field, **kwargs)
    flow_backward = get_ds_flow(field, direction='backwards', **kwargs)
    if replace_missing:
        flow_forward[0][-1], flow_forward[1][-1] = -flow_backward[0][-1], -flow_backward[1][-1]
        flow_backward[0][0], flow_backward[1][0] = -flow_forward[0][0], -flow_forward[1][0]
        wh = np.logical_or(np.isnan(flow_forward[0]),
                           np.isnan(flow_forward[1]))
        if np.any(wh):
            flow_forward[wh] = 0
        wh = np.logical_or(np.isnan(flow_backward[0]),
                           np.isnan(flow_backward[1]))
        if np.any(wh):
            flow_backward[wh] = 0

    flow_x_for = flow_forward[0].to_masked_array()
    flow_y_for = flow_forward[1].to_masked_array()
    flow_x_back = flow_backward[0].to_masked_array()
    flow_y_back = flow_backward[1].to_masked_array()

    flow_func = Flow_Func(flow_x_for, flow_x_back, flow_y_for, flow_y_back)

    # Check that the forward and backward vectors for between two frames are
    # equal and opposite
    if post_iter > 0:
        for i in range(post_iter):
            x_for_interp = -interp_ds_flow(
                xr.DataArray(flow_func.flow_x_back, dims=('t','y','x')),
                *flow_func(1), method='linear').data
            y_for_interp = -interp_ds_flow(
                xr.DataArray(flow_func.flow_y_back, dims=('t','y','x')),
                *flow_func(1), method='linear').data
            x_back_interp = -interp_ds_flow(
                xr.DataArray(flow_func.flow_x_for, dims=('t','y','x')),
                *flow_func(-1), direction='backwards', method='linear').data
            y_back_interp = -interp_ds_flow(
                xr.DataArray(flow_func.flow_y_for, dims=('t','y','x')),
                *flow_func(-1), direction='backwards', method='linear').data
            flow_func.flow_x_for[1:-1] = (flow_func.flow_x_for[1:-1]
                                          + x_for_interp[1:-1]) / 2
            flow_func.flow_y_for[1:-1] = (flow_func.flow_y_for[1:-1]
                                          + y_for_interp[1:-1]) / 2
            flow_func.flow_x_back[1:-1] = (flow_func.flow_x_back[1:-1]
                                           + x_back_interp[1:-1]) / 2
            flow_func.flow_y_back[1:-1] = (flow_func.flow_y_back[1:-1]
                                           + y_back_interp[1:-1]) / 2

    return flow_func

def interp_flow(da, flow, method='linear'):
    x = np.arange(da.shape[-1])
    y = np.arange(da.shape[-2])
    xx, yy = np.meshgrid(x, y)
    new_xx, new_yy = xx + flow[...,0], yy + flow[...,1]
    new_xx = np.minimum(np.maximum(new_xx, 0), x.max())
    new_yy = np.minimum(np.maximum(new_yy, 0), y.max())
    interp_da = xr.apply_ufunc(interpolate.interpn, (y, x), da, (new_yy, new_xx), kwargs={'method':method, 'bounds_error':False, 'fill_value':None})
    return interp_da

def interp_ds_flow(da, x_flow, y_flow, direction='forwards', method='linear'):
    da_flow = xr.zeros_like(da).compute()
    if da_flow.dtype in [np.float16, np.float32, np.float64, np.float128]:
        da_flow *= np.nan
    if direction == 'forwards':
        for i in range(da.coords['t'].size-1):
            da_flow[i] = interp_flow(da[i+1].compute(), np.stack([x_flow[i], y_flow[i]], axis=-1), method=method)
    elif direction == 'backwards':
        for i in range(1,da.coords['t'].size):
            da_flow[i] = interp_flow(da[i-1].compute(), np.stack([x_flow[i], y_flow[i]], axis=-1), method=method)
    else:
        raise ValueError("""keyword 'direction' only accepts 'forwards' or 'backwards' as acceptable values""")
    return da_flow

def get_flow_stack(field, flow_func, method='linear'):
    return xr.concat([interp_ds_flow(field, *flow_func(-1), direction='backwards', method=method),
                      field,
                      interp_ds_flow(field, *flow_func(1), method=method)],
                     dim='t_off')

def test_flow(field, plot=True, return_flow=False, **flow_kwargs):
    field_flow = get_flow_func(field, **flow_kwargs)
    field_stack = get_flow_stack(field, field_flow).data
    residual_mag = np.zeros(field.shape)
    for i in range(field.shape[0]):
        residual_flow = get_flow_func(xr.DataArray(field_stack[1:,i],dims=('t','y','x')),
                                               **flow_kwargs)
        residual_mag[i] = 0.5 * ((residual_flow.flow_x_for[0]**2 + residual_flow.flow_y_for[0]**2)**0.5
                                 + (residual_flow.flow_x_back[0]**2 + residual_flow.flow_y_back[0]**2)**0.5)

    flow_mag = 0.5 * ((field_flow.flow_x_for**2 + field_flow.flow_y_for**2)**0.5
                      + (field_flow.flow_x_back**2 + field_flow.flow_y_back**2)**0.5)

    residual_mag = ma.array(residual_mag, mask=flow_mag.mask)/2**0.5
    relative_mag = residual_mag/flow_mag
    if return_flow:
        return residual_mag, relative_mag, field_flow
    else:
        return residual_mag, relative_mag

def flow_convolve(flow_data, structure=None, wrap=False, function=None, dtype=None, **kwargs):
    if dtype == None:
        dtype=flow_data.dtype
    n_dims = len(flow_data.shape)-1
    assert(n_dims > 0)
    if hasattr(structure, "shape"):
        if len(structure.shape) > n_dims:
            raise ValueError("Input structure has too many dimensions")
        for s in structure.shape:
            if s not in [1,3]:
                raise ValueError("structure input must be an array with dimensions of length 1 or 3")
        if len(structure.shape) < n_dims:
            nd_diff = n_dims - len(structure.shape)
            structure = structure.reshape((1,)*nd_diff+structure.shape)
    else:
        if structure == None:
            structure = np.zeros((3,)*n_dims)
            structure[(1,)*n_dims] = 1
            for i in range(n_dims):
                index = [1]*n_dims
                index[i] = 0
                structure[tuple(index)] = 1
                index[i] = 2
                structure[tuple(index)] = 1
    if n_dims > 2:
        spat = np.meshgrid(*(range(s) for s in flow_data.shape[2:]), indexing='ij')
    elif n_dims == 2:
        spat = (np.arange(flow_data.shape[2]),)
    elif n_dims == 1:
        spat = (np.array([0]),)
    spat_nd = len(spat[0].shape)
    re_shape = (-1,)+(1,)*spat_nd
    tt = np.full_like(spat[0],1)
    wh_struct = structure.ravel()!=0
    multi_struct = structure.ravel()[wh_struct].reshape(re_shape).astype(dtype)
    n_struct = np.sum(wh_struct)
    offsets = [offset-1 for offset
               in np.unravel_index(np.arange(structure.size)[wh_struct], structure.shape)]
    offset_inds = [offsets[0].reshape(re_shape)+tt] + [offsets[i+1].reshape(re_shape)+spat[i]
                                                       for i in range(len(spat))]
    offset_inds_corrected = [offset_inds[i]%maxi for i, maxi in enumerate(flow_data[:,0].shape)]
    if not wrap:
        offset_mask = np.any([offset_inds[i]!=offset_inds_corrected[i] for i in range(len(offset_inds))],0)
    offset_inds_corrected = tuple(offset_inds_corrected)
    inds_shape = offset_inds_corrected[0].shape
    offset_inds_corrected = np.ravel_multi_index(offset_inds_corrected, inds_shape).ravel()

    if function is not None:
        if hasattr(function, '__iter__'):
            n_func = len(function)
            output = ma.empty((n_func,)+flow_data.shape[1:], dtype)
            for i in range(flow_data.shape[1]):
                temp = ma.array(flow_data.transpose([1,0,2,3])[i].ravel()[offset_inds_corrected]).reshape(inds_shape) * multi_struct
                temp.mask = np.logical_or(np.isnan(temp), offset_mask)
                for j in range(n_func):
                    output[j,i] = function[j](temp, 0, **kwargs)
        else:
            output = ma.empty(flow_data.shape[1:], dtype)
            for i in range(flow_data.shape[1]):
                temp = ma.array(flow_data.transpose([1,0,2,3])[i].ravel()[offset_inds_corrected]).reshape(inds_shape) * multi_struct
                temp.mask = np.logical_or(np.isnan(temp), offset_mask)
                output[i] = function(temp, 0, **kwargs)
    else:
        output = ma.empty((n_struct,)+flow_data.shape[1:], dtype)
        for i in range(flow_data.shape[1]):
            temp = ma.array(flow_data.transpose([1,0,2,3])[i].ravel()[offset_inds_corrected]).reshape(inds_shape) * multi_struct
            temp.mask = np.logical_or(np.isnan(temp), offset_mask)
            output[:,i] = temp
    return output

def _checkstruct(structure, n_dims):
    if structure is None:
        structure = ndi.generate_binary_structure(n_dims,1)
    if hasattr(structure, "shape"):
        if len(structure.shape) > n_dims:
            raise ValueError("Input structure has too many dimensions")
        for s in structure.shape:
            if s not in [1,3]:
                raise ValueError("structure input must be an array with dimensions of length 1 or 3")
        if len(structure.shape) < n_dims:
            nd_diff = n_dims - len(structure.shape)
            structure = structure.reshape((1,)*nd_diff+structure.shape)
    else:
        raise ValueError("""structure input must be an array-like object""")

    return structure

def _gen_flow_ravel_inds(flow_func, structure, wrap=False):
    shape = flow_func.shape
    n_dims = len(shape)

    structure_offsets = [arr.reshape((-1,)+(1,)*(n_dims-1))-1 for arr in np.where(structure!=0)]
    whp1 = (structure_offsets[0] == 1)
    whm1 = (structure_offsets[0] == -1)
    n_elements = np.sum(structure!=0)

    shape_ranges = [np.arange(s).reshape(np.roll((-1,)+(1,)*(n_dims-2), i))
                    for i, s in enumerate(shape[1:])]

    flow_inds = [shape_ranges[i] + structure_offsets[i+1] for i in range(n_dims-1)]
    for t in range(shape[0]):
        # Todo: make this generalised for more dimensions
        temp_inds = [None, None]
        temp_inds[0] = (flow_inds[0] + np.round(flow_func.flow_y_for[t]).astype(int)*whp1
                                     + np.round(flow_func.flow_y_back[t]).astype(int)*whm1)
        temp_inds[1] = (flow_inds[1] + np.round(flow_func.flow_x_for[t]).astype(int)*whp1
                                     + np.round(flow_func.flow_x_back[t]).astype(int)*whm1)

        ravelled_index = np.ravel_multi_index([(structure_offsets[0]+t)%shape[0],
                                               temp_inds[0]%shape[1],
                                               temp_inds[1]%shape[2]],
                                              shape).ravel()
        if wrap:
            mask = False
        else:
            mask = sum([(structure_offsets[0]+t)%shape[0] != (structure_offsets[0]+t),
                        (temp_inds[0]%shape[1]) != temp_inds[0],
                        (temp_inds[1]%shape[2]) != temp_inds[1]])

        yield ravelled_index, mask

def flow_convolve_nearest(data, flow_func, structure=None, wrap=False, function=None, dtype=None, debug=False, **kwargs):
    """
    A function to compute a Semi-Lagrangian convolution using the nearest neighbour method. This can be performed
    faster as no interpolation is required.
    Input:
        data:
            An n-dimensional array or array-like input of values to perform the convolution on
        flow_func:
            A lambda function that returns the flow vectors of the data field in n-1 dimensions

    Output:
        convolve_data:
            An output array of convoluted data. If not function keyword is provided, this will be an n+1
            dimension array, where the leading dimension is the same length as the number of non-zero values
            in the provided structure, and the remaining dimensions of the same size as the data input.
            If the function keyword is defined, this will be an array of the same shape as the input data

    Optional:
        structure:
            An array-like structure to apply to the convolution. By default this is set to square connectivity.
            This must have n or fewer dimensions, and each dimension length must be 3 or 1. The value of the
            convolution output will be multiplied by the correspinding structure values.
        wrap:
            If true then any points in the convolution which exceed the limits of the relevant dimenion will be
            wrapped around to the other side of the array. Otherwise these points will be masked. Defualts to False
        function:
            A function to apply to the convoluted data along to convolution dimension. This function must have an
            axis keyword, will will be set to 0
        dtype:
            Data type of the returned array. Defaults to the dtype of the input data
        **kwargs:
            Keywords passed to any function called by the function keyword.
    """
    if dtype == None:
        dtype = data.dtype
    n_dims = len(data.shape)
    assert(n_dims > 1)

    structure = _checkstruct(structure, n_dims)
    structure_factor = structure[structure!=0].reshape((-1,)+(1,)*(n_dims-1))
    n_elements = np.sum(structure!=0)

    inds_gen = _gen_flow_ravel_inds(flow_func, structure, wrap=wrap)
    if function is None:
        out_arr = ma.empty((n_elements,)+data.shape, dtype=dtype)
    else:
        out_arr = ma.empty(data.shape, dtype=dtype)

    for t in range(data.shape[0]):
        ravelled_index, mask = next(inds_gen)
        temp = ma.array(data.ravel()[ravelled_index].reshape((n_elements,)+data.shape[1:]) * structure_factor,
                        mask=mask, dtype=data.dtype)
        if function is None:
            out_arr[:,t] = temp
        else:
            out_arr[t] = function(temp, 0, **kwargs)

    return out_arr

def flow_argmin_nearest(data, argmin, flow_func, structure=None, dtype=None):
    """
    A function to find the data at the locations provided by an argmin of the convolved field using nearest
    neighbour method
    Input:
        data:
            An n-dimensional array or array-like input of values to perform the convolution on
        argmin:
            An n-dimensional array or array-like input of values of the argmin of the convolution of the
            field.
        flow_func:
            A lambda function that returns the flow vectors of the data field in n-1 dimensions

    Output:
        convolve_data:
            An output array of convoluted data. If not function keyword is provided, this will be an n+1
            dimension array, where the leading dimension is the same length as the number of non-zero values
            in the provided structure, and the remaining dimensions of the same size as the data input.
            If the function keyword is defined, this will be an array of the same shape as the input data

    Optional:
        structure:
            An array-like structure to apply to the convolution. By default this is set to square connectivity.
            This must have n or fewer dimensions, and each dimension length must be 3 or 1. The value of the
            convolution output will be multiplied by the correspinding structure values.
        dtype:
            Data type of the returned array. Defaults to the dtype of the input data
    """
    if dtype == None:
        dtype = data.dtype
    n_dims = len(data.shape)
    assert(n_dims > 1)

    structure = _checkstruct(structure, n_dims)

    shape_arrays = np.meshgrid(*(np.arange(s, dtype=int) for s in argmin.shape[1:]), indexing='ij')

    out_arr = np.empty(argmin.shape, dtype=dtype)

    for t in range(argmin.shape[0]):
        argmin_offsets = [wh[argmin[t]]-1 for wh in np.where(structure!=0)]

        whp1 = (argmin_offsets[0] == 1)
        whm1 = (argmin_offsets[0] == -1)

        argmin_offsets[0] += t
        argmin_offsets[1] += (np.round(flow_func.flow_y_for[t]).astype(int)*whp1
                              + np.round(flow_func.flow_y_back[t]).astype(int)*whm1
                              + shape_arrays[0])
        argmin_offsets[2] += (np.round(flow_func.flow_x_for[t]).astype(int)*whp1
                              + np.round(flow_func.flow_x_back[t]).astype(int)*whm1
                              + shape_arrays[1])

        ravelled_index = np.ravel_multi_index([argmin_offsets[0]%data.shape[0],
                                               argmin_offsets[1]%data.shape[1],
                                               argmin_offsets[2]%data.shape[2]],
                                               data.shape).ravel()

        out_arr[t] = data.ravel()[ravelled_index].reshape(out_arr.shape[1:])

    return out_arr

def flow_local_min(flow_stack, structure=None, ignore_nan=False):
    if structure is not None:
        mp = structure.size//2
    else:
        mp = 3
    if ignore_nan:
        return (flow_convolve(flow_stack, structure=structure, function=np.nanmin)==flow_stack[1])
    else:
        return (flow_convolve(flow_stack, structure=structure, function=np.min)==flow_stack[1])

def get_sobel_matrix(ndims):
    sobel_matrix = np.array([-1,0,1])
    for i in range(ndims-1):
        sobel_matrix = np.multiply.outer(np.array([1,2,1]), sobel_matrix)
    return sobel_matrix

def flow_sobel(flow_stack, axis=None, direction=None, magnitude=False):
    # temp_convolve = flow_convolve(flow_stack, structure=np.ones([3,3,3]))
    nd = len(flow_stack.shape)-1
    output = []
    if axis is None:
        axis = range(nd)
    if not hasattr(axis, '__iter__'):
        axis = [axis]
    if direction is None:
        if magnitude:
            def temp_sobel_func(temp, ax, counter=[0]):
                output = ma.zeros(temp.shape[1:])
                for i in axis:
                    sobel_matrix = np.transpose(get_sobel_matrix(3),
                               np.roll(np.arange(3),(1+i)%3)).ravel().reshape((-1,1,1)).astype(temp.dtype)
                    output += np.sum((temp-flow_stack[1][counter[0]]) * sobel_matrix, 0)**2
                counter[0]+=1
                return output**0.5
            output = flow_convolve(flow_stack, structure=np.ones([3,3,3]),
                                   function=temp_sobel_func)
        else:
            for i in axis:
                def temp_sobel_func(temp, axis, counter=[0]):
                    sobel_matrix = np.transpose(get_sobel_matrix(3),
                                   np.roll(np.arange(3),(1+i)%3)).ravel().reshape((-1,1,1)).astype(temp.dtype)
                    output = np.sum((temp-flow_stack[1][counter[0]]) * sobel_matrix, axis)
                    counter[0]+=1
                    return output

                output.append(flow_convolve(flow_stack, structure=np.ones([3,3,3]),
                                            function=temp_sobel_func))

    elif direction == 'uphill':
        if magnitude:
            def temp_sobel_func(temp, ax, counter=[0]):
                output = ma.zeros(temp.shape[1:])
                for i in axis:
                    sobel_matrix = np.transpose(get_sobel_matrix(3),
                               np.roll(np.arange(3),(1+i)%3)).ravel().reshape((-1,1,1)).astype(temp.dtype)
                    output += np.sum(np.maximum(temp-flow_stack[1][counter[0]], 0)
                                     * sobel_matrix, 0)**2
                counter[0]+=1
                return output**0.5
            output = flow_convolve(flow_stack, structure=np.ones([3,3,3]),
                                   function=temp_sobel_func)
        else:
            for i in axis:
                def temp_sobel_func(temp, axis, counter=[0]):
                    sobel_matrix = np.transpose(get_sobel_matrix(3),
                                   np.roll(np.arange(3),(1+i)%3)).ravel().reshape((-1,1,1))
                    output = np.sum(np.maximum(temp-flow_stack[1][counter[0]], 0)
                                    * sobel_matrix, axis)
                    counter[0]+=1
                    return output

                output.append(flow_convolve(flow_stack, structure=np.ones([3,3,3]),
                                            function=temp_sobel_func))

    elif direction == 'downhill':
        if magnitude:
            def temp_sobel_func(temp, ax, counter=[0]):
                output = ma.zeros(temp.shape[1:])
                for i in axis:
                    sobel_matrix = np.transpose(get_sobel_matrix(3),
                               np.roll(np.arange(3),(1+i)%3)).ravel().reshape((-1,1,1)).astype(temp.dtype)
                    output += np.sum(np.minimum(temp-flow_stack[1][counter[0]], 0)
                                     * sobel_matrix, 0)**2
                counter[0]+=1
                return output**0.5
            output = flow_convolve(flow_stack, structure=np.ones([3,3,3]),
                                   function=temp_sobel_func)
        else:
            for i in axis:
                def temp_sobel_func(temp, axis, counter=[0]):
                    sobel_matrix = np.transpose(get_sobel_matrix(3),
                                   np.roll(np.arange(3),(1+i)%3)).ravel().reshape((-1,1,1))
                    output = np.sum(np.minimum(temp-flow_stack[1][counter[0]], 0)
                                    * sobel_matrix, axis)
                    counter[0]+=1
                    return output

                output.append(flow_convolve(flow_stack, structure=np.ones([3,3,3]),
                                            function=temp_sobel_func))
    else:
        raise ValueError("""direction must be 'uphill', 'downhill' or None""")
    return output

def flow_gradient_watershed(flow_stack, flow_func, markers, mask=None, max_iter=100, max_no_progress=10, expand_mask=True):
    import numpy as np
    from scipy import interpolate
    import scipy.ndimage as ndi
    from skimage.feature import peak_local_max
    import warnings
    shape = [np.arange(shape) for shape in flow_stack.shape[1:]]
    grids = np.stack(np.meshgrid(*shape, indexing='ij'),-1)
    grads = np.stack([np.concatenate([np.gradient(-flow_stack[1:,0], axis=0)[0][np.newaxis],
                  np.gradient(-flow_stack[:,1:-1], axis=0)[1],
                  np.gradient(-flow_stack[:-1,-1], axis=0)[1][np.newaxis]], 0)]
                 + [np.gradient(-flow_stack, axis=i)[1] for i in range(2, len(flow_stack.shape))], -1)
    grads_mag = (np.sum(grads**2,-1)**0.5)
    wh_mag = grads_mag!=0
    new_grads = grads.copy()
    new_grads[wh_mag] /= grads_mag[wh_mag][...,np.newaxis]
    pos = grids.astype(float)+new_grads
    if pos.shape[-1] == 2:
        pos[...,1] += flow_func(new_grads[...,0])
    else:
        pos[...,1:] += np.stack(flow_func(new_grads[...,0]), -1)
    pos = np.maximum(np.minimum(pos, (np.array(flow_stack.shape[1:])-1)), 0)

    local_min = flow_local_min(flow_stack)
    max_markers = np.nanmax(markers)
    new_markers = ndi.label(np.logical_or(local_min, np.logical_not(wh_mag))*(markers==0), structure=np.ones([3]*len(shape)))[0]
    new_markers[new_markers>0]+=max_markers
    fill_markers = markers+new_markers
    if mask is not None:
        fill_markers[mask] = -1
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
            new_flow =  lambda t : (0.5*t*(t+1)*interpolate.interpn(shape, flow_func(1)[0], pos[wh], method='linear')
                        + 0.5*t*(t-1)*interpolate.interpn(shape, flow_func(-1)[0], pos[wh], method='linear'),
                        0.5*t*(t+1)*interpolate.interpn(shape, flow_func(1)[1], pos[wh], method='linear')
                        + 0.5*t*(t-1)*interpolate.interpn(shape, flow_func(-1)[1], pos[wh], method='linear'))
            if new_grads.shape[-1] == 2:
                new_grads[...,1] += new_flow(new_grads[...,0])
            else:
                new_grads[...,1:] += np.stack(new_flow(new_grads[...,0]), -1)
            pos[wh] += new_grads
            pos = np.maximum(np.minimum(pos, (np.array(flow_stack.shape[1:])-1)), 0)
    else:
        warnings.warn('Reached maximum iterations without completion. Remaining unfilled pixels = '+str(n_to_fill))

    for i in np.unique(fill[fill>max_markers]):
        wh_i = fill==i
        edges = np.logical_and(ndi.morphology.binary_dilation(wh_i), np.logical_not(wh_i))
        if expand_mask:
            wh = edges*(fill!=0)
        else:
            wh = edges*(fill>0)
        if np.any(wh):
            min_label = fill[wh][np.argmin(flow_stack[1][wh])]
        else:
            min_label = 0
        fill[wh_i] = min_label
    print(step)
    return np.maximum(fill,0)

def flow_network_watershed(field, markers, flow_func, mask=None, structure=None, max_iter=100, debug_mode=False, low_memory=False):
    # Check structure input, set default and check dimensions and shape
    if structure is None:
        if debug_mode:
            print("Setting structure to default")
        structure = ndi.generate_binary_structure(3,1)
    if len(structure.shape)<3:
        if debug_mode:
            print("Setting structure to 3d")
        structure = np.atleast_3d(structure)
    if np.any([s not in [1,3] for s in structure.shape]):
        raise Exception("Structure must have a size of 1 or 3 in each dimension")
    if np.any([s != 3 for s in structure.shape]):
        if debug_mode:
            print("Inserting structure into 3x3x3 array")
        wh = [slice(0,3) if s==3 else slice(1,2) for s in structure.shape]
        temp = np.zeros([3,3,3])
        temp[wh] = structure
        structure=temp
    if isinstance(structure, ma.core.MaskedArray):
        structure = structure.filled(fill_value=0)
    structure = structure.astype('bool')

    # Check mask input
    if mask is None:
        if debug_mode:
            print("Setting mask to default")
        mask = np.zeros_like(field, dtype='bool')
    if isinstance(mask, ma.core.MaskedArray):
        mask = mask.filled(fill_value=True)

    # Check markers input
    if isinstance(markers, ma.core.MaskedArray):
        markers = markers.filled(fill_value=False)

    # Check field input
    if isinstance(field, ma.core.MaskedArray):
        field = field.filled(fill_value=np.nanmax(field))
    wh = np.isnan(field)
    if np.any(wh):
        field[wh] = np.nanmax(field)
        mask[wh] = True
        markers[wh] = False

    # Get ravelled indices for each pixel in the field, and find nearest neighbours using flow field
    # Set inds dtype to minimum possible to contain all values to save memory
    if field.size<np.iinfo(np.uint16).max:
        inds_dtype = np.uint16
    elif field.size<np.iinfo(np.uint32).max:
        inds_dtype = np.uint32
    else:
        inds_dtype = np.uint64
    inds = np.arange(field.size, dtype=inds_dtype).reshape(field.shape)
    if debug_mode:
        print("Calculating nearest neighbours")
    # Now using the more efficient flow_convolve_nearest function:
    min_convolve = flow_convolve_nearest(field, flow_func,
                                         structure=structure, function=ma.argmin,
                                         dtype=np.uint8)
    min_convolve = np.minimum(np.maximum(min_convolve, 0), np.sum(structure!=0).astype(np.uint8)-1)
    inds_neighbour = flow_argmin_nearest(inds, min_convolve, flow_func,
                                         structure=structure,
                                         dtype=inds_dtype)
    del min_convolve
    # inds_neighbour = inds_convolve[tuple([min_convolve.data.astype(int)]+np.meshgrid(*(range(s) for s in inds.shape), indexing='ij'))].astype(int)
    if hasattr(inds_neighbour, "mask"):
        wh = np.logical_or(np.logical_or(inds_neighbour.data<0, inds_neighbour.data>inds.max()), inds_neighbour.mask)
        if np.any(wh):
            inds_neighbour.data[wh] = inds[wh]
    else:
        wh = np.logical_or(inds_neighbour<0, inds_neighbour>inds.max())
        if np.any(wh):
            inds_neighbour[wh] = inds[wh]
    inds_neighbour = inds_neighbour.astype(inds_dtype)
    # Now iterate over neighbour network to find minimum convergence point for each pixel
        # Each pixel will either reach a minimum or loop back to itself
    if markers.max()<np.iinfo(np.int16).max:
        mark_dtype = np.int16
    elif markers.max()<np.iinfo(np.int32).max:
        mark_dtype = np.int32
    else:
        mark_dtype = np.int64
    fill_markers = (markers.astype(mark_dtype)-mask.astype(mark_dtype))
    wh_local_min = np.logical_and(inds_neighbour==inds, fill_markers==0)
    wh_markers = np.logical_or(wh_local_min, fill_markers!=0)
    wh_to_fill = np.logical_not(wh_markers.copy())
    if debug_mode:
        print("Finding network convergence locations")
        print("Pixels to fill:", np.sum(wh_to_fill))
    for i in range(max_iter):
        inds_neighbour[wh_to_fill] = inds_neighbour.ravel()[inds_neighbour[wh_to_fill].ravel()]
        # Check if any pixels have looped back to their original location
        wh_loop = np.logical_and(wh_to_fill, inds_neighbour==inds)
        if np.any(wh_loop):
            if debug_mode:
                print('Loop')
            wh_to_fill[wh_loop] = False
            wh_local_min[wh_loop] = True
            wh_markers[wh_loop] = True

        # Now check if any have met a convergence location
        wh_converge = wh_markers.ravel()[inds_neighbour[wh_to_fill]].ravel()
        if np.any(wh_converge):
            if debug_mode:
                print('Convergence')
            wh_to_fill[wh_to_fill] = np.logical_not(wh_converge)

        if debug_mode:
            print("Iteration:", i+1)
            print("Pixels converged", np.sum(np.logical_not(wh_to_fill)))
        if not np.any(wh_to_fill):
            if debug_mode:
                print("All pixels converged")
            break
    # del old_neighbour
    # Use converged locations to fill watershed basins
    if debug_mode:
        print("Filling basins")
    # wh = np.logical_and(type_converge==1, np.logical_not(np.logical_xor(markers!=0, mask)))
    max_markers = np.nanmax(markers)
    temp_markers = ndi.label(wh_local_min)[0][wh_local_min]+max_markers
    if temp_markers.max()<np.iinfo(np.int16).max:
        mark_dtype = np.int16
    elif temp_markers.max()<np.iinfo(np.int32).max:
        mark_dtype = np.int32
    else:
        mark_dtype = np.int64
    fill_markers = fill_markers.astype(mark_dtype)
    fill_markers[wh_local_min] = temp_markers
    fill = fill_markers.copy()
    wh = fill==0
    fill[wh] = fill.ravel()[inds_neighbour[wh].ravel()]
    # fill = fill_markers.ravel()[inds_neighbour.ravel()].reshape(fill_markers.shape)
    del fill_markers, temp_markers, inds_neighbour
    # fill[markers>0]=markers[markers>0]
    # fill[mask]=-1
    wh = fill==0
    if np.any(wh):
        if debug_mode:
            print("Some pixels not filled, adding")
        fill[wh] = ndi.label(wh)[0][wh]+np.nanmax(fill)
    # Now we've filled all the values, we change the mask values back to 0 for the next step
    if isinstance(fill, ma.core.MaskedArray):
        fill = np.maximum(fill.filled(fill_value=0),0)
    else:
        fill = np.maximum(fill, 0)
    # Now overflow watershed basins into neighbouring basins until only marker labels are left
    if debug_mode:
        print("Joining labels")
        print("Max label:", np.nanmax(fill))
        print("max_markers:", max_markers.astype(int))
    # we can set the middle value of the structure to 0 as we are only interested in the surrounding pixels
    new_struct = structure.copy()
    new_struct[1,1,1] = 0
    for iter in range(1, max_iter+1):
        # Make a flow stack using the current fill
        fill_gen = (fill[t] for t in range(fill.shape[0]))
        def _fill_mask_argmin(temp, axis):
            temp.mask = np.logical_or(temp.mask, temp==next(fill_gen))
            return ma.array(np.argmin(temp, 0), mask=np.all(temp.mask, 0), dtype=bool)

        flow_gen = enumerate(_gen_flow_ravel_inds(flow_func, new_struct, wrap=False))
        # Temporary function that masks the values with the same fill as the origin point
        def fill_mask_argmin(temp, axis):
            t, inds_gen = next(flow_gen)
            ravelled_index, mask = inds_gen
            temp_fill = ma.array(fill.ravel()[ravelled_index].reshape(temp.shape),
                                 mask=mask, dtype=fill.dtype)
            temp.mask = np.logical_or(temp.mask, fill[t]==temp_fill)
            return ma.array(np.argmin(temp, axis), mask=np.all(temp.mask, axis), dtype=np.uint8)

        argmin_edge = flow_convolve_nearest(field, flow_func,
                                            structure=new_struct,
                                            function=fill_mask_argmin,
                                            dtype=np.uint8)
        min_edge = flow_argmin_nearest(field, argmin_edge.filled(fill_value=0),
                                       flow_func, structure=new_struct)
        min_edge = ma.array(min_edge, mask=argmin_edge.mask)
        inds_edge = flow_argmin_nearest(inds, argmin_edge.filled(fill_value=0),
                                       flow_func, structure=new_struct,
                                       dtype=inds_dtype)
        # New method using bincount and argsort:
        # We add 1 to the fill so that the first value is 0, this allows to index from i:i+1 for all fill values
        region_bins = np.nancumsum(np.bincount(fill.ravel()+1))
        n_bins = region_bins.size-1
        region_inds = np.argsort(fill.ravel())
        def get_new_label(j):
            wh = region_inds[region_bins[j]:region_bins[j+1]]
            # Occasionally a region won't be able to find a neighbour, in this case we set it to masked
            if wh.size>0:
                if np.all(min_edge.mask.ravel()[wh]):
                    return 0
                else:
                    output = fill.ravel()[inds_edge.ravel()[wh][np.nanargmin(np.maximum(min_edge.ravel()[wh], field.ravel()[wh]))]]
            else:
                return 0
            # Now need to check if output is masked
            if (type(output) is ma.MaskedArray and output.mask):
                raise ValueError("Output label is masked!")
            else:
                output = output.item()
            # and check if nan
            if np.all(np.isfinite(output)):
                assert output != j
                return output
            else:
                raise ValueError("Output label is not finite!")
        new_label = ma.array(list(range(max_markers+1))
                             + [get_new_label(k) if region_bins[k] > 0 else 0 for k in range(max_markers+1, n_bins)],
                             dtype=mark_dtype)
        new_label.fill_value=0
        new_label=new_label.filled()
        # new_label=np.minimum(new_label, np.arange(new_label.size, dtype=mark_dtype))
        for jiter in range(1,max_iter+1):
            wh = new_label[max_markers+1:]>max_markers
            new = np.minimum(new_label, new_label[new_label])[max_markers+1:][wh]
            if np.all(new_label[max_markers+1:][wh]==new):
                break
            else:
                new_label[max_markers+1:][wh] = new
        for k in range(max_markers+1, n_bins):
            if region_bins[k]<region_bins[k+1]:
                fill.ravel()[region_inds[region_bins[k]:region_bins[k+1]]] = new_label[k]
        if debug_mode:
            print("Iteration:", iter)
            print("Remaining labels:", np.unique(fill).size)
            # print("Max label:", np.nanmax(fill))
            # if np.unique(fill).size<=10:
            #     print("Labels:", np.unique(fill))
            #     print("New:", new_label[np.maximum(0,np.unique(fill).astype(int))])
            # else:
            #     print("Labels:", np.unique(fill)[:10])
            #     print("New:", new_label[np.maximum(0,np.unique(fill).astype(int)[:10])])
        if np.nanmax(fill)<=max_markers:
            break
    return fill
