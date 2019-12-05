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

def get_flow_func(field, replace_missing=False, **kwargs):
    flow_forward = get_ds_flow(field, **kwargs)
    flow_backward = get_ds_flow(field, direction='backwards', **kwargs)
    if replace_missing:
        flow_forward[0][-1], flow_forward[1][-1] = -flow_backward[0][-1], -flow_backward[1][-1]
        flow_backward[0][0], flow_backward[1][0] = -flow_forward[0][0], -flow_forward[1][0]
        wh = np.logical_or(np.isnan(flow_forward[0]), np.isnan(flow_forward[1]))
        if np.any(wh):
            flow_forward[wh] = 0
        wh = np.logical_or(np.isnan(flow_backward[0]), np.isnan(flow_backward[1]))
        if np.any(wh):
            flow_backward[wh] = 0
    return lambda t : (0.5*t*(t+1)*flow_forward[0].to_masked_array() + 0.5*t*(t-1)*flow_backward[0].to_masked_array(),
                       0.5*t*(t+1)*flow_forward[1].to_masked_array() + 0.5*t*(t-1)*flow_backward[1].to_masked_array())

def interp_flow(da, flow, method='linear'):
    x = np.arange(da.coords['x'].size)
    y = np.arange(da.coords['y'].size)
    xx, yy = np.meshgrid(x, y)
    new_xx, new_yy = xx + flow[...,0], yy + flow[...,1]
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
    if function is not None:
        if hasattr(function, '__iter__'):
            n_func = len(function)
            output = ma.empty((n_func,)+flow_data.shape[1:], dtype)
            for i in range(flow_data.shape[1]):
                temp = ma.array(flow_data[:,i][offset_inds_corrected])*multi_struct
                temp.mask = np.logical_or(np.isnan(temp), offset_mask)
                for j in range(n_func):
                    output[j,i] = function[j](temp, 0, **kwargs)
        else:
            output = ma.empty(flow_data.shape[1:], dtype)
            for i in range(flow_data.shape[1]):
                temp = ma.array(flow_data[:,i][offset_inds_corrected])*multi_struct
                temp.mask = np.logical_or(np.isnan(temp), offset_mask)
                output[i] = function(temp, 0, **kwargs)
    else:
        output = ma.empty((n_struct,)+flow_data.shape[1:], dtype)
        for i in range(flow_data.shape[1]):
            temp = ma.array(flow_data[:,i][offset_inds_corrected])*multi_struct
            temp.mask = np.logical_or(np.isnan(temp), offset_mask)
            output[:,i] = temp
    return output

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
        structure = np.ones([3,3,3], 'bool')
        structure[0,0,0], structure[0,0,-1], structure[0,-1,0], structure[0,-1,-1], structure[-1,0,0], structure[-1,0,-1], structure[-1,-1,0], structure[-1,-1,-1] = 0,0,0,0,0,0,0,0
    if len(structure.shape)<3:
        if debug_mode:
            print("Setting structure to 3d")
        structure = np.atleast_3d(structure)
    if np.any([s not in [1,3] for s in structure.shape]):
        raise Exception("Structure must have a size of 1 or 3 in each dimension")
    if mask is None:
        if debug_mode:
            print("Setting mask to default")
        mask = np.zeros_like(field, dtype='bool')
    if hasattr(field, 'mask'):
        mask = np.logical_or(mask, field.mask)
    if np.any(np.isnan(field)):
        mask = np.logical_or(mask, np.isnan(field))
    if np.any([s != 3 for s in structure.shape]):
        if debug_mode:
            print("Inserting structure into 3x3x3 array")
        wh = [slice(0,3) if s==3 else slice(1,2) for s in structure.shape]
        temp = np.zeros([3,3,3])
        temp[wh] = structure
        structure=temp
    structure = structure.astype('bool')
    # Get ravelled indices for each pixel in the field, and find nearest neighbours using flow field
    # Set inds dtype to minimum possible to contain all values to save memory
    if field.size<np.iinfo(np.uint16).max:
        inds_dtype = np.uint16
    elif field.size<np.iinfo(np.uint32).max:
        inds_dtype = np.uint32
    else:
        inds_dtype = np.uint64
    inds = np.arange(field.size, dtype=inds_dtype).reshape(field.shape)
    if not low_memory:
        if debug_mode:
            print("Calculating field stack")
        field_stack = get_flow_stack(xr.DataArray(field, dims=('t','y','x')),
                                     flow_func, method='nearest').to_masked_array()
        if debug_mode:
            print("Calculating indices stack")
        ind_stack = get_flow_stack(xr.DataArray(inds, dims=('t','y','x')),
                               flow_func, method='nearest').to_masked_array()
    # Find index of the smallest neighbour to each pixel in the field
    if debug_mode:
        print("Calculating nearest neighbours")
    if low_memory:
        min_convolve = flow_convolve(get_flow_stack(xr.DataArray(field, dims=('t','y','x')),
                                     flow_func, method='nearest').to_masked_array(),
                                     structure=structure, function=np.nanargmin,
                                     dtype=np.uint8)
    else:
        min_convolve = flow_convolve(field_stack,
                                     structure=structure, function=np.nanargmin,
                                     dtype=np.uint8)
    min_convolve = np.minimum(np.maximum(min_convolve, 0), np.sum(structure!=0).astype(np.uint8)-1)
    # inds_convolve = flow_convolve(ind_stack, structure=structure)
    # inds_convolve.mask = np.logical_or(inds_convolve.mask, inds_convolve<0)
    def min_inds_func(inds_convolve, axis, counter=[0]):
        inds_convolve.mask = np.logical_or(inds_convolve.mask, inds_convolve<0)
        inds_neighbour = inds_convolve[tuple([min_convolve[counter[0]]]
                                             + np.meshgrid(*(np.arange(s, dtype=inds_dtype)
                                                             for s in inds.shape[1:]),
                                             indexing='ij'))]
        counter[0]+=1
        return inds_neighbour

    if low_memory:
        inds_neighbour = flow_convolve(get_flow_stack(xr.DataArray(inds, dims=('t','y','x')),
                                       flow_func, method='nearest').to_masked_array(),
                                       structure=structure, function=min_inds_func,
                                       dtype=inds_dtype)
    else:
        inds_neighbour = flow_convolve(ind_stack,
                                       structure=structure, function=min_inds_func,
                                       dtype=inds_dtype)
    del min_convolve
    # inds_neighbour = inds_convolve[tuple([min_convolve.data.astype(int)]+np.meshgrid(*(range(s) for s in inds.shape), indexing='ij'))].astype(int)
    wh = np.logical_or(np.logical_or(inds_neighbour.data<0, inds_neighbour.data>inds.max()), inds_neighbour.mask)
    if np.any(wh):
        inds_neighbour.data[wh] = inds[wh]
    inds_neighbour.fill_value=0
    # Now iterate over neighbour network to find minimum convergence point for each pixel
        # Each pixel will either reach a minimum or loop back to itself
    type_converge = np.zeros(inds_neighbour.shape, np.uint8)
    # iter_converge = np.zeros(inds_neighbour.shape, np.uint8)
    # ind_converge = np.zeros(inds_neighbour.shape, inds_dtype)
    # fill_markers = markers.astype(ind_stack)-mask.astype(int)
    print(type(markers), markers.dtype)
    max_markers = np.nanmax(markers)
    if debug_mode:
        print("Finding network convergence locations")
    for i in range(max_iter):
        old_neighbour, inds_neighbour = inds_neighbour.copy(), inds_neighbour.ravel()[inds_neighbour.ravel()].reshape(inds_neighbour.shape)
        # wh_mark = np.logical_and(type_converge==0, fill_markers.ravel()[inds_neighbour.ravel().reshape(fill_markers.shape)]!=0)
        # if np.any(wh_mark):
        #     type_converge[wh_mark] = 3
        #     iter_converge[wh_mark] = i
        #     ind_converge[wh_mark] = inds_neighbour[wh_mark]
        wh_ind = np.logical_and(type_converge==0, inds_neighbour==inds)
        if np.any(wh_ind):
            type_converge[wh_ind] = 1
            # iter_converge[wh_ind] = i
            # ind_converge[wh_ind] = inds_neighbour[wh_ind]
        wh_conv = np.logical_and(type_converge==0, inds_neighbour==old_neighbour)
        if np.any(wh_conv):
            type_converge[wh_conv] = 2
            # iter_converge[wh_conv] = i
            # ind_converge[wh_ind] = inds_neighbour[wh_ind]
        if debug_mode:
            print("Iteration:", i+1)
            print("Pixels converged", np.sum(type_converge!=0))
        if np.all(type_converge!=0):
            if debug_mode:
                print("All pixels converged")
            break
    del old_neighbour
    # Use converged locations to fill watershed basins
    if debug_mode:
        print("Filling basins")
    wh = np.logical_and(type_converge==1, np.logical_not(np.logical_xor(markers!=0, mask)))
    temp_markers = ndi.label(wh)[0][wh]+max_markers
    if temp_markers.max()<np.iinfo(np.int16).max:
        mark_dtype = np.int16
    elif temp_markers.max()<np.iinfo(np.int32).max:
        mark_dtype = np.int32
    else:
        mark_dtype = np.int64
    fill_markers = (markers.astype(mark_dtype)-mask.astype(mark_dtype))
    fill_markers[wh] = temp_markers
    # fill = fill_markers.ravel()[ind_converge.ravel()].reshape(fill_markers.shape)
    fill = fill_markers.ravel()[inds_neighbour.ravel()].reshape(fill_markers.shape)
    del fill_markers, temp_markers, type_converge, inds_neighbour
    fill[markers>0]=markers[markers>0]
    fill[mask]=-1
    wh = fill==0
    if np.any(wh):
        if debug_mode:
            print("Some pixels not filled, adding")
        fill[wh] = ndi.label(wh)[0][wh]+np.nanmax(fill)
    # Now we've filled all the values, we change the mask values back to 0 for the next step
    fill = np.maximum(fill,0)
    # Now overflow watershed basins into neighbouring basins until only marker labels are left
    if debug_mode:
        print("Joining labels")
        print("max_markers:", max_markers)
    # we can set the middle value of the structure to 0 as we are only interested in the surrounding pixels
    new_struct = structure.copy()
    new_struct[1,1] = 0
    for iter in range(1, max_iter+1):
        # Make a flow stack using the current fill
        temp_fill = get_flow_stack(xr.DataArray(fill, dims=('t','y','x')),
                                                flow_func, method='nearest').to_masked_array()

        # Function to find the minimum neighbour value with a different label
        def min_edge_func(temp, axis, counter=[0]):
            fill_wh = flow_convolve(temp_fill[:,counter[0]].reshape((3,1)+fill.shape[1:]),
                                                   structure=new_struct) == fill[counter[0]]
            fill_wh_mask = np.logical_or(fill_wh.data, fill_wh.mask)
            temp.mask = np.logical_or(temp.mask, fill_wh_mask.squeeze())
            output = np.nanmin(temp, axis)
            counter[0]+=1
            return output
        if low_memory:
            min_edge, argmin_edge = flow_convolve(get_flow_stack(xr.DataArray(field, dims=('t','y','x')),
                                     flow_func, method='nearest').to_masked_array(),
                                     structure=new_struct,
                                     function=[min_edge_func, np.nanargmin],
                                     dtype=np.float32)
        else:
            min_edge, argmin_edge = flow_convolve(field_stack,
                                     structure=new_struct,
                                     function=[min_edge_func, np.nanargmin],
                                     dtype=np.float32)
        # Note that we can call nanargmin directly the second time, as the mask changes have already been made by the temporary function
        # Function to find the offset of the minimum neighbour with a different label
        # def argmin_edge_func(temp, axis, counter=[0]):
        #     fill_wh = flow_convolve(temp_fill[:,counter[0]].reshape((3,1)+fill.shape[1:]),
        #                                            structure=structure) == fill[counter[0]]
        #     fill_wh_mask = np.logical_or(fill_wh.data, fill_wh.mask)
        #     temp.mask = np.logical_or(temp.mask, fill_wh_mask.squeeze())
        #     output = np.nanargmin(temp, axis)
        #     counter[0]+=1
        #     return output
        # if low_memory:
        #     argmin_edge = flow_convolve(get_flow_stack(xr.DataArray(field, dims=('t','y','x')),
        #                                 flow_func, method='nearest').to_masked_array(),
        #                                 structure=structure,
        #                                 function=argmin_edge_func,#[min_edge_func, argmin_edge_func],
        #                                 dtype=np.uint8)
        # else:
        #     argmin_edge = flow_convolve(field_stack,
        #                                 structure=structure,
        #                                 function=argmin_edge_func,
        #                                 dtype=np.uint8)

        def min_inds_func(inds_convolve, axis, counter=[0]):
            inds_convolve.mask = np.logical_or(inds_convolve.mask, inds_convolve<0)
            inds_neighbour = inds_convolve[tuple([argmin_edge[counter[0]].data.astype(inds_dtype)]
                                                 + np.meshgrid(*(np.arange(s, dtype=inds_dtype)
                                                                 for s in inds.shape[1:]),
                                                 indexing='ij'))]
            counter[0]+=1
            return inds_neighbour

        if low_memory:
            inds_edge = flow_convolve(get_flow_stack(xr.DataArray(inds, dims=('t','y','x')),
                                      flow_func, method='nearest').to_masked_array(),
                                      structure=new_struct, function=min_inds_func,
                                      dtype=inds_dtype)
        else:
            inds_edge = flow_convolve(ind_stack,
                                      structure=new_struct, function=min_inds_func,
                                      dtype=inds_dtype)
        # inds_edge = inds_convolve[tuple([argmin_edge.data.astype(int)]+np.meshgrid(*(range(s) for s in inds.shape), indexing='ij'))].astype(int)
        # Old, slow method
        # object_slices=ndi.find_objects(np.maximum(fill,0))
        # for j in range(max_markers, len(object_slices)):
        #     if object_slices[j] is not None:
        #         wh = fill[object_slices[j]]==j+1
        #         argmin = np.maximum(np.minimum(np.nanargmin(np.maximum(min_edge, field)[object_slices[j]][wh]),
        #                                        np.sum(wh)), 0).astype(inds_dtype)
        #         try:
        #             new_label = fill.ravel()[inds_edge[object_slices[j]][wh][argmin]]
        #         except:
        #             if debug_mode:
        #                 print('Failed to assign new_label, label:',j+1)
        #             new_label = -1
        #         if new_label<=j:
        #             fill[object_slices[j]][wh] = new_label

        # New method using bincount and argsort:
        # We add 1 to the fill so that the first value is 0, this allows to index from i:i+1 for all fill values
        region_bins = np.nancumsum(np.bincount(fill.ravel()+1))
        n_bins = region_bins.size-1
        region_inds = np.argsort(fill.ravel())
        def get_new_label(j):
            wh = region_inds[region_bins[j]:region_bins[j+1]]
            # Occasionally a region won't be able to find a neighbour, in this case we set it to masked
            try:
                output = fill.ravel()[inds_edge.ravel()[wh][np.nanargmin(np.maximum(min_edge.ravel()[wh], field.ravel()[wh]))]]
            except:
                return 0
            # Now need to check if output is masked
            if (type(output) is ma.MaskedArray and output.mask):
                return 0
            else:
                output = output.item()
            # and check if nan
            if np.all(np.isfinite(output)):
                return output
            else:
                return 0
        new_label = ma.array([get_new_label(k) for k in range(n_bins)]).astype(int)
        new_label.fill_value=0
        new_label=new_label.filled()
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
            print("Max label:", np.nanmax(fill))
            if np.unique(fill).size<=10:
                print("Labels:", np.unique(fill))
                print("New:", new_label[np.maximum(0,np.unique(fill).astype(int))])
            else:
                print("Labels:", np.unique(fill)[:10])
                print("New:", new_label[np.maximum(0,np.unique(fill).astype(int)[:10])])
        if np.nanmax(fill)<=max_markers:
            break
    return fill
