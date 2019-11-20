import cv2 as cv
import numpy as np
from numpy import ma
import xarray as xr
from scipy import interpolate
from python_toolbox import dataset_tools
from scipy import ndimage as ndi


def ds_to_8bit(ds, vmin=None, vmax=None):
    if vmin is None:
        vmin= ds.min()
    if vmax is None:
        vmax= ds.max()
    ds_out = (ds-vmin)*255/(vmax-vmin)
    return ds_out.astype('uint8')

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
    da_flow = xr.zeros_like(da).compute()*np.nan
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

def flow_convolve(flow_data, structure=None, wrap=False, function=None, **kwargs):
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
    multi_struct = structure.ravel()[wh_struct].reshape(re_shape)
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
        output = ma.empty(flow_data.shape[1:])
        for i in range(flow_data.shape[1]):
            temp = ma.array(flow_data[:,i][offset_inds_corrected])*multi_struct
            temp.mask = np.logical_or(np.isnan(temp), offset_mask)
            output[i] = function(temp, 0, **kwargs)
    else:
        output = ma.empty((n_struct,)+flow_data.shape[1:])
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

def flow_sobel(flow_stack, axis=None, direction=None):
    # temp_convolve = flow_convolve(flow_stack, structure=np.ones([3,3,3]))
    nd = len(flow_stack.shape)-1
    output = []
    if axis is None:
        axis = range(nd)
    if not hasattr(axis, '__iter__'):
        axis = [axis]
    if direction is None:
        for i in axis:
            def temp_sobel_func(temp, axis, counter=[0]):
                sobel_matrix = np.transpose(get_sobel_matrix(3),
                               np.roll(np.arange(3),(1+i)%3)).ravel().reshape((-1,1,1))
                output = np.sum((temp-flow_stack[1][counter[0]]) * sobel_matrix, axis)
                counter[0]+=1
                return output

            output.append(flow_convolve(flow_stack, structure=np.ones([3,3,3]),
                                        function=temp_sobel_func))

    elif direction == 'uphill':
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

def flow_network_watershed(field, markers, flow_func, mask=None, structure=None, max_iter=100, max_no_progress=10, expand_mask=True, debug_mode=False):
    # Check structure input, set default and check dimensions and shape
    if structure is None:
        if debug_mode:
            print("Setting structure to default")
        structure = np.ones([3,3,3])
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
        mask = np.zeros_like(field)
    if np.any([s != 3 for s in structure.shape]):
        if debug_mode:
            print("Inserting structure into 3x3x3 array")
        wh = [slice(0,3) if s==3 else slice(1,2) for s in structure.shape]
        temp = np.zeros([3,3,3])
        temp[wh] = structure
        structure=temp

    # Get ravelled indices for each pixel in the field, and find nearest neighbours using flow field
    if debug_mode:
        print("Calculated indices stack")
    inds = np.arange(field.size).reshape(field.shape)
    ind_stack = get_flow_stack(xr.DataArray(inds, dims=('t','y','x')),
                               flow_func, method='nearest').to_masked_array().astype(int)
    # Get nearest flow neighbour values for the field
    if debug_mode:
        print("Calculating field stack")
    field_stack = get_flow_stack(xr.DataArray(field, dims=('t','y','x')),
                                 flow_func, method='nearest').to_masked_array()
    # Find index of the smallest neighbour ro each pixel in the field
    if debug_mode:
        print("Calculating nearest neighbours")
    min_convolve = flow_convolve(field_stack, structure=structure, function=np.nanargmin)
    inds_convolve = flow_convolve(ind_stack, structure=structure)
    inds_convolve.mask = np.logical_or(inds_convolve.mask, inds_convolve<0)
    inds_neighbour = inds_convolve[tuple([min_convolve.data.astype(int)]+np.meshgrid(*(range(s) for s in inds.shape), indexing='ij'))].astype(int)
    # Now iterate over neighbour network to find minimum convergence point for each pixel
        # Each pixel will either reach a minimum or loop back to itself
    test_neighbour = inds_neighbour.copy()
    type_converge = np.zeros(test_neighbour.shape)
    iter_converge = np.zeros(test_neighbour.shape)
    ind_converge = np.zeros(test_neighbour.shape).astype(int)
    fill_markers = markers.astype(int)-mask.astype(int)
    max_markers = np.nanmax(markers)
    if debug_mode:
        print("Finding network convergence locations")
    for i in range(max_iter):
        old_neighbour, test_neighbour = test_neighbour.copy(), test_neighbour.ravel()[test_neighbour.ravel()].reshape(test_neighbour.shape)
        # wh_mark = np.logical_and(type_converge==0, fill_markers.ravel()[test_neighbour.ravel().reshape(fill_markers.shape)]!=0)
        # if np.any(wh_mark):
        #     type_converge[wh_mark] = 3
        #     iter_converge[wh_mark] = i
        #     ind_converge[wh_mark] = test_neighbour[wh_mark]
        wh_ind = np.logical_and(type_converge==0, test_neighbour==inds)
        if np.any(wh_ind):
            type_converge[wh_ind] = 1
            iter_converge[wh_ind] = i
            ind_converge[wh_ind] = test_neighbour[wh_ind]
        wh_conv = np.logical_and(type_converge==0, test_neighbour==old_neighbour)
        if np.any(wh_conv):
            type_converge[wh_conv] = 2
            iter_converge[wh_conv] = i
            ind_converge[wh_ind] = test_neighbour[wh_ind]
        if debug_mode:
            print("Iteration:", i+1)
            print("Pixels converged", np.sum(type_converge!=0))
        if np.all(type_converge!=0):
            if debug_mode:
                print("All pixels converged")
            break

    # Use converged locations to fill watershed basins
    if debug_mode:
        print("Filling basins")
    fill_markers = markers.astype(int)-mask.astype(int)
    wh = np.logical_and(type_converge==1, fill_markers==0)
    fill_markers[wh] = ndi.label(wh)[0][wh]+max_markers
    # fill = fill_markers.ravel()[ind_converge.ravel().astype(int)].reshape(fill_markers.shape)
    fill = fill_markers.ravel()[test_neighbour.ravel().astype(int)].reshape(fill_markers.shape)
    fill[markers]=1
    fill[mask]=-1
    wh = fill==0
    if np.any(wh):
        if debug_mode:
            print("Some pixels not filled, adding")
        fill[wh] = ndi.label(wh)[0][wh]+np.nanmax(fill)

    # Now overflow watershed basins into neighbouring basins until only marker labels are left
    if debug_mode:
        print("Joining labels")
    while np.any(fill>max_markers):
        iter = 1
        # Make a flow stack using the current fill
        temp_fill = get_flow_stack(xr.DataArray(fill, dims=('t','y','x')),
                                                flow_func, method='nearest').to_masked_array()

        # Function to find the minimum neighbour value with a different label
        def min_edge_func(temp, axis, counter=[0]):
            fill_wh = flow_convolve(temp_fill[:,counter[0]].reshape((3,1)+fill.shape[1:]),
                                                   structure=structure) == fill[counter[0]]
            fill_wh_mask = np.logical_or(fill_wh.data, fill_wh.mask)
            temp.mask = np.logical_or(temp.mask, fill_wh_mask.squeeze())
            output = np.nanmin(temp, axis)
            counter[0]+=1
            return output
        min_edge = flow_convolve(field_stack, structure=structure, function=min_edge_func)

        # Function to find the offset of the minimum neighbour with a different label
        def argmin_edge_func(temp, axis, counter=[0]):
            fill_wh = flow_convolve(temp_fill[:,counter[0]].reshape((3,1)+fill.shape[1:]),
                                                   structure=structure) == fill[counter[0]]
            fill_wh_mask = np.logical_or(fill_wh.data, fill_wh.mask)
            temp.mask = np.logical_or(temp.mask, fill_wh_mask.squeeze())
            output = np.nanargmin(temp, axis)
            counter[0]+=1
            return output
        argmin_edge = flow_convolve(field_stack, structure=structure, function=argmin_edge_func)
        inds_edge = inds_convolve[tuple([argmin_edge.data.astype(int)]+np.meshgrid(*(range(s) for s in inds.shape), indexing='ij'))].astype(int)

        object_slices=ndi.find_objects(np.maximum(fill,0))
        for i in np.arange(1, len(object_slices)):
            if object_slices[i] is not None:
                wh = fill[object_slices[i]]==i+1
                argmin = np.nanargmin(np.maximum(min_edge, field)[object_slices[i]][wh])
                new_label = fill.ravel()[inds_edge[object_slices[i]][wh][argmin]]
                if new_label<=i:
                    fill[object_slices[i]][wh] = new_label
        if debug_mode:
            print("Iteration:", iter)
            print("Remaining labels:", np.unique(fill).size)
        iter += 1
    return fill
