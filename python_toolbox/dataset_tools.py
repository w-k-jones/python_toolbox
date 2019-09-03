import xarray as xr
import numpy as np

def match_coords(ds_list, dim):
    for ds in ds_list[1:]:
        ds[dim] = ds_list[0][dim]
    return

def interp_ds_area(ds, l=1, axis=None):
    if axis is None:
        interp_dims = {dim:ds[dim].data[:ds[dim].size//l*l].reshape(-1,l).mean(-1) for dim in ds.dims}
    else:
        if not hasattr(axis, '__iter__'):
            axis = [axis]
        interp_dims = { (i if type(i) is str else ds.dims[i]):(
                         ds[ds[i]].data[:ds[ds[i]].size//l*l].reshape(-1,l).mean(-1) if type(i) is str
                         else ds[ds.dims[i]].data[:ds[ds.dims[i]].size//l*l].reshape(-1,l).mean(-1))
                         for i in axis }
    return ds.interp(interp_dims)

def get_ds_area_mean(ds, l=1, axis=None):
    return ds_area_func(np.mean, ds, l, dims=axis, chop=True)

def apply_area_func(func, data, l, axis=None, chop=False, **kwargs):
    if axis == None:
        axis = range(len(data.shape))
    if not hasattr(axis, '__iter__'):
        axis = [axis]
    if hasattr(l, '__iter__'):
        ax_l = dict(zip(axis, l))
    else:
        ax_l = {ax:l for ax in axis}
    data_slice = tuple()
    reshape_1 = []
    reshape_2 = [-1]
    move_axis = []
    counter = 1
    for dim, shape in enumerate(data.shape):
        if dim in axis:
            if shape % ax_l[dim] != 0:
                if chop:
                    data_slice += (slice(0, shape//ax_l[dim]*ax_l[dim]),)
                else:
                    raise shapeError('Length scale '+str(l)+' is not a factor of axis '+str(dim)+
                                     '. Please reshape input array or use keyword "chop=True"')
            else:
                data_slice += (slice(None),)
            reshape_1.extend([shape//ax_l[dim], ax_l[dim]])
            reshape_2.append(shape//ax_l[dim])
            move_axis.append(counter)
            counter += 2
        else:
            data_slice += (slice(None),)
            reshape_1.append(shape)
            reshape_2.append(shape)
            counter += 1
    return func(np.moveaxis(data[data_slice].reshape(reshape_1), move_axis, range(len(move_axis))).reshape(reshape_2), 0, **kwargs)

def ds_area_func(func, da, l, dims=None, chop=False, coords_func=np.mean, **kwargs):
    if dims == None:
        dims = da.dims
    if not hasattr(dims, '__iter__') or type(dims) is str:
        dims = [dims]
    if np.all([type(dim)==str for dim in dims]):
        axis = [da.dims.index(dim) for dim in dims]
    elif np.all([type(dim)==int for dim in dims]):
        axis = dims
        dims = [da.dims[ax] for ax in axis]
    else:
        raise ValueError('Dims must either be all str type or all integer type')
    if hasattr(l, '__iter__'):
        dim_l = dict(zip(dims, l))
    else:
        dim_l = {dim:l for dim in dims}

    new_coords = {key:da.coords[key] for key in da.coords.keys() if key not in da.dims}
    for dim in da.dims:
        new_coords[dim] = apply_area_func(coords_func, da[dim].data, dim_l[dim], chop=chop) if dim in dims else da[dim]


    return xr.DataArray(apply_area_func(func, da.data, l, axis=axis, chop=chop, **kwargs),
                        dims=da.dims, coords=new_coords)

def absmax(data, axis=None):
    if axis == None:
        return np.ravel(data)[np.argmax(np.abs(data))]
    else:
        data_slice = np.meshgrid(*(np.arange(shape) for dim, shape in enumerate(data.shape) if dim != axis), indexing='ij')
        data_slice.insert(axis, np.argmax(np.abs(data), axis))
        return data[tuple(data_slice)]

def absmin(data, axis=None):
    if axis == None:
        return np.ravel(data)[np.argmin(np.abs(data))]
    else:
        data_slice = np.meshgrid(*(np.arange(shape) for dim, shape in enumerate(data.shape) if dim != axis), indexing='ij')
        data_slice.insert(axis, np.argmin(np.abs(data), axis))
        return data[tuple(data_slice)]

def nanabsmax(data, axis=None):
    if axis == None:
        return np.ravel(data)[np.nanargmax(np.abs(data))]
    else:
        data_slice = np.meshgrid(*(np.arange(shape) for dim, shape in enumerate(data.shape) if dim != axis), indexing='ij')
        data_slice.insert(axis, np.nanargmax(np.abs(data), axis))
        return data[tuple(data_slice)]

def nanabsmin(data, axis=None):
    if axis == None:
        return np.ravel(data)[np.nanargmin(np.abs(data))]
    else:
        data_slice = np.meshgrid(*(np.arange(shape) for dim, shape in enumerate(data.shape) if dim != axis), indexing='ij')
        data_slice.insert(axis, np.nanargmin(np.abs(data), axis))
        return data[tuple(data_slice)]
