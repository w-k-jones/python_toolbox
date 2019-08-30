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
        interp_dims = {ds.dims[i]:ds[ds.dims[i]].data[:ds[ds.dims[i]].size//l*l].reshape(-1,l).mean(-1) for i in axis}
    return ds.interp(interp_dims)

def get_ds_area_mean(ds, l=1, axis=None):
    return ds_area_func(np.mean, da, l, dims=axis, chop=True)

def ds_area_func(func, da, l, dims=None, chop=False):
    if dims == None:
        dims = da.dims
    if hasattr(dims, '__iter__') and type(dims) is not str:
        dim_inds = [da.dims.index(dim) if type(dim) is str else dim for dim in dims ]
    elif type(dims) is str:
        dim_inds = [da.dims.index(dims),]
    else:
        dim_inds = [dims,]

    if not hasattr(l, '__iter__'):
        l = [l]*len(dims)

    shape = tuple()
    shape_slice = tuple()
    action_inds = tuple()
    action_count = 0
    l_ind = 0
    for i in range(len(da.dims)):
        if i in dim_inds:
            shape += (da.shape[i]//l[l_ind], l[l_ind])
            action_inds += (action_count+1,)
            action_count += 2
            if chop:
                shape_slice += (slice(0, da.shape[i]//l[l_ind]*l[l_ind]),)
            else:
                shape_slice += (slice(0, da.shape[i]),)
            l_ind += 1
        else:
            shape += (da.shape[i],)
            action_count += 1
            shape_slice += (slice(0, test.shape[i]),)

    return func(da[shape_slice].data.reshape(shape), action_inds)
