import xarray as xr

def match_coords(ds_list, dim):
    for ds in ds_list[1:]:
        ds[dim] = ds_list[0][dim]
    return

def get_ds_area_mean(ds, l=1, axis=None):
    if axis is None:
        interp_dims = {dim:ds[dim].data[:ds[dim].size//l*l].reshape(-1,l).mean(-1) for dim in ds.dims}
    else:
        if not hasattr(axis, '__iter__'):
            axis = [axis]
        interp_dims = {ds.dims[i]:ds[ds.dims[i]].data[:ds[ds.dims[i]].size//l*l].reshape(-1,l).mean(-1) for i in axis}
    return ds.interp(interp_dims)
