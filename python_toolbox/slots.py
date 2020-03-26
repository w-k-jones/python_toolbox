import numpy as np
from numpy import ma
import xarray as xr
from scipy import ndimage as ndi
from scipy import interpolate
import cv2 as cv
import dask.array as da
from python_toolbox import opt_flow

def _to_8bit(array, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    array_out = (array-vmin) * 255 / (vmax-vmin)
    return array_out.astype('uint8')

def _cv_flow(a, b, pyr_scale=0.5, levels=5, winsize=16, iterations=4,
             poly_n=5, poly_sigma=1.1, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN):
    flow = cv.calcOpticalFlowFarneback(_to_8bit(a), _to_8bit(b), None,
                                       pyr_scale, levels, winsize, iterations,
                                       poly_n, poly_sigma, flags)
    return flow[...,0], flow[...,1]

def dask_flow(a, b, pyr_scale=0.5, levels=5, winsize=16, iterations=4,
              poly_n=5, poly_sigma=1.1, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN,
              dtype=float):
    _dask_flow = da.gufunc(_cv_flow, signature='(a,b),(a,b),(),(),(),(),(),(),()->(a,b),(a,b)',
                       output_dtypes=(dtype, dtype), vectorize=True)
    return _dask_flow(da.asarray(a).rechunk((1,-1,-1)), da.asarray(b).rechunk((1,-1,-1)),
                      pyr_scale, levels, winsize, iterations, poly_n,
                      poly_sigma, flags)

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
        if t==1:
            return self.flow_x_for, self.flow_y_for
        elif t==-1:
            return self.flow_x_back, self.flow_y_back
        else:
            return (0.5*t*(t+1)*self.flow_x_for + 0.5*t*(t-1)*self.flow_x_back,
                    0.5*t*(t+1)*self.flow_y_for + 0.5*t*(t-1)*self.flow_y_back)

def get_flow_func(field, post_iter=0, dtype=float, compute=False, **flow_kwargs):
    input_shape = field.shape

    flow_x_for, flow_y_for = dask_flow(field[:-1], field[1:], dtype=dtype, **flow_kwargs)
    flow_x_back, flow_y_back = dask_flow(field[1:], field[:-1], dtype=dtype, **flow_kwargs)

    flow = Flow_Func(
        da.concatenate([flow_x_for, -flow_x_back[-1:]], 0),
        da.concatenate([-flow_x_for[:1], flow_x_back], 0),
        da.concatenate([flow_y_for, -flow_y_back[-1:]], 0),
        da.concatenate([-flow_y_for[:1], flow_y_back], 0))

    if post_iter>0:
        for i in range(post_iter):
            flow = _smooth_flow(flow, dtype=dtype)

    if compute:
        flow = Flow_Func(flow(1)[0].compute(),
                         flow(-1)[0].compute(),
                         flow(1)[1].compute(),
                         flow(-1)[1].compute())
    return flow

def _smooth_flow(flow, dtype=float):
    x_for_interp = -dask_interp_flow(flow.flow_x_back[1:], flow[:-1], t=1, dtype=dtype)
    y_for_interp = -dask_interp_flow(flow.flow_y_back[1:], flow[:-1], t=1, dtype=dtype)
    x_back_interp = -dask_interp_flow(flow.flow_x_for[:-1], flow[1:], t=-1, dtype=dtype)
    y_back_interp = -dask_interp_flow(flow.flow_y_for[:-1], flow[1:], t=-1, dtype=dtype)

    x_for_smoothed = 0.5 * (x_for_interp + flow.flow_x_for[:-1])
    y_for_smoothed = 0.5 * (y_for_interp + flow.flow_y_for[:-1])
    x_back_smoothed = 0.5 * (x_back_interp + flow.flow_x_back[1:])
    y_back_smoothed = 0.5 * (y_back_interp + flow.flow_y_back[1:])

    return Flow_Func(
        da.concatenate([x_for_smoothed, -x_back_smoothed[-1:]], 0),
        da.concatenate([-x_for_smoothed[:1], x_back_smoothed], 0),
        da.concatenate([y_for_smoothed, -y_back_smoothed[-1:]], 0),
        da.concatenate([-y_for_smoothed[:1], y_back_smoothed], 0))

def _interp_flow(data, flow_x, flow_y, method='linear'):
    x = np.arange(data.shape[-1])
    y = np.arange(data.shape[-2])
    xx, yy = np.meshgrid(x, y)
    new_xx, new_yy = xx + flow_x, yy + flow_y
    new_xx = np.minimum(np.maximum(new_xx, 0), x.max())
    new_yy = np.minimum(np.maximum(new_yy, 0), y.max())
    interp_da = xr.apply_ufunc(interpolate.interpn, (y, x), data, (new_yy, new_xx),
                               kwargs={'method':method, 'bounds_error':False,
                                       'fill_value':None})
    return interp_da

def dask_interp_flow(data, flow, method='linear', t=1, dtype=float):
    _dask_interp_flow = da.gufunc(_interp_flow, signature='(a,b),(a,b),(a,b),()->(a,b)',
                              output_dtypes=dtype, vectorize=True)
    flow_t = flow(t)
    return _dask_interp_flow(da.asarray(data).rechunk((1,-1,-1)),
                             da.asarray(flow_t[0]).rechunk((1,-1,-1)),
                             da.asarray(flow_t[1]).rechunk((1,-1,-1)), method)

def get_flow_stack(field, flow, method='linear', dtype=float):
    input_shape = field.shape
    interp_back = da.concatenate(
        [da.full((1,)+input_shape[1:], np.nan),
         dask_interp_flow(field[:-1], flow[1:], method=method, t=-1, dtype=dtype)], 0)
    interp_for = da.concatenate(
        [dask_interp_flow(field[1:], flow[:-1], method=method, t=1, dtype=dtype),
         da.full((1,)+input_shape[1:], np.nan)], 0)

    return da.stack([interp_back,
                     field,
                     interp_for], 0).rechunk([-1,1,-1,-1])

def get_flow_diff(field, flow, method='linear', dtype=float):
    input_shape = field.shape
    interp_back = dask_interp_flow(field[:-1], flow[1:], method=method, t=-1, dtype=dtype)
    interp_for = dask_interp_flow(field[1:], flow[:-1], method=method, t=1, dtype=dtype)

    diff = da.concatenate([
                interp_for[0:1] - field[0:1],
                (interp_for[1:] - interp_back[:-1])/2,
                field[-1:] - interp_back[-1:]
                ], 0)

    return diff

def convolve_stack(data_stack, structure=ndi.generate_binary_structure(3,1), bc=np.nan):
    data_overlap = da.overlap.overlap(data_stack,
        depth={0:0, 1:0, 2:1, 3:1}, boundary={2:bc, 3:bc})

    shape = data_overlap.shape
    data_convolve = da.concatenate(
        [data_overlap[
            z[0]:shape[0]-2+z[0], :,
            z[1]:shape[2]-2+z[1], z[2]:shape[3]-2+z[2]]
         for z in zip(*np.where(structure))],
            0).rechunk([-1,1,-1,-1])

    return data_convolve

def flow_convolve(
    data, flow, structure=ndi.generate_binary_structure(3,1),
    method='linear', bc=np.nan, dtype=float):

    data_stack = get_flow_stack(data, flow, method=method, dtype=dtype)

    data_convolve = convolve_stack(data_stack, structure=structure, bc=bc)

    return data_convolve

def _sobel_matrix(ndims):
    sobel_matrix = np.array([-1,0,1])
    for i in range(ndims-1):
        sobel_matrix = np.multiply.outer(np.array([1,2,1]), sobel_matrix)
    return sobel_matrix

def flow_sobel(data, flow, method='linear', direction=None, magnitude=False, dtype=float):
    sobel_matrix = _sobel_matrix(3)
    structure = np.ones((3,3,3))
    data_convolve = flow_convolve(data, flow, structure=structure,
                                  method=method, bc=np.nan, dtype=dtype) - data
    if direction == 'uphill':
        data_convolve = da.fmax(data_convolve, 0)
    elif direction =='downhill':
        data_convolve = da.fmin(data_convolve, 0)

    output = (da.nansum(data_convolve
                  * sobel_matrix.transpose([2,0,1]).ravel()[:,np.newaxis,np.newaxis,np.newaxis], 0),
              da.nansum(data_convolve
                  * sobel_matrix.transpose([1,2,0]).ravel()[:,np.newaxis,np.newaxis,np.newaxis], 0),
              da.nansum(data_convolve
                  * sobel_matrix.ravel()[:,np.newaxis,np.newaxis,np.newaxis], 0))

    if magnitude:
        output = (output[0]**2 + output[1]**2 + output[2]**2)**0.5

    return output

def watershed(field, flow, markers,
                      mask=None,
                      structure=ndi.generate_binary_structure(3,1),
                      verbose=False):


    if field.size<np.iinfo(np.int16).max:
        inds_dtype = np.uint16
        fill_dtype = np.int16
    elif field.size<np.iinfo(np.int32).max:
        inds_dtype = np.uint32
        fill_dtype = np.int32
    else:
        inds_dtype = np.uint64
        fill_dtype = np.int64

    inds_neighbour = opt_flow.flow_argmin_nearest(
                            np.arange(field.size, dtype=inds_dtype).reshape(field.shape),
                            da.nanargmin(
                                flow_convolve(field, flow,
                                    structure=ndi.generate_binary_structure(3,1),
                                    method='nearest', dtype=field.dtype),
                                0),
                            flow,
                            ndi.generate_binary_structure(3,1),
                            dtype=inds_dtype)

    wh_marker_mask = np.logical_or(markers!=0, mask!=0)
    inds_neighbour[wh_marker_mask] = np.arange(
        inds_neighbour.size, dtype=inds_dtype)[wh_marker_mask.ravel()]

    fill = ndi.label(np.logical_and(
            (inds_neighbour.ravel()[inds_neighbour.ravel()]
             == np.arange(inds_neighbour.size)).reshape(inds_neighbour.shape),
            np.logical_not(wh_marker_mask)))[0].astype(fill_dtype)

    del wh_marker_mask

    max_markers = markers.max().astype(fill_dtype)
    fill[fill>0] = fill[fill>0] + max_markers
    fill = (fill + markers.astype(fill_dtype) - mask.astype(fill_dtype))
    for i in range(10):
        wh_to_converge = inds_neighbour.ravel()[inds_neighbour.ravel()].reshape(inds_neighbour.shape) != inds_neighbour
        if not np.any(wh_to_converge):
            break
        else:
            inds_neighbour[wh_to_converge] = inds_neighbour.ravel()[inds_neighbour[wh_to_converge]]

    del wh_to_converge

    wh = fill==0
    fill[wh] = fill.ravel()[inds_neighbour[wh]]

    del inds_neighbour

    wh = fill==0
    if np.any(wh):
        fill[wh] = ndi.label(wh)[0][wh]
    fill[fill<=0] = 0

    if fill.max() < np.iinfo(np.int16).max:
        fill_dtype = np.int16
    elif fill.max() < np.iinfo(np.int32).max:
        fill_dtype = np.int32
    else:
        fill_dtype = np.int64

    fill = fill.astype(fill_dtype)

    del wh

    new_struct = ndi.generate_binary_structure(3,1)
    new_struct[1,1,1] = 0

    field_convolve = flow_convolve(field, flow, structure=new_struct,
                                   method='nearest', dtype=field.dtype)
    field_range = np.nanmax(field) - np.nanmin(field)

    for i in range(100):
        if fill.max() <= max_markers:
            break

        fill_convolve = flow_convolve(fill, flow, structure=new_struct,
                                              method='nearest', dtype=fill.dtype)

        argmin_neighbour = da.nanargmin(field_convolve + 2*field_range*(fill_convolve==fill), 0).astype(np.uint8).compute()

        field_neighbour = np.fmax(opt_flow.flow_argmin_nearest(
                        field, argmin_neighbour,
                        flow, structure=new_struct,
                        dtype=field.dtype
                        ),
                        field).astype(field.dtype)

        fill_neighbour = opt_flow.flow_argmin_nearest(
                    fill, argmin_neighbour,
                    flow, structure=new_struct,
                    dtype=fill_dtype)

        field_neighbour[fill_neighbour==fill] = np.inf

        region_bins = np.nancumsum(np.bincount(fill.ravel()))
        region_inds = np.argsort(fill.ravel())

        region_map = {}

        for label in range(max_markers+1, region_bins.size):
            if region_bins[label]>region_bins[label-1]:
                wh = region_inds[region_bins[label-1]:region_bins[label]]
                if np.any(np.isfinite(field_neighbour.ravel()[wh])):
                    region_map[label] = fill_neighbour.ravel()[wh][np.nanargmin(field_neighbour.ravel()[wh])]
                    if region_map[label] == label:
                        region_map[label] = 0
                else:
                    region_map[label] = 0

        for k in region_map:
            for i in range(100):
                if region_map[k] <= max_markers:
                    break
                if region_map[region_map[k]] == k:
                    if k > region_map[k]:
                        break
                    else:
                        region_map[k] = k
                        break
                else:
                    region_map[k] = region_map[region_map[k]]

        for label in region_map:
            if region_map[label] != label:
                if region_bins[label]>region_bins[label-1]:
                    fill.ravel()[region_inds[region_bins[label-1]:region_bins[label]]] = region_map[label]

    return fill
