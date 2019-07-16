"""
SEVIRI_tools.py
module containing various tools for SEVIRI data analysis

Changelog:
2018/04/10, WJ: Bug fix to Gradient + Hessians - factor of two missing

Bugs:
Not convergent as of 2018/04/10
"""

import numpy as np
import numpy.ma as ma
import scipy as sp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import netCDF4 as nc
from datetime import datetime,timedelta
from glob import glob
from scipy import interpolate
import itertools
import pyproj

def get_seviri_file_time(file):
    """
    Returns a datetime from the input filename or list of filenames
    """
    if hasattr(file, '__iter__'):
        filenames = [f.split('/')[-1] for f in file]
        date = [datetime(int(f[38:42]), int(f[42:44]),
                                 int(f[44:46]), int(f[46:48]),
                                 int(f[48:50])) for f in filenames]
    else:
        f = file.split('/')[-1]
        date = datetime(int(f[38:42]), int(f[42:44]),
                                 int(f[44:46]), int(f[46:48]),
                                 int(f[48:50]))
    return date

def amv_file_search(year, month, day, hour=None, minute=None, quiet=False):
    date = datetime(year=year, month=month, day=day)
    base_path = '/group_workspaces/jasmin2/acpc/Data/SEVIRI-AMV/'
    globstr = (base_path + 'MSG3-SEVI-MSGAMVE-0100-0100-'
              + str(date.year).zfill(4)
              + str(date.month).zfill(2)
              + str(date.day).zfill(2))
    if hour != None:
        globstr += str(hour).zfill(2)
        if minute != None:
            minute = (int(minute)/60)*60+45
            globstr += str(minute).zfill(2)
    globstr += '*.bfr'
    try:
        files = glob(globstr)
        files.sort()
    except:
        raise Exception('Cannot find AMV directory: ' + base_path)
    if not quiet:
        print 'Files found: ' + str(len(files))
    return files

def imerg_file_search(year, month, day, hour=None, minute=None, quiet=False):
    date = datetime(year=year, month=month, day=day)
    base_path = r'/group_workspaces/cems2/nceo_generic/satellite_data/imerg/'
    path_to_file = (base_path + str(date.year).zfill(4) + '/'
                    + str(date.month).zfill(2) + '/')
    globstr = (path_to_file + '/3B-HHR-E.MS.MRG.3IMERG.'
                   + str(date.year).zfill(4)
                   + str(date.month).zfill(2)
                   + str(date.day).zfill(2))
    if hour != None:
        globstr += '-S'+str(hour).zfill(2)
        if minute != None:
            minute = (int(minute)/30)*30
            globstr += str(minute).zfill(2)
    globstr += '*.nc'
    try:
        files = glob(globstr)
        files.sort()
    except:
        raise Exception('Cannot find IMERG directory: ' + path_to_file)
    if not quiet:
        print 'Files found: ' + str(len(files))
    return files

def orac_merged_file_search(year, month, day, hour=None, minute=None, quiet=False):
    date = datetime(year=year, month=month, day=day)
    path_to_file = r'/group_workspaces/cems2/nceo_generic/satellite_data/seviri/ORAC/clarify/merged/'
    globstr = (path_to_file + '*' + str(date.year).zfill(4)
                    + str(date.month).zfill(2)
                    + str(date.day).zfill(2))
    if hour != None:
        globstr += str(hour).zfill(2)
        if minute != None:
            minute = (int(minute)/15)*15+12
            globstr += str(minute).zfill(2)
    globstr += '*_fv2.0.merged.nc'
    try:
        files = glob(globstr)
        files.sort()
    except:
        raise Exception('Cannot find ORAC-SEVIRI directory: ' + path_to_file)
    if ~quiet:
        print 'Files found: ' + str(len(files))
    return files

# Calculate the mean value over each lxl grid square. Returns an m//l * n//l matrix
def seviri_area_mean(x_in, l=1, nanmean=False):
    x = x_in[:(x_in.shape[0]//l)*l,:(x_in.shape[1]//l)*l]
    x = x.reshape(x_in.shape[0]//l, l, x_in.shape[1]//l, l)
    x = np.moveaxis(x, 1, 2).reshape(x_in.shape[0]//l, x_in.shape[1]//l, l**2)
    if nanmean:
        x = np.nanmean(x, axis=2)
    else:
        x = x.mean(2)
    return x

# Calculate the covariance matrix of each l*l area for a stack of matrices.
# returns an x * x * m//l * n//l size matrix
def seviri_area_cov(x_in, l=1, ignore_nan=False):
    if type(x_in) == type([]):
        n_vars = len(x_in)
    elif type(x_in) == type(np.array([1])):
        n_vars = x_in.shape[0]
    else:
        x_in = [x_in]
        n_vars = 1
    x = [x_i[:(x_i.shape[0]//l)*l,:(x_i.shape[1]//l)*l] for x_i in x_in]
    x = np.stack([np.moveaxis(x_i.reshape(x_i.shape[0]//l, l, x_i.shape[1]//l, l), 1, 2).reshape(x_i.shape[0]//l, x_i.shape[1]//l, l**2) for x_i in x])
    covar = np.full([x_in[0].shape[0]//l, x_in[0].shape[1]//l, n_vars, n_vars], np.nan)
    #covar_inv = np.full([x_i.shape[0]//l, x_i.shape[1]//l, n_vars, n_vars], np.nan)
    if ignore_nan:
        wh_finite = np.isfinite(np.sum(x, axis=0))
        n_finite = np.sum(wh_finite, axis=2)
        for i,j in itertools.product(range(covar.shape[0]), range(covar.shape[1])):
            if n_finite[i,j] > 1:
                covar[i,j] = np.cov(x[:,i,j,wh])
                #covar_inv[i,j] = np.linalg.inv(covar[i,j])
    else:
        for i,j in itertools.product(range(covar.shape[0]), range(covar.shape[1])):
            covar[i,j] = np.cov(x[:,i,j])
            #covar_inv[i,j] = np.linalg.inv(covar[i,j])
    # Now move the covariance axes to the 0,1 positions to ease use of matrices
    #covar = np.transpose(covar, [2,3,0,1])
    #covar_inv = np.transpose(covar_inv, [2,3,0,1])
    return covar#, covar_inv

# Map sensor lat/lon - vor example from SEVIRI - to a rectangular grid of float
#  index values corresponding to locations on another grid. Primarily for use as
#  x, y inputs for interp_tll
def map_ll_to_grid(lon, lat, gridsize=1., limit=[360,180], offset=[0.,90.], direction=[1,-1]):
    x = (direction[0] * lon + offset[0])%limit[0] / gridsize
    y = (direction[1] * lat + offset[1])%limit[1] / gridsize
    return x, y

# 3D interpolation in time, x and y coordinates for collocating time varying
#  data
def interp_tll(data_in, x_in, y_in, t_in, axis=None, mask=False, limit=np.inf,
                limit_type='wrap'):
    """
    This function performs 3d linear interpolation on a grid of data values with
    given x, y and t points.
    """
    data_tmp = data_in.copy()
    # If given axis input roll axes of interest to the front
    if t_in.shape != x_in.shape != y_in.shape:
        raise Exception('Inputs for x_in, y_in, t_in must have the same shape')
    if axis != None:
        try:
            nd = len(axis)
        except:
            raise Exception('Axis keyword input must be array like')
        if nd != 3:
            raise Exception('Axis keyword input must have length 3')
        data_tmp = np.moveaxis(data_tmp, axis, [-1,-2,-3])
    try:
        nd = len(limit)
        if nd != 3:
            raise Exception('limit keyword input must have length 3 or be a scalar')
    except:
        limit = [limit, limit, limit]
    if len(data_tmp.shape) > 3:
        out_shape = data_tmp.shape[:-3] + x_in.shape
        out_mask = np.tile(mask, data_tmp.shape[:-3] + (1,1))
    else:
        out_shape = x_in.shape
        out_mask = mask
    data_out = np.full(out_shape, np.nan)
    t = t_in[~mask]%1
    T = t_in[~mask].astype(int)
    x = x_in[~mask]%1
    X = x_in[~mask].astype(int)
    y = y_in[~mask]%1
    Y = y_in[~mask].astype(int)
    if limit_type == 'wrap':
        data_out[...,~mask] = (
            (1-t)*(1-y)*(1-x)*data_tmp[...,T,Y,X]
            + (1-t)*(1-y)*(x)*data_tmp[...,T,Y,(X+1)%limit[0]]
            + (1-t)*(y)*(1-x)*data_tmp[...,T,(Y+1)%limit[1],X]
            + (1-t)*(y)*(x)*data_tmp[...,T,(Y+1)%limit[1],(X+1)%limit[0]]
            + (t)*(1-y)*(1-x)*data_tmp[...,(T+1)%limit[2],Y,X]
            + (t)*(1-y)*(x)*data_tmp[...,(T+1)%limit[2],Y,(X+1)%limit[0]]
            + (t)*(y)*(1-x)*data_tmp[...,(T+1)%limit[2],(Y+1)%limit[1],X]
            + (t)*(y)*(x)*data_tmp[...,(T+1)%limit[2],(Y+1)%limit[1],(X+1)%limit[0]]
            )
    else:
        raise Exception('limit_type inputs other than wrap not yet accepted')
    return ma.array(data_out, mask=out_mask)

def map_ll_to_seviri(lon, lat):
    """
    This function maps lat/lon points onto pixel location on the SEVIRI imager
    grid. Return is a tuple of masked arrays of the x and y imager grid
    locations.

    (SEVIRI pixel locations) = map_LL_to_SEV(lon, lat)

    This mapping can then be used to find NN or interpolate values much faster
    as bilinear methods can be used directly.
    e.g:
    x, y = map_LL_to_SEV(lon,lat)
    Sindx,Sindy=np.meshgrid(np.arange(3712),np.arange(3712))
    data_grid = interpolate.griddata((x.compressed(), y.compressed()), data,
                                     (Sindx,Sindy), method='linear')

    The function will also screen for input points that are outside the SEVIRI
    instrument field of view by calculating the effective instrument zenith
    angle.
    """
    # new method
    # project lat/lon input to meteosat view, mask out of bounds data
    geos = pyproj.Proj(proj='geos', h=35785831.0,lon_0=0,lat_0=0,x_0=0,y_0=0,units='m')
    x,y = geos(lon,lat)
    x = ma.masked_equal(x,1e30)
    y = ma.masked_equal(y,1e30)
    # Convert to index. ~3000.5m per pixel, centre pixel index is [1855,1855]
    x = x/-3000.5+1855
    y = y/3000.5+1855
    return x,y
    # old method
    """
    # Define Earth radius and geostationary orbit height in km and calucalte max
    #  viewer angle
    r_sat = 42164.
    r_earth = 6378.
    zenith_max = np.arcsin(r_earth/r_sat)
    # convert lat/lon to cartesian coordinates
    x = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    y = np.sin(np.radians(lat))
    z = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    # x,y vector magnitude
    d = np.sqrt(x**2 + y**2)
    # Calculate footprint SEVIRI effective zenith angle and mask for > pi/2
    #  values
    zenith = np.arctan2(d, z) + np.arctan2(r_earth*d, r_sat-r_earth*z)
    zenith_mask = np.abs(zenith) >= (0.5 * np.pi)
    # Calculate x and y viewer angles
    theta_x = np.arctan2(r_earth*x, r_sat-r_earth*z)
    theta_y = np.arctan2(r_earth*y, r_sat-r_earth*z)
    # Define SEVIRI global index range and offset
    # These should be the same on all files, but may need to check
    x_irange = 3623
    x_ioffset = 44
    y_irange = 3611
    y_ioffset = 51
    # Remap viewer angles to indexes using max viewer angle, index range and
    #  offset. Note -ve theta_y as SEVIRI indexes the x-axis right to left(E-W)
    x_out = (1 - theta_x / zenith_max) * 0.5 * x_irange + x_ioffset
    y_out = (1 + theta_y / zenith_max) * 0.5 * y_irange + y_ioffset
    # Return masked arrays using the zenith angle mask
    return ma.array(x_out, mask=zenith_mask), ma.array(y_out, mask=zenith_mask)
    """

def get_seviri_base_time():
    t_base = np.tile(np.arange(0,1237./100,1./100)[...,np.newaxis],1237)
    return t_base

# Iterator class for heat retrieval
class heat_iterator:
    def __init__(self, x_init, x_apriori, x_covar, y_mean, y_var ,h_x=1, h_y=1, step=1):
        try:
            if (x_init.shape[0:1] != x_apriori.shape[0:1] != x_covar.shape[0:1]
                != y_mean.shape[0:1] != y_var.shape[0:1]):
                raise Exception('All inputs must have the same first two dimensions')
        except:
            raise Exception('All inputs must have the same first two dimensions')
        try:
            if h_x.shape[0:1] != x_init.shape[0:1]:
                raise Exception('Array like h_x must have shape shape as first two dimensions of data inputs')
        except:
            try:
                if len(h_x) > 1:
                    raise Exception('h_x must be 2d array_like or scalar')
                else:
                    h_x = h_x[0]
            except:
                pass
        try:
            if h_y.shape[0:1] != x_init.shape[0:1]:
                raise Exception('Array like h_y must have shape shape as first two dimensions of data inputs')
        except:
            try:
                if len(h_y) > 1:
                    raise Exception('h_y must be 2d array_like or scalar')
                else:
                    h_y = h_y[0]
            except:
                pass
        self.x = x_init
        self.xa = x_apriori
        self.x_var = x_covar
        wh = np.linalg.det(x_covar) != 0
        self.x_var_inv = np.full(x_covar.shape, np.nan)
        self.x_var_inv[wh,...] = np.linalg.inv(x_covar[wh,...])
        self.ya = y_mean
        self.y_var = y_var
        self.h_x = h_x
        self.h_y = h_y
        self.step = step
        self.forward_model()
    def NLL(self):
        self.nll = (
            ((self.x - self.xa)[...,np.newaxis,:] *
                (self.x_var_inv * (self.x - self.xa)[...,np.newaxis]).sum(-1)).sum(-1)
            + (self.y - self.ya)**2 * self.y_var
                )
        return
    def cdiff_x(self, x_in, h_x=1):
        x = np.full(x_in.shape, np.nan)
        x[:,1:-1] = 0.5 * (x_in[:,:-2] - x_in[:,2:])
        x[:,0] = x_in[:,0] - x_in[:,1]
        x[:,-1] = x_in[:,-2] - x_in[:,-1]
        if type(x_in) != type(ma.array(0)):
            x[:,1:-1][~np.isfinite(x_in[:,2:])] = (x_in[:,:-2] - x_in[:,1:-1])[~np.isfinite(x_in[:,2:])]
            x[:,1:-1][~np.isfinite(x_in[:,:-2])] = (x_in[:,1:-1] - x_in[:,2:])[~np.isfinite(x_in[:,:-2])]
        else:
            x[:,1:-1][x_in.mask[:,2:]] = (x_in[:,:-2] - x_in[:,1:-1])[x_in.mask[:,2:]]
            x[:,1:-1][x_in.mask[:,:-2]] = (x_in[:,1:-1] - x_in[:,2:])[x_in.mask[:,:-2]]
            x = ma.array(x, mask=np.logical_or(x_in.mask, np.isnan(x)))
        x[~np.isfinite(x_in)] = np.nan
        x /= h_x
        return x
    def cdiff_y(self, x_in, h_y=1):
        x = np.full(x_in.shape, np.nan)
        x[1:-1] = 0.5 * (x_in[:-2] - x_in[2:])
        x[0] = x_in[0] - x_in[1]
        x[-1] = x_in[-2] - x_in[-1]
        if type(x_in) != type(ma.array(0)):
            x[1:-1][~np.isfinite(x_in[2:])] = (x_in[:-2] - x_in[1:-1])[~np.isfinite(x_in[2:])]
            x[1:-1][~np.isfinite(x_in[:-2])] = (x_in[1:-1] - x_in[2:])[~np.isfinite(x_in[:-2])]
        else:
            x[1:-1][x_in.mask[2:]] = (x_in[:-2] - x_in[1:-1])[x_in.mask[2:]]
            x[1:-1][x_in.mask[:-2]] = (x_in[1:-1] - x_in[2:])[x_in.mask[:-2]]
            x = ma.array(x, mask=np.logical_or(x_in.mask, np.isnan(x)))
        x[~np.isfinite(x_in)] = np.nan
        x /= h_y
        return x
    def get_dx(self):
        self.dx = np.stack([self.cdiff_x(self.x[...,1], h_x=self.h_x)
                            + self.cdiff_y(self.x[...,2], h_y=self.h_y),
                            self.cdiff_x(self.x[...,0], h_x=self.h_x),
                            self.cdiff_y(self.x[...,0], h_y=self.h_y)], axis=-1)
        return
    def get_dx_upwind(self):
        dx = np.full(self.x.shape, np.nan)
        wh = np.logical_and(np.isfinite(self.x), self.x > 0)[1:-1,1:-1,...]
        dx[1:-1,1:-1,0][wh[...,0]] = ((self.x[1:-1,1:-1,1][wh[...,0]]
                                        - self.x[1:-1,2:,1][wh[...,0]]) / self.h_x[1:-1,1:-1][wh[...,0]]
                                    + (self.x[1:-1,1:-1,2][wh[...,0]]
                                        - self.x[:-2,1:-1,2][wh[...,0]]) / self.h_y[1:-1,1:-1][wh[...,0]])
        dx[1:-1,1:-1,1][wh[...,1]] = (self.x[1:-1,1:-1,0][wh[...,1]]
                                        - self.x[1:-1,2:,0][wh[...,1]]) / self.h_x[1:-1,1:-1][wh[...,1]]
        dx[1:-1,1:-1,2][wh[...,2]] = (self.x[1:-1,1:-1,0][wh[...,2]]
                                        - self.x[:-2,1:-1,0][wh[...,2]]) / self.h_y[1:-1,1:-1][wh[...,2]]
        wh = np.logical_and(np.isfinite(self.x), self.x <= 0)[1:-1,1:-1,...]
        dx[1:-1,1:-1,0][wh[...,0]] = ( - (self.x[1:-1,1:-1,1][wh[...,0]]
                                        - self.x[1:-1,:-2,1][wh[...,0]]) / self.h_x[1:-1,1:-1][wh[...,0]]
                                    - (self.x[1:-1,1:-1,2][wh[...,0]]
                                        - self.x[2:,1:-1,2][wh[...,0]]) / self.h_y[1:-1,1:-1][wh[...,0]])
        dx[1:-1,1:-1,1][wh[...,1]] = - (self.x[1:-1,1:-1,0][wh[...,1]]
                                        - self.x[1:-1,:-2,0][wh[...,1]]) / self.h_x[1:-1,1:-1][wh[...,1]]
        dx[1:-1,1:-1,2][wh[...,2]] = - (self.x[1:-1,1:-1,0][wh[...,2]]
                                        - self.x[2:,1:-1,0][wh[...,2]]) / self.h_y[1:-1,1:-1][wh[...,2]]
        self.dx = dx
        return
    def forward_model(self):
        self.get_dx()
        self.y = np.sum(self.x * self.dx, axis=-1)
        return
    def get_gradient(self):
        self.grad = 2 * (
            (self.x_var_inv * (self.x - self.xa)[...,np.newaxis]).sum(-1)
            + (self.y - self.ya)[...,np.newaxis] * self.y_var[...,np.newaxis]
            * self.dx
                )
        return
    def get_hessian(self):
        d2x = self.dx[...,np.newaxis] * self.dx[...,np.newaxis,:]
        wh = np.linalg.det(d2x) != 0
        self.hess = np.full(d2x.shape, np.nan)
        self.hess[wh,...] = np.linalg.inv(d2x[wh,...])
        # Inverse method, needs checking
        """
        self.hess = (
            self.x_var
            + (1./(1 + np.trace(
                np.matmul(self.y_var[...,np.newaxis,np.newaxis] * d2x,
                            self.x_var_inv),
                            axis1=-2, axis2=-1)
                                )[...,np.newaxis,np.newaxis])
            * np.matmul(self.x_var, np.matmul(
                self.y_var[...,np.newaxis,np.newaxis] * d2x, self.x_var))
                )
        """
        return
    def next(self):
        self.forward_model()
        self.get_gradient()
        self.get_hessian()
        self.x[1:-1,1:-1,...] = (np.stack([0.25 * (self.x[:-2,1:-1,0]
                                                + self.x[2:,1:-1,0]
                                                + self.x[1:-1,:-2,0]
                                                + self.x[1:-1,2:,0]),
                                            0.5 * (self.x[1:-1,:-2,1]
                                                + self.x[1:-1,2:,1]),
                                            0.5 * (self.x[:-2,1:-1,2]
                                                + self.x[2:,1:-1,2])], axis=-1)
                                + (self.step
                                * (self.hess[1:-1,1:-1,...]
                                * self.grad[1:-1,1:-1,...,np.newaxis]).sum(-1)))
        return

    def plot(self):
        return

# central differences for x and y in seviri data
def cdiff_x(x_in, h_x=1):
    x = np.full(x_in.shape, np.nan)
    x[:,1:-1] = 0.5 * (x_in[:,:-2] - x_in[:,2:])
    x[:,0] = x_in[:,0] - x_in[:,1]
    x[:,-1] = x_in[:,-2] - x_in[:,-1]
    if type(x_in) != type(ma.array(0)):
        x[:,1:-1][~np.isfinite(x_in[:,2:])] = (x_in[:,:-2] - x_in[:,1:-1])[~np.isfinite(x_in[:,2:])]
        x[:,1:-1][~np.isfinite(x_in[:,:-2])] = (x_in[:,1:-1] - x_in[:,2:])[~np.isfinite(x_in[:,:-2])]
    else:
        x[:,1:-1][x_in.mask[:,2:]] = (x_in[:,:-2] - x_in[:,1:-1])[x_in.mask[:,2:]]
        x[:,1:-1][x_in.mask[:,:-2]] = (x_in[:,1:-1] - x_in[:,2:])[x_in.mask[:,:-2]]
        x = ma.array(x, mask=np.logical_or(x_in.mask, np.isnan(x)))
    x[~np.isfinite(x_in)] = np.nan
    x /= h_x
    return x

def cdiff_y(x_in, h_y=1):
    x = np.full(x_in.shape, np.nan)
    x[1:-1] = 0.5 * (x_in[:-2] - x_in[2:])
    x[0] = x_in[0] - x_in[1]
    x[-1] = x_in[-2] - x_in[-1]
    if type(x_in) != type(ma.array(0)):
        x[1:-1][~np.isfinite(x_in[2:])] = (x_in[:-2] - x_in[1:-1])[~np.isfinite(x_in[2:])]
        x[1:-1][~np.isfinite(x_in[:-2])] = (x_in[1:-1] - x_in[2:])[~np.isfinite(x_in[:-2])]
    else:
        x[1:-1][x_in.mask[2:]] = (x_in[:-2] - x_in[1:-1])[x_in.mask[2:]]
        x[1:-1][x_in.mask[:-2]] = (x_in[1:-1] - x_in[2:])[x_in.mask[:-2]]
        x = ma.array(x, mask=np.logical_or(x_in.mask, np.isnan(x)))
    x[~np.isfinite(x_in)] = np.nan
    x /= h_y
    return x

"""
def regrid_to_seviri(data_in, lon, lat, l=1):
    if len(lon.shape) == 1 or len(lat.shape) == 1:
        lon, lat = np.meshgrid(lon, lat)
    x, y = map_ll_to_seviri(lon, lat)
    nd = len(data_in.shape)
    if nd > 2:
        data = data_in.reshape(-1,data_in.shape[-2],data_in.shape[-1])
        data_out = [interpolate.griddata((x.compressed()//l, y.compressed()//l),
                                    data[i][x.mask],
                                    tuple(np.meshgrid(np.arange(3712//l), np.arange(3712//l))))
                    for i in range(data.shape[0])]
        del data
        data_out = np.stack(data_out)
        new_dim = list(data_in.shape[0:-2])
        new_dim.extend([3712//l, 3712//l])
        data_out.shape = tuple(new_dim)
    else:
        data_out = interpolate.griddata((x.compressed()/l, y.compressed()/l),
                                    data_in[x.mask + y.mask == 0],
                                    tuple(np.meshgrid(np.arange(3712//l), np.arange(3712//l))))
    return data_out
"""
def nc_get_ll(nc_file):
    try:
        lon = nc_file.variables['longitude'][:]
        lat = nc_file.variables['latitude'][:]
    except:
        try:
            lon = nc_file.variables['lon'][:]
            lat= nc_file.variables['lat'][:]
        except:
            try:
                lon = nc_file.variables['Longitude'][:]
                lat = nc_file.variables['Latitude'][:]
            except:
                raise Exception('Could not find latitude and longitude variables')
    if len(lon.shape) == len(lat.shape) == 1:
        lon, lat = np.meshgrid(lon, lat)
    return lon, lat

def nc_load_regrid(file_names, variables, l=1, lon=None, lat=None, sum=False):
    seviri_inds = tuple(np.meshgrid(np.arange(3712//l), np.arange(3712//l)))
    # test for multiple files
    if type(file_names) == type('str'):
        n_files = 1
        file_names = [file_names]
    else:
        n_files = len(file_names)
    # test for multiple variables
    if type(variables) == type('str'):
        n_vars = 1
        variables = [variables]
    else:
        n_vars = len(variables)
    out_list = []
    for i,file in enumerate(file_names):
        print file
        nc_file = nc.Dataset(file)
        if (i == 0 and (lon == None or lat == None)):
            lon, lat = nc_get_ll(nc_file)
            """
            x, y = map_ll_to_seviri(lon, lat)
            map_inds = (x.compressed()/l, y.compressed()/l)
            map_mask = x.mask + y.mask == 0
            """
        """
        for var in variables:
            try:
                var_list.append(regrid_to_seviri(nc_file.variables[var][:], lon, lat))
            except:
                raise Exception('Could not load variable ' + var + ' in file ' + file)
        """
        var_list = [regrid_to_seviri(nc_file.variables[var][:], lon, lat)
                    for var in variables]
        if sum:
            out_list.append(np.sum(np.stack(var_list), axis=0))
        else:
            out_list.append(var_list)
        nc_file.close()

    return out_list
"""
def seviri_imshow(figure, data, subplot=111, **kwargs):
    geo_p = ccrs.Geostationary(central_longitude=0.0, satellite_height=35785831, false_easting=0, false_northing=0, globe=None)
    limits = ( geo_p.x_limits[0], geo_p.x_limits[1], geo_p.y_limits[0], geo_p.y_limits[1] )
    ax = figure.add_subplot(subplot, projection=geo_p)
    ax.coastlines()
    ax.set_global()
    ax.imshow(data[::-1,::-1], transform=geo_p, extent=limits, origin='upper', **kwargs)
    return ax
"""
"""SEV_plot_global( SEVIRI_data )

Cartopy cannot plot full disk SEVIRI data using pcolormesh due to the missing lat/lon values.
This routine instead plots the SEVIRI data as an image, dealing with the correct sizing to the geostationary projection internally.
Note: SEVIRI data must be in the level 1 image pixel resolution

inputs: SEVIRI_data array (2D array in native pixel resolution (3712,3712))

results: plots SEVIRI data image

2018/02/22, WJ: Function creation and testing
2018/02/23, WJ: Bug fix; SEVIRI_data indexing was the wring dimensions
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def SEV_plot_global(SEVIRI_data):
    #define geostaionary projection
    geo_p = ccrs.Geostationary(central_longitude=0.0, satellite_height=35785831, false_easting=0, false_northing=0, globe=None)

    #get projection limits
    limits = ( geo_p.x_limits[0], geo_p.x_limits[1], geo_p.y_limits[0], geo_p.y_limits[1] )

    #plot SEVIRI image in geostationary projection, cropping to image size (indexing), flipping the x axis and setting the limits to the extend of the geostationary projection
    plt.imshow(SEVIRI_data[3660:50-1,3666:43:-1], transform=geo_p, extent=limits, origin='upper')
    return
