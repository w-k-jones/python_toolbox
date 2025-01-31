import numpy as np
from numpy import ma
import xarray as xr
from glob import glob
from datetime import datetime, timedelta
from scipy.signal import convolve
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pyproj import Proj, Geod
import cv2 as cv
from dateutil.parser import parse as parse_date
from .dataset_tools import get_ds_area_mean, match_coords, ds_area_func

def get_goes_abi_files(input_file):
    # Returns a list of the datetime and all 16 channel file names for ABI lvl1 data from the path of the name of one file
    datestr = input_file.split('/')[-1].split('_')[3]
    yearstr = datestr[1:5]
    doystr = datestr[5:8]
    hourstr = datestr[8:10]
    minstr = datestr[10:12]
    secstr = datestr[12:14]
    file_date = datetime(year=int(yearstr), month=1, day=1, hour=int(hourstr), minute=int(minstr), second=int(secstr)) + timedelta(days=int(doystr)-1)
    files = glob('/'.join(input_file.split('/')[:-1])+'/*'+'-'.join(input_file.split('/')[-1].split('_')[1].split('-')[:-1])+'*s'+yearstr+doystr+hourstr+minstr+'*.nc')
    return [file_date]+files

def get_abi_date_from_filename(filename):
    base_string = filename.split('/')[-1].split('_s')[-1]
    date = parse_date(base_string[:4]+'0101'+base_string[7:13]) + timedelta(days=int(base_string[4:7])-1)
    return date

def get_abi_basemap(ds):
    lon_0 = ds.goes_imager_projection.longitude_of_projection_origin
    h = ds.goes_imager_projection.perspective_point_height
    X = ds.x*h
    Y = ds.y*h
    basemap = Basemap(projection='geos', lon_0=lon_0, area_thresh=5000, resolution='i',
                      llcrnrx=np.nanmin(X), llcrnry=np.nanmin(Y),
                      urcrnrx=np.nanmax(X), urcrnry=np.nanmax(Y),
                      rsphere=(6378137.00,6356752.3142))
    return basemap

def plot_goes_file(filename):
    date = get_abi_date_from_filename(filename)
    with xr.open_dataset(filename) as ds:
        channel = ds.band_id.data[0]
        wavelength = ds.band_wavelength.data[0]
        if channel<7:
            data = get_abi_ref(ds)
            vmin = 0
            vmax = 1
            cmap = 'viridis'
        else:
            data = get_abi_IR(ds)
            vmin= 180
            vmax=320
            cmap='inferno'
        m = get_abi_basemap(ds)
    m.drawcoastlines()
    m.drawparallels(np.arange(20.,51,10.),labels=[False,True,False,False])
    m.drawmeridians(np.arange(180.,351,10.),labels=[False,False,False,True])
    m.imshow(data[::-1], vmin=vmin, vmax=vmax, cmap=cmap)
    m.colorbar()
    plt.title('GOES-16 Channel '+str(channel)+': '+str(wavelength)+' micron', fontweight='semibold', fontsize=15)
    plt.title('%s' % date.strftime('%d %B %Y'), loc='right')
    return

def get_abi_proj(dataset):
    return Proj(proj='geos', h=dataset.goes_imager_projection.perspective_point_height,
                lon_0=dataset.goes_imager_projection.longitude_of_projection_origin,
                sweep=dataset.goes_imager_projection.sweep_angle_axis)

def get_abi_lat_lon(dataset, dtype=None):
    if dtype == None:
        dtype = dataset.dtype
    p = get_abi_proj(dataset)
    xx, yy = np.meshgrid((dataset.x.data*dataset.goes_imager_projection.perspective_point_height).astype(dtype),
                         (dataset.y.data*dataset.goes_imager_projection.perspective_point_height).astype(dtype))
    lons, lats = p(xx, yy, inverse=True)
    lons[lons>=1E30] = np.nan
    lats[lats>=1E30] = np.nan
    return lats, lons

def get_abi_pixel_lengths(dataset, dtype=None):
    """
    Returns the length scales in x and y of each pixel in the input dataset
    """
    if dtype == None:
        dtype = dataset.dtype
    g = Geod(ellps='WGS84')
    lat, lon = get_abi_lat_lon(dataset)
    dy, dx = np.zeros(lat.shape, dtype=dtype), np.zeros(lat.shape, dtype=dtype)
    dy[:-1] = g.inv(lon[:-1],lat[:-1],lon[1:],lat[1:])[-1]/1e3
    dx[:,:-1] = g.inv(lon[:,:-1],lat[:,:-1],lon[:,1:],lat[:,1:])[-1]/1e3
    dy[1:]+=dy[:-1]
    dy[1:-1]/=2
    dx[:,1:]+=dx[:,:-1]
    dx[:,1:-1]/=2
    return dx, dy

def get_abi_pixel_area(dataset, dtype=None):
    if dtype == None:
        dtype = dataset.dtype
    # lat, lon = get_abi_lat_lon(dataset, dtype=dtype)
    # nadir_res = np.array([dataset.spatial_resolution.split('km')[0]]).astype(dtype)
    # xx, yy = np.meshgrid(dataset.astype(dtype).x.data, dataset.astype(dtype).y.data)
    # lx_factor = np.cos(np.abs(np.radians(dataset.goes_imager_projection.longitude_of_projection_origin-lon))+np.abs(xx))
    # ly_factor = np.cos(np.abs(np.radians(dataset.goes_imager_projection.latitude_of_projection_origin-lat))+np.abs(yy))
    # area = nadir_res**2/(lx_factor*ly_factor)
    dx, dy = get_abi_pixel_lengths(dataset, dtype=dtype)
    area = dx*dy
    return area

def get_abi_ref(dataset, check=False, dtype=None):
    ref = dataset.Rad * dataset.kappa0
    if check:
        DQF = dataset.DQF
        ref[DQF<0] = np.nan
        ref[DQF>1] = np.nan
    if dtype == None:
        return ref
    else:
        return ref.astype(dtype)

def get_abi_IR(dataset, check=False, dtype=None):
    bt = (dataset.planck_fk2 / (np.log((dataset.planck_fk1 / dataset.Rad) + 1)) - dataset.planck_bc1) / dataset.planck_bc2
    if check:
        DQF = dataset.DQF
        bt[DQF<0] = np.nan
        bt[DQF>1] = np.nan
    if dtype == None:
        return bt
    else:
        return bt.astype(dtype)

def get_abi_ds_from_files(filenames, check=False, dtype=None, slices={}, **kwargs):
    if type(filenames) is str:
        with xr.open_dataset(filenames, **kwargs).isel(**slices) as ds:
            channel = ds.band_id.data[0]
            wavelength = ds.band_wavelength.data[0]
            if channel<7:
                DataArray = get_abi_ref(ds, dtype=dtype)
            else:
                DataArray = get_abi_IR(ds, dtype=dtype)
    elif hasattr(filenames, '__iter__'):
        with xr.open_mfdataset(filenames, combine='nested', concat_dim='t', **kwargs).isel(**slices) as ds:
            channel = ds.band_id.data[0]
            wavelength = ds.band_wavelength.data[0]
            if channel<7:
                DataArray = get_abi_ref(ds, dtype=dtype)
            else:
                DataArray = get_abi_IR(ds, dtype=dtype)
    else:
        raise ValueError("""Error in 'get_abi_ds_from_files: filenames input must be either a string
                            or a list of strings'""")
    DataArray.attrs['goes_imager_projection'] = ds.goes_imager_projection
    DataArray.attrs['band_id'] = ds.band_id
    DataArray.attrs['band_wavelength'] = ds.band_wavelength
    return DataArray


def plot_goes_file(filename):
    date = get_abi_date_from_filename(filename)
    with xr.open_dataset(filename) as ds:
        channel = ds.band_id.data[0]
        wavelength = ds.band_wavelength.data[0]
        if channel<7:
            data = get_abi_ref(ds)
            vmin = 0
            vmax = 1
            cmap = 'viridis'
        else:
            data = get_abi_IR(ds)
            vmin= 180
            vmax=320
            cmap='inferno'
        m = get_abi_basemap(ds)
    m.drawcoastlines()
    m.drawparallels(np.arange(20.,51,10.),labels=[False,True,False,False])
    m.drawmeridians(np.arange(180.,351,10.),labels=[False,False,False,True])
    m.imshow(data[::-1], vmin=vmin, vmax=vmax, cmap=cmap)
    m.colorbar()
    plt.title('GOES-16 Channel '+str(channel)+': '+str(wavelength)+' micron', fontweight='semibold', fontsize=15)
    plt.title('%s' % date.strftime('%d %B %Y'), loc='right')
    return

def get_abi_BT_from_files(filenames, check=False, dtype=None):
    """
    This function is being depreciated, using xr.open_mfdataset is much better.
    See function "get_abi_ds_from_files"
    """
    if type(filenames) is str:
        with xr.open_dataset(filenames) as ds:
            return(get_abi_IR(ds, dtype=dtype))
    elif hasattr(filenames, '__iter__'):
        data_list = []
        for f in filenames:
            with xr.open_dataset(f) as ds:
                data_list.append(get_abi_IR(ds, dtype=dtype))
        return xr.concat(data_list, dim='t')
    else:
        raise ValueError("""Error in 'get_abi_BT_from_files: filenames input must be either a string
                            or a list of strings'""")

def _contrast_correction(color, contrast):
    """
    Modify the contrast of an R, G, or B color channel
    See: #www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/

    Input:
        C - contrast level
    """
    F = (259*(contrast + 255))/(255.*259-contrast)
    COLOR = F*(color-.5)+.5
    COLOR = np.minimum(COLOR, 1)
    COLOR = np.maximum(COLOR, 0)
    return COLOR

def get_abi_rgb(C01_ds, C02_ds, C03_ds, IR_ds=None, gamma=0.4, contrast=75, l=1):
    if IR_ds is not None:
        l = l*2
    l = int(l)
    R = ds_area_func(np.mean, C02_ds, l*2, dims=('x','y'), chop=True)
    if l > 1:
        G = ds_area_func(np.mean, C03_ds, l, dims=('x','y'), chop=True)
        B = ds_area_func(np.mean, C01_ds, l, dims=('x','y'), chop=True)
    else:
        G = C03_ds.copy()
        B = C01_ds.copy()
    match_coords([R,G,B], 'x')
    match_coords([R,G,B], 'y')
    match_coords([R,G,B], 't')
    R = np.maximum(R, 0)
    R = np.minimum(R, 1)
    G = np.maximum(G, 0)
    G = np.minimum(G, 1)
    B = np.maximum(B, 0)
    B = np.minimum(B, 1)
    R = np.power(R, gamma)
    G = np.power(G, gamma)
    B = np.power(B, gamma)
    G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
    G_true = np.maximum(G_true, 0)
    G_true = np.minimum(G_true, 1)
    if IR_ds is not None:
        if l//2 > 1:
            IR = get_ds_area_mean(IR_ds, l//2)
        else:
            IR = IR_ds.copy()
        match_coords([R,IR], 'x')
        match_coords([R,IR], 'y')
        match_coords([R,IR], 't')
        IR = np.maximum(IR, 90)
        IR = np.minimum(IR, 313)
        IR = (IR-90)/(313-90)
        IR = (1 - IR.data)/1.5
        RGB = _contrast_correction(np.stack([np.maximum(R, IR), np.maximum(G_true, IR), np.maximum(B, IR)], -1), contrast=contrast)
    else:
        RGB = _contrast_correction(np.stack([R, G_true, B], -1), contrast=contrast)

    return RGB

def get_abi_RGB_from_files(C01_file, C02_file, C03_file, IR_file=None, gamma=0.4, contrast=75, l=1):
    if IR_file != None:
        RGB = get_abi_rgb(get_abi_ds_from_files(C01_file),
                          get_abi_ds_from_files(C02_file),
                          get_abi_ds_from_files(C03_file),
                          IR_ds=get_abi_ds_from_files(IR_file),
                          gamma=gamma, contrast=contrast, l=l)
    else:
        RGB = get_abi_rgb(get_abi_ds_from_files(C01_file),
                          get_abi_ds_from_files(C02_file),
                          get_abi_ds_from_files(C03_file),
                          gamma=gamma, contrast=contrast, l=l)
    return RGB

def _recursive_linker(links_list1=None, links_list2=None, label_list1=None, label_list2=None, overlap_list1=None, overlap_list2=None):
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
        links_list2, links_list1 = _recursive_linker(links_list1=links_list2, links_list2=links_list1, label_list1=label_list2, label_list2=label_list1, overlap_list1=overlap_list2, overlap_list2=overlap_list1)
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
        temp_links1, temp_links2 = _recursive_linker([label_list1[0]], label_list1=label_list1, label_list2=label_list2, overlap_list1=overlap_list1, overlap_list2=overlap_list2)
        links_list1.append(temp_links1)
        links_list2.append(temp_links2)
    return links_list1, links_list2

def get_flow(frame0, frame1, dtype=None):
    if dtype == None:
        flow = cv.calcOpticalFlowFarneback(ds_to_8bit(frame0).data.compute(),ds_to_8bit(frame1).data.compute(), None, 0.5, 3, 4, 3, 5, 1.2, 0)
    else:
        flow = cv.calcOpticalFlowFarneback(ds_to_8bit(frame0).data.compute(),ds_to_8bit(frame1).data.compute(), None, 0.5, 3, 4, 3, 5, 1.2, 0).astype(dtype)
    return flow[...,0], flow[...,1]
