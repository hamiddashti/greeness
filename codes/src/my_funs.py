import xarray as xr
import rioxarray
import matplotlib.pylab as plt
import urllib3
from bs4 import BeautifulSoup
import requests
import re
import urllib.request
import geopandas as gpd
import os
import dask
import numpy as np
import chime
from scipy import stats
import numpy as np
import pandas as pd

def download_noaa(host,fname,data_dir):
    """Downloading NOAA-CDR data
    
    Argument:
    host:: https://www.ncei.noaa.gov/data/avhrr-land-leaf-area-index-and-fapar/access/
    fname:: name of files. Use get_filenames function """
    url = host+fname
    path = url.split('/')[-1].split('?')[0]
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(data_dir+fname, 'wb') as f:
            f.write(r.content)
    
def get_filenames(host,name_key):
    """ Get all the link names of files from NOAA repo
    Argument:
    host:: NOAA repo
    name_key:: a keyword common in all file names (e.g. AVHRR) 
    """
    req = requests.get(host)
    soup = BeautifulSoup(req.text,features="html.parser")
    pattern = re.compile(name_key)
    filenames = []
    for link in soup.find_all("a", href=pattern):
        fname = link.get('href')
        filenames.append(fname)
    return filenames

def clip_noaa_sequential(host,filenames,shp_file,data_dir):
    """Sequentially (not parallel) downloading and clip data.
    - Note the clip_noaa_parallel function does the same thing 
    but in parallel which is faster 
    
    Arguments:
    host:: NOAA Repo
    filenames:: name of files (use get_filenames function)
    shp_dir:: shp file directory
    year_dir:: each year is in a different folder
    """
    for fname in filenames:
        download_noaa(host,fname,data_dir)
        ds = xr.open_dataset(data_dir+fname,decode_coords="all")
        ds = ds.rio.write_crs(ds.rio.crs)
        geodf = gpd.read_file(shp_file)
        clipped = ds.rio.clip(geodf.geometry)
        clipped.to_netcdf(data_dir+"clipped_"+fname)
        os.remove(data_dir+fname)

@dask.delayed
def clip_noaa_parallel(fname,host,shp_file,data_dir,product):
    """Downloading data from NOAA and clip them using ABoVE shp
    It is parallelized using DASK
    
    Arguments:
    host:: NOAA Repo
    filenames:: name of files (use get_filenames function)
    shp_dir:: shp file directory
    year_dir:: each year is in a different folder
    """
    
    download_noaa(host,fname,data_dir)
    if (product =="ndvi")|(product=="reflectance"):
        ds = xr.open_dataset(data_dir+fname,decode_coords="all",drop_variables="TIMEOFDAY")
    elif (product=="lai"):
        ds = xr.open_dataset(data_dir+fname,decode_coords="all")

    ds = ds.rio.write_crs(ds.rio.crs)
    # Readin the ABoVE reagion shp file
    geodf = gpd.read_file(shp_file)
    # Clip data using rioxarray
    clipped = ds.rio.clip(geodf.geometry)
    clipped.to_netcdf(data_dir+"clipped_"+fname)
    os.remove(data_dir+fname)

# ------------------------------------
#    Decimal to binary for QA
# ------------------------------------
# Convert decimal to binary and apply it 
# xarray object
def f_dec2bin(x,n):
    return np.binary_repr(x,n)
def dec2bin(xrd,n):
    return xr.apply_ufunc(
        f_dec2bin,
        xrd,
        n,
        dask="parallelized",
        output_dtypes = int,
        vectorize=True,
    )

def beep():
    return chime.warning()
# ------------------------------------


# -----------------------------------
#     Create and apply mask
# -----------------------------------
def f_mask(x,var):
    
    if var=="lai":
        x = '{0:09}'.format(int(x))
        x = str(x)
        if len(x)==9:
            if x=="000000000":
                return np.nan
            else:
                return((x[0:2]=='00')&(x[2]=='1')&(x[7:9]=='00'))
    if var=="ndvi":
        x = '{0:016}'.format(int(x))
        x = str(x)
        if len(x)==16:
            if x=="0000000000000000":
                return np.nan
            else:
                return((x[1]=='0')&(x[2]=='0')&(x[3]=='0')&(x[4]=='0')
                &(x[5]=='0')&(x[6]=='0')&(x[7]=='0')&(x[8]=='0')&(x[9]=='0')
                &(x[10]=='0')&(x[11]=='0')&(x[12]=='0')&(x[13]=='0')&(x[14]=='0')
                &(x[15]=='0'))

def avhrr_mask(xrd,var,dask):
    # Dask allowed only if xrd is dask array
    # (i.e. not .compute() or .load())
    if dask=="allowed":
        return xr.apply_ufunc(
        f_mask,
        xrd,
        var,
        dask="parallelized",
        output_dtypes = float,
        vectorize=True,
    )
    elif dask == "not_allowed":
        return xr.apply_ufunc(
        f_mask,
        xrd,
        var,
        vectorize=True,
    )

def growing_season(da):
	# Taking the mean of the LST data from April to October. Selection of the month is just beacuse
	# initital investigation of the Landsat NDVI data showed the satrt and end of the season.

	da_grouped = da.where(
		da.time.dt.month.isin([4, 5, 6, 7, 8, 9, 10])
	)  # This line set other months than numbered to nan
	da_growing = da_grouped.groupby("time.year").mean()
	return da_growing

def xarray_Linear_trend(xarr, var_unit):
    # getting shapes

    a = xarr.time.to_pandas().index
    b = pd.to_datetime(a, format='%Y')
    xarr["time"] = b

    m = np.prod(xarr.shape[1:]).squeeze()
    n = xarr.shape[0]

    # creating x and y variables for linear regression
    x = xarr.time.to_pandas().index.to_julian_date().values[:, None]
    y = xarr.to_masked_array().reshape(n, -1)

    # ############################ #
    # LINEAR REGRESSION DONE BELOW #
    xm = x.mean(0)  # mean
    ym = y.mean(0)  # mean
    ya = y - ym  # anomaly
    xa = x - xm  # anomaly

    # variance and covariances
    xss = (xa**2).sum(0) / (n - 1)  # variance of x (with df as n-1)
    yss = (ya**2).sum(0) / (n - 1)  # variance of y (with df as n-1)
    xys = (xa * ya).sum(0) / (n - 1)  # covariance (with df as n-1)
    # slope and intercept
    slope = xys / xss
    intercept = ym - (slope * xm)
    # statistics about fit
    df = n - 2
    r = xys / (xss * yss)**0.5
    t = r * (df / ((1 - r) * (1 + r)))**0.5
    p = stats.distributions.t.sf(abs(t), df)

    # misclaneous additional functions
    # yhat = dot(x, slope[None]) + intercept
    # sse = ((yhat - y)**2).sum(0) / (n - 2)  # n-2 is df
    # se = ((1 - r**2) * yss / xss / df)**0.5

    # preparing outputs
    out = xarr[:2].mean('time')
    # first create variable for slope and adjust meta
    xarr_slope = out.copy()
    xarr_slope.name = '_slope'
    xarr_slope.attrs['units'] = var_unit
    xarr_slope.values = slope.reshape(xarr.shape[1:])
    # do the same for the p value
    xarr_p = out.copy()
    xarr_p.name = '_Pvalue'
    xarr_p.attrs[
        'info'] = "If p < 0.05 then the results from 'slope' are significant."
    xarr_p.values = p.reshape(xarr.shape[1:])
    # join these variables
    xarr_out = xarr_slope.to_dataset(name='slope')
    xarr_out['pval'] = xarr_p

    return xarr_out