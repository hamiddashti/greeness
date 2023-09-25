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
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import pymannkendall as mk


def download_noaa(host, fname, data_dir):
    """Downloading NOAA-CDR data

    Argument:
    host:: https://www.ncei.noaa.gov/data/avhrr-land-leaf-area-index-and-fapar/access/
    fname:: name of files. Use get_filenames function"""
    url = host + fname
    path = url.split("/")[-1].split("?")[0]
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(data_dir + fname, "wb") as f:
            f.write(r.content)


def get_filenames(host, name_key):
    """Get all the link names of files from NOAA repo
    Argument:
    host:: NOAA repo
    name_key:: a keyword common in all file names (e.g. AVHRR)
    """
    req = requests.get(host)
    soup = BeautifulSoup(req.text, features="html.parser")
    pattern = re.compile(name_key)
    filenames = []
    for link in soup.find_all("a", href=pattern):
        fname = link.get("href")
        filenames.append(fname)
    return filenames


def outliers_index(data, m=3.5):
    """
    Returns true if a value is outlier

    :param int data: numpy array
    :param int m: # of std to include data 
    """
    import numpy as np
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d / mdev if mdev else 0.
    return ~(s < m)


def clip_noaa_sequential(host, filenames, shp_file, data_dir):
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
        download_noaa(host, fname, data_dir)
        ds = xr.open_dataset(data_dir + fname, decode_coords="all")
        ds = ds.rio.write_crs(ds.rio.crs)
        geodf = gpd.read_file(shp_file)
        clipped = ds.rio.clip(geodf.geometry)
        clipped.to_netcdf(data_dir + "clipped_" + fname)
        os.remove(data_dir + fname)


@dask.delayed
def clip_noaa_parallel(fname, host, shp_file, data_dir, product):
    """Downloading data from NOAA and clip them using ABoVE shp
    It is parallelized using DASK

    Arguments:
    host:: NOAA Repo
    filenames:: name of files (use get_filenames function)
    shp_dir:: shp file directory
    year_dir:: each year is in a different folder
    """

    download_noaa(host, fname, data_dir)
    if (product == "ndvi") | (product == "reflectance"):
        ds = xr.open_dataset(data_dir + fname,
                             decode_coords="all",
                             drop_variables="TIMEOFDAY")
    elif product == "lai":
        ds = xr.open_dataset(data_dir + fname, decode_coords="all")

    ds = ds.rio.write_crs(ds.rio.crs)
    # Readin the ABoVE reagion shp file
    geodf = gpd.read_file(shp_file)
    # Clip data using rioxarray
    clipped = ds.rio.clip(geodf.geometry)
    clipped.to_netcdf(data_dir + "clipped_" + fname)
    os.remove(data_dir + fname)


# ------------------------------------
#    Decimal to binary for QA
# ------------------------------------
# Convert decimal to binary and apply it
# xarray object


def f_dec2bin(x, n):
    return np.binary_repr(x, n)


def dec2bin(xrd, n):
    return xr.apply_ufunc(
        f_dec2bin,
        xrd,
        n,
        dask="parallelized",
        output_dtypes=int,
        vectorize=True,
    )


def beep():
    return chime.warning()


# ------------------------------------


# -----------------------------------
#     Create and apply mask
# -----------------------------------
def f_mask(x, var):

    if var == "LAI":
        x = "{0:09}".format(int(x))
        x = str(x)
        if len(x) == 9:
            if x == "000000000":
                return np.nan
            else:
                return (x[0:2] == "00") & (x[2] == "1") & (x[7:9] == "00")
    if var == "NDVI":
        x = "{0:016}".format(int(x))
        x = str(x)
        if len(x) == 16:
            if x == "0000000000000000":
                return np.nan
            else:
                return ((x[2] == "0")
                        & (x[6] == "0")
                        & (x[7] == "0")
                        & (x[9] == "0")
                        & (x[13] == "0")
                        & (x[14] == "0"))


def avhrr_mask(xrd, var, dask):
    # Dask allowed only if xrd is dask array
    # (i.e. not .compute() or .load())
    if dask == "allowed":
        return xr.apply_ufunc(
            f_mask,
            xrd,
            var,
            dask="parallelized",
            output_dtypes=float,
            vectorize=True,
        )
    elif dask == "not_allowed":
        return xr.apply_ufunc(
            f_mask,
            xrd,
            var,
            vectorize=True,
        )


# ------------ Time series resampling functions ----------------------------------


def growing_season(da):
    # Taking the mean of the LST data from April to October. Selection of the month is just beacuse
    # initital investigation of the Landsat NDVI data showed the satrt and end of the season.

    da_grouped = da.where(da.time.dt.month.isin(
        [5, 6, 7, 8, 9,
         10]))  # This line set other months than numbered to nan
    da_growing = da_grouped.groupby("time.year").mean().rename(
        {"year": "time"})
    # da_growing = da_growing.rename({"year":"time"})
    return da_growing


ndvi_seasonal_resample = dpm = {
    "noleap": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "365_day": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "standard": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "proleptic_gregorian": [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "all_leap": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "366_day": [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    "360_day": [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
}


def leap_year(year, calendar="standard"):
    """Determine if year is a leap year"""
    leap = False
    if (calendar in ["standard", "gregorian", "proleptic_gregorian", "julian"
                     ]) and (year % 4 == 0):
        leap = True
        if ((calendar == "proleptic_gregorian") and (year % 100 == 0)
                and (year % 400 != 0)):
            leap = False
        elif ((calendar in ["standard", "gregorian"]) and (year % 100 == 0)
              and (year % 400 != 0) and (year < 1583)):
            leap = False
    return leap


def get_dpm(time, calendar="standard"):
    """
    return a array of days per month corresponding to the months provided in `months`
    """
    import numpy as np

    month_length = np.zeros(len(time), dtype=int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar):
            month_length[i] += 1
    return month_length


def weighted_season_group(ds):
    # Make a DataArray with the number of days in each month, size = len(time)
    import xarray as xr

    month_length = xr.DataArray(
        get_dpm(ds.time.to_index(), calendar="noleap"),
        coords=[ds.time],
        name="month_length",
    )
    # Calculate the weights by grouping by 'time.season'
    weights = (month_length.groupby("time.season") /
               month_length.groupby("time.season").sum())
    # Calculate the weighted average
    season_grouped = (ds * weights).groupby("time.season").sum(dim="time",
                                                               skipna=False)
    return season_grouped


def weighted_season_resmaple(ds):
    # Make a DataArray with the number of days in each month, size = len(time)
    import xarray as xr

    month_length = xr.DataArray(
        get_dpm(ds.time.to_index(), calendar="noleap"),
        coords=[ds.time],
        name="month_length",
    )
    season_resample = (ds * month_length).resample(time="QS-DEC").sum() / (
        month_length.where(ds.notnull()).resample(time="QS-DEC").sum())
    return season_resample


# ----------------------------------------------------------------

#                       Estimate trend

# ----------------------------------------------------------------
# def _theilsen(y):
#     x = np.arange(len(y)).reshape(-1, 1)
#     if np.isnan(y).all():
#         return np.nan

#     I = np.where(np.isnan(y))
#     if len(I[0]) > 1:
#         return np.nan

#     yy = np.delete(y, I)
#     x = np.arange(len(yy)).reshape(-1, 1)
#     reg = TheilSenRegressor(random_state=0).fit(x, yy)
#     return reg.coef_

# def est_trend(xrd, method, **kwargs):
#     if method == "theilsen":
#         return xr.apply_ufunc(
#             _theilsen,
#             xrd,
#             input_core_dims=[["time"]],
#             # dask="allowed",
#             # output_dtypes=float,
#             vectorize=True,
#         )


def _theilsen_mannkendall(y):
    x = np.arange(len(y)).reshape(-1, 1)
    if np.isnan(y).all():
        return np.nan, np.nan, np.nan

    I = np.where(np.isnan(y))
    if len(I[0]) > 1:
        return np.nan, np.nan, np.nan

    yy = np.delete(y, I)
    x = np.arange(len(yy)).reshape(-1, 1)
    result = mk.original_test(yy)
    # reg = TheilSenRegressor(random_state=0).fit(x, yy)
    # return result.slope, result.p, result.h
    # return result.slope
    return float(result.slope), float(result.p), float(result.h)


def est_trend(xrd, method, **kwargs):
    xrd = xrd.rename("trend")
    if method == "theilsen":
        cof, p, h = xr.apply_ufunc(
            _theilsen_mannkendall,
            xrd,
            input_core_dims=[["time"]],
            output_core_dims=[[], [], []],
            # dask="parallelized",
            # output_dtypes=float,
            vectorize=True,
        )
        ds_out = cof.to_dataset()
        ds_out["p_value"] = p
        ds_out["h"] = h
        ds_out = ds_out.assign_attrs({
            "trend":
            "The estimated trend using Theilsen method",
            "p":
            "Estimated p_value using Mann-Kendall",
            "h":
            "1 if trend is significant (p<0.05), 0 otherwise",
        })

        return ds_out


def xarray_Linear_trend(xarr, var_unit):
    # getting shapes

    a = xarr.time.to_pandas().index
    b = pd.to_datetime(a, format="%Y")
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
    out = xarr[:2].mean("time")
    # first create variable for slope and adjust meta
    xarr_slope = out.copy()
    xarr_slope.name = "_slope"
    xarr_slope.attrs["units"] = var_unit
    xarr_slope.values = slope.reshape(xarr.shape[1:])
    # do the same for the p value
    xarr_p = out.copy()
    xarr_p.name = "_Pvalue"
    xarr_p.attrs[
        "info"] = "If p < 0.05 then the results from 'slope' are significant."
    xarr_p.values = p.reshape(xarr.shape[1:])
    # join these variables
    xarr_out = xarr_slope.to_dataset(name="slope")
    xarr_out["pval"] = xarr_p

    return xarr_out


def isfinite(x):
    # Find non-nan values
    func = lambda x: np.isfinite(x)
    return xr.apply_ufunc(func, x)


# --------------------------------------------------------------
#                  Calculate the natural variability
# --------------------------------------------------------------
def dist_matrix(x_size, y_size):
    import numpy as np

    a1 = np.floor(x_size / 2)
    a2 = np.floor(y_size / 2)
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    cell = (a1, a2)
    dists = np.sqrt((x_arr - cell[0])**2 + (y_arr - cell[1])**2)
    dists[int(a1), int(a2)] = np.nan
    return dists


def estimate_lcc_trend(percent_cover, trend_total, thresh, winsize):
    win_size_half = int(np.floor(winsize / 2))
    dist_m = dist_matrix(winsize, winsize)

    lc_diff = percent_cover.diff("time")
    diff = (abs(lc_diff) > thresh) * 1
    changed_pixels = (diff == 1).any(dim=["time", "band"])

    trend_roll = (trend_total.rolling({
        "lat": winsize,
        "lon": winsize
    },
                                      center=True).construct({
                                          "lat": "lat_dim",
                                          "lon": "lon_dim"
                                      }).values)
    changed_pixels_roll = (changed_pixels.rolling(
        {
            "lat": winsize,
            "lon": winsize
        }, center=True).construct({
            "lat": "lat_dim",
            "lon": "lon_dim"
        }).values)
    percent_cover_roll = (percent_cover.rolling(
        {
            "lat": winsize,
            "lon": winsize
        }, center=True).construct({
            "lat": "lat_dim",
            "lon": "lon_dim"
        }).values)

    trend_nv_eps = xr.full_like(trend_total, fill_value=np.nan, dtype=float)
    trend_nv = xr.full_like(trend_total, fill_value=np.nan, dtype=float)
    trend_eps = xr.full_like(trend_total, fill_value=np.nan, dtype=float)

    for i in range(0, changed_pixels.shape[0]):
        for j in range(0, changed_pixels.shape[1]):

            # Continue if central pixel not changed
            if changed_pixels_roll[i, j][win_size_half, win_size_half] == 0:
                continue

            mask = changed_pixels_roll[i, j]

            lc_stable = np.argwhere(mask == 0)

            trend_tmp = trend_roll[i, j]

            # if the central pixel trend is nan skip it
            if np.isnan(trend_tmp[win_size_half, win_size_half]):
                continue

            # print(trend_tmp[win_size_half, win_size_half])

            percent_cover_tmp = np.isfinite(percent_cover_roll[
                0, :, i, j, :, :])  # shape (bands=10, winsize, winsize)
            center_lc = percent_cover_tmp[:, win_size_half, win_size_half]

            trend_tmp_masked = []
            dist_tmp_masked = []
            for m in range(len(lc_stable)):
                neighbor_lc = percent_cover_tmp[:, lc_stable[m][0],
                                                lc_stable[m][1]]
                if np.equal(center_lc, neighbor_lc).all():
                    trend_tmp_masked.append(trend_tmp[lc_stable[m][0],
                                                      lc_stable[m][1]])
                    dist_tmp_masked.append(dist_m[lc_stable[m][0],
                                                  lc_stable[m][1]])
            if len(trend_tmp_masked) == 0:
                continue

            trend_tmp_masked = np.array(trend_tmp_masked)
            dist_tmp_masked = np.array(dist_tmp_masked)
            tmp_var1 = np.nansum(trend_tmp_masked / dist_tmp_masked)
            trend_nv[i, j] = tmp_var1 / (np.nansum(dist_tmp_masked))
            trend_eps[i, j] = trend_tmp[win_size_half,
                                        win_size_half] - trend_nv[i, j]
    trend_nv_eps = trend_nv + trend_eps
    included_pixels = np.isfinite(trend_nv_eps)

    return trend_nv, trend_eps, trend_nv_eps, included_pixels
