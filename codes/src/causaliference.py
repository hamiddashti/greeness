import xarray as xr
import numpy as np
from my_funs import est_trend
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from causalimpact import CausalImpact

lai_max_file = "lai_max_norm_included_overlap.nc"
ndvi_max_file = "ndvi_max_norm_included.nc"
# trend_file = "lai_max_trend.nc"
# var_name = "lai_max"

# arr is an empty matrix where pixels valuses are associated with IDs in ct dataset 
arr = np.arange(0, 448 * 1348).reshape(448, 1348, order="F")
t = pd.date_range(start="1985", end="2014", freq="A-Dec").year  # Time range

lai_max = xr.open_dataarray(
    dir + "data/processed_data/noaa_nc/lai_fapar/trend/" + lai_max_file
)
ndvi_max = xr.open_dataarray(
    dir + "data/processed_data/landsat/trend/" + ndvi_max_file
)

lai_max = xr.open_dataarray(
    dir + "data/processed_data/noaa_nc/lai_fapar/resampled/lai_annual_resample_max.nc"
).rename({"latitude":"lat","longitude":"lon"})
ndvi_max = xr.open_dataarray(dir+"data/processed_data/landsat/resampled/ndvi_annual_max.nc")
ndvi_max = ndvi_max.sel(time=slice("1985","2013")).rename({"latitude":"lat","longitude":"lon"})
ndvi_max = ndvi_max/1e4
nir = xr.open_dataarray(dir+"data/processed_data/landsat/resampled/nir.nc")
nir = nir.sel(time=slice("1985","2013"))
swi = xr.open_dataarray(dir+"data/processed_data/swi/swi.nc")
# lai_max["time"] = pd.date_range("1984","2014",freq = "A")
lai_max = lai_max.sel(time=slice("1985","2014"))  # 1984 has many nan values
arr = xr.open_dataarray(dir+"data/arr_id.nc")
percent_cover = (
    xr.open_dataarray(dir + "data/processed_data/percent_cover/percent_cover.nc") * 100
)
percent_cover = percent_cover.loc["1984":"2013"]
percent_cover = percent_cover.round(4)
# If a class is 0 change it no nan to prevent false zeros in diff later
percent_cover = percent_cover.where(percent_cover != 0)
percent_cover["lat"] = lai_max["lat"]
percent_cover["lon"] =  lai_max["lon"]
percent_cover = percent_cover/100

event_year = 2002
lc_diff = percent_cover.diff("time")
event = lc_diff.sel(time=event_year)
lcc = (event.where(event>0)).sum("band")
thresh = 0.5
lcc_dom = lcc.where(lcc>thresh)
changed_pixels = np.isfinite(lcc_dom)
I = np.where(changed_pixels==True)

n = 5
idx = [I[0][n],I[1][n]]
lc_diff[:,:,idx[0],idx[1]].to_pandas().plot(kind="bar",stacked=True,figsize=(18,10))

winsize=3
win_size_half = int(np.floor(winsize / 2))
changed_pixels_roll = (changed_pixels.rolling(
    {
        "lat": winsize,
        "lon": winsize
    }, center=True).construct({
        "lat": "lat_dim",
        "lon": "lon_dim"
    }).values)
lai_max_roll = (lai_max.rolling(
    {
        "lat": winsize,
        "lon": winsize
    }, center=True).construct({
        "lat": "lat_dim",
        "lon": "lon_dim"
    }).values)

ndvi_max_roll = (ndvi_max.rolling(
    {
        "lat": winsize,
        "lon": winsize
    }, center=True).construct({
        "lat": "lat_dim",
        "lon": "lon_dim"
    }).values)
swi_roll = (swi.rolling(
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

