import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar
import matplotlib.pylab as plt
import my_funs

dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/"
out_dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/working/"

lai = xr.open_dataarray(dir+"data/processed_data/noaa_nc/lai_fapar/resampled/lai_monthly_resample_mean.nc")

k = 4
lai_total = lai.where(lai.time.dt.month==k,drop=True)
lai_total = lai_total.rename({"latitude":"lat","longitude":"lon"})

