import xarray as xr
import Rbeast as rb
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle as pk

dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/"
out_dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/data/processed_data/"

percent = xr.open_dataset(dir +
                          "data/processed_data/percent_cover/percent_cover.nc"
                          )["__xarray_dataarray_variable__"]
changed_pixels_mask = xr.open_dataarray(
    dir + "data/processed_data/noaa_nc/lai_fapar/trend/changed_pixels.nc")

# CT is the dataset that includes the confusion table for all the changed pixels (changed_pixels_mask.sum())
ct = xr.open_dataset(
    dir + "data/processed_data/confusion_tables/ct_all_years_corrected.nc")
pix_id = ct["PIX_INDEX"]
ctn = ct["NORMALIZED_CONFUSION"] * 100
dlcc = ct["DLCC"]  #
conf = ct["CONFUSION"]

var_file = "lai_annual_resample_max.nc"
trend_file = "lai_max_trend.nc"
var_name = "lai_max"

# arr is an empty matrix where pixels valuses are associated with IDs in ct dataset
arr = np.arange(0, 448 * 1348).reshape(448, 1348, order="F")
t = pd.date_range(start="1984", end="2014", freq="A-Dec").year  # Time range

var = xr.open_dataarray(dir +
                        "data/processed_data/noaa_nc/lai_fapar/resampled/" +
                        var_file).rename({
                            "latitude": "lat",
                            "longitude": "lon"
                        })

# lai_data = lai.isel(latitude=200,longitude=400).values
var_changed = var.where(changed_pixels_mask)

metadata = rb.args(whichDimIsTime=1, season="none", startTime=1984)
prior = rb.args(trendMinSepDist=5, trendMaxKnotNum=3)
mcmc = rb.args(seed=1)
extra = rb.args(  # a set of options to specify the outputs or computational configurations
    dumpInputData=
    True,  # make a copy of the aggregated input data in the beast ouput
    numThreadsPerCPU=2,  # Paralell  computing: use 2 threads per cpu core
    numParThreads=
    0,  # `0` means using all CPU cores: total num of ParThreads = numThreadsPerCPU * core Num
    printOptions=False,
)
# season = "none"
out = rb.beast123(var_changed.values, metadata, prior, mcmc, extra)

cp = out.trend.cp  #the most possible changepoint locations in the trend
ncp_med = np.round(out.trend.ncp)  #; the mean number of trend changepoints
ncp1 = np.argwhere(ncp_med == 1)
cp1 = np.squeeze(cp[0, :, :])  # The most possible change point year
occ_mat = np.zeros((10, 10, 448 * 1348))
ct_percent_mat = np.zeros((10, 10, 448 * 1348))
ct_percent_mat[:] = np.nan
percent_mat = np.zeros((10, 10, 448 * 1348))
percent_mat[:] = np.nan
ui = np.triu_indices(10, k=1)
li = np.tril_indices(10, k=-1)
no_lcc = 0
counter = 0

# Go over all pixels and drive some relvant Rbeast info for changed pixles
for i in np.arange(0, cp1.shape[0]):
    for j in np.arange(0, cp1.shape[1]):

        # skip if no change point detected
        if np.isnan(cp1[i, j]):
            continue

        cp_time = int(cp1[i, j])  # Year with the highest possible change point
        idx = arr[i, j]
        id = np.where(pix_id.isel(time=0).values == idx)[0]

        # Select the confusion matrix associated to i and j pixel
        ctn_sel = ctn.isel(ID=id).sel(time=str(cp_time)).squeeze().values

        np.fill_diagonal(ctn_sel, 0)

        I_max = np.unravel_index(ctn_sel.argmax(), ctn_sel.shape)
        if I_max == (0, 0):
            no_lcc += 1
            continue
        abs_lcc = abs(ctn_sel[ui] - ctn_sel[li])
        net_lcc = abs_lcc.sum()

        if net_lcc < 2:
            continue
        percent_mat[I_max + (idx, )] = net_lcc
        # I_max_reversed = I_max[::-1]
        max_ct = np.argmax(abs_lcc)

        #max of upper & lower confusion matrix triangles
        ct_percent_mat[I_max + (idx, )] = max(
            ctn_sel[li[0][max_ct], li[1][max_ct]], ctn_sel[ui[0][max_ct],
                                                           ui[1][max_ct]])
        occ_mat[I_max + (idx, )] += 1

        counter = counter + 1

names = [
    "EF",
    "DF",
    "shrub",
    "Herb",
    "sparse",
    "Barren",
    "Fen",
    "Bog",
    "Shallow/Littoral",
    "Water",
]
occurance = occ_mat.sum(axis=2)
df = xr.DataArray(
    data=occurance,
    dims=["LC_t1", "LC_t2"],
    coords={
        "LC_t1": names,
        "LC_t2": names
    },
)
percent_mat_data = percent_mat[~np.isnan(percent_mat)]
percent_mat_data_df = xr.DataArray(percent_mat_data, dims=["ID"])
df.to_netcdf(dir +
             "data/processed_data/noaa_nc/lai_fapar/Rbeast/dfRbeast_LAIMax.nc")
percent_mat_data_df.to_netcdf(
    dir + "data/processed_data/noaa_nc/lai_fapar/Rbeast/percent_mat_LAIMax.nc")
