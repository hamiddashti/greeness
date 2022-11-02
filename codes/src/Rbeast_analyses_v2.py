import xarray as xr
import Rbeast as rb
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/"
out_dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/data/processed_data/"
arr = np.arange(0, 448 * 1348).reshape(448, 1348, order="F")
t = pd.date_range(start="1984", end="2014", freq="A-Dec").year

lai = xr.open_dataarray(
    dir + "data/processed_data/noaa_nc/lai_fapar/resampled/lai_growing_mean.nc"
)
lai = lai.rename({"latitude": "lat", "longitude": "lon"})

percent = xr.open_dataset(dir + "data/processed_data/percent_cover/percent_cover.nc")[
    "__xarray_dataarray_variable__"
]
changed_pixels_mask = xr.open_dataarray(
    dir + "data/processed_data/noaa_nc/lai_fapar/trend/changed_pixels.nc"
)
# lai_data = lai.isel(latitude=200,longitude=400).values
lai_changed = lai.where(changed_pixels_mask)

ct = xr.open_dataset(dir + "data/processed_data/confusion_tables/ct_all_years.nc")
pix_id = ct["PIX_INDEX"]
lc1 = ct["LC_2003"]
lc2 = ct["LC_2013"]
ctn = ct["NORMALIZED_CONFUSION"] * 100
dlcc = ct["DLCC"] * -1  #
conf = ct["CONFUSION"]

metadata = rb.args(whichDimIsTime=1, season="none", startTime=1984)
mcmc = rb.args(seed=1)
extra = rb.args(  # a set of options to specify the outputs or computational configurations
    dumpInputData=True,  # make a copy of the aggregated input data in the beast ouput
    numThreadsPerCPU=2,  # Paralell  computing: use 2 threads per cpu core
    numParThreads=0,  # `0` means using all CPU cores: total num of ParThreads = numThreadsPerCPU * core Num
    printOptions=False,
)
season = "none"
out2 = rb.beast123(lai_changed.values, metadata, [], mcmc, extra)

cp = out2.trend.cp
ncp_med = np.round(out2.trend.ncp)

# for i in np.arange(0,ncp_med.shape[0]*ncp_med.shape[1]):
ncp1 = np.argwhere(ncp_med == 1)
cp1 = np.squeeze(cp[0, :, :])

occ_mat = np.zeros((10, 10, 448 * 1348))
ct_percent_mat = np.zeros((10, 10, 448 * 1348))
ct_percent_mat[:] = np.nan
percent_mat = np.zeros((10, 10, 448 * 1348))
percent_mat[:] = np.nan
ui = np.triu_indices(10, k=1)
li = np.tril_indices(10, k=-1)
no_lcc = 0
counter = 0
for i in np.arange(0, cp1.shape[0]):
    for j in np.arange(0, cp1.shape[1]):
        if np.isnan(cp1[i, j]):
            continue
        cp_time = int(cp1[i, j])
        idx = arr[i, j]
        # if idx == 220137:
        #     print([i,j])
        #     break
        # else:
        #     continue
        id = np.where(pix_id.isel(time=0).values == idx)[0]
        ct_sel = ctn.isel(ID=id).sel(time=cp_time).squeeze().values
        np.fill_diagonal(ct_sel, 0)
        I_max = np.unravel_index(ct_sel.argmax(), ct_sel.shape)
        if I_max == (0, 0):
            no_lcc += 1
            continue
        abs_lcc = abs(ct_sel[ui] - ct_sel[li])
        net_lcc = abs_lcc.sum()
        if net_lcc == 0:
            continue
        percent_mat[I_max + (idx,)] = net_lcc
        I_max_reversed = I_max[::-1]
        max_ct = np.argmax(abs_lcc)
        ct_percent_mat[I_max + (idx,)] = max(
            ct_sel[li[0][max_ct], li[1][max_ct]], ct_sel[ui[0][max_ct], ui[1][max_ct]]
        )
        occ_mat[I_max + (idx,)] += 1

        counter = counter + 1
        print(counter)

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
df = pd.DataFrame(data=occurance, index=names, columns=names)

ct_sel[0, 4]
ct_percent_mat[:, :, idx]


abs((dlcc.sel(time=2010).isel(ID=id) * 100)).sum()
percent_diff.sel(time=2010).isel(lat=200, lon=400)
a = dlcc.sel(time=2010).isel(ID=id).values * 100
a[np.where(a < 0)].sum()
a[np.where(a > 0)].sum()

percent = xr.open_dataset(dir + "data/processed_data/percent_cover/percent_cover.nc")[
    "__xarray_dataarray_variable__"
]
percent_diff = percent.diff("time")


# np.fill_diagonal(occurance, 0)
df.iloc[9, 9]


(occ_mat[0, 0, :] == 1).sum()
(occ_mat[0, 0, :] == 0).sum()
occ_mat[0, 0, :].sum()

occ_mat[0, 0, :][3]
np.nonzero(occ_mat[0, 0, :])
