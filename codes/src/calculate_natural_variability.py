import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar


dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/"
out_dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/working/"


def dist_matrix(x_size, y_size):
    import numpy as np

    a1 = np.floor(x_size / 2)
    a2 = np.floor(y_size / 2)
    x_arr, y_arr = np.mgrid[0:x_size, 0:y_size]
    cell = (a1, a2)
    dists = np.sqrt((x_arr - cell[0]) ** 2 + (y_arr - cell[1]) ** 2)
    dists[int(a1), int(a2)] = np.nan
    return dists


WINSIZE = 51
dist_m = dist_matrix(WINSIZE, WINSIZE)

percent_cover = xr.open_dataarray(
    dir + "data/processed_data/percent_cover/percent_cover.nc"
)
var_ts = xr.open_dataarray(
    dir + "data/processed_data/noaa_nc/lai_fapar/resampled/lai_growing.nc"
)





percent_cover = percent_cover.rename({"x": "lon", "y": "lat"})
LST = xr.open_dataarray(dir + "LST_Final/LST/Annual_Mean/lst_mean_annual.nc")
LST_DAY = xr.open_dataarray(dir + "LST_Final/LST/Annual_Mean/lst_day_annual.nc")
LST_NIGHT = xr.open_dataarray(dir + "LST_Final/LST/Annual_Mean/lst_night_annual.nc")
ET = xr.open_dataarray(dir + "ET_Final/Annual_ET/ET_Annual.nc")
ET = ET.rename({"x": "lon", "y": "lat"})
ALBEDO = xr.open_dataarray(dir + "ALBEDO_Final/Annual_Albedo/Albedo_annual.nc")
ALBEDO = ALBEDO.rename({"x": "lon", "y": "lat"})

luc = percent_cover.loc[2003:2013]
lst = LST.loc[2003:2013]
lst_day = LST_DAY.loc[2003:2013]
lst_night = LST_NIGHT.loc[2003:2013]
et = ET.loc[2003:2013]
albedo = ALBEDO.loc[2003:2013]

# lst = lst.isel(lat=range(1400, 1600), lon=range(4400, 4600))
# lst_day = lst_day.isel(lat=range(1400, 1600), lon=range(4400, 4600))
# lst_night = lst_night.isel(lat=range(1400, 1600), lon=range(4400, 4600))
# luc = luc.isel(lat=range(1400, 1600), lon=range(4400, 4600))
# et = et.isel(lat=range(1400, 1600), lon=range(4400, 4600))
# albedo = albedo.isel(lat=range(1400, 1600), lon=range(4400, 4600))

dluc = luc.loc[2013] - luc.loc[2003]
dluc_abs = abs(dluc)
tmp = xr.ufuncs.isnan(dluc_abs.where((luc.loc[2013] == 0) & (luc.loc[2003] == 0)))
# To convert tmp from True/False to one/zero
mask = tmp.where(tmp == True)
dluc_abs = dluc_abs * mask
changed_pixels = (dluc_abs > 1).any("band") * 1

dlst_total = lst.loc[2013] - lst.loc[2003]
dlst_day_total = lst_day.loc[2013] - lst_day.loc[2003]
dlst_night_total = lst_night.loc[2013] - lst_night.loc[2003]
det_total = et.loc[2013] - et.loc[2003]
dalbedo_total = albedo.loc[2013] - albedo.loc[2003]

dlst_total_roll = (
    dlst_total.rolling({"lat": WINSIZE, "lon": WINSIZE}, center=True)
    .construct({"lat": "lat_dim", "lon": "lon_dim"})
    .values
)

dlst_day_total_roll = (
    dlst_day_total.rolling({"lat": WINSIZE, "lon": WINSIZE}, center=True)
    .construct({"lat": "lat_dim", "lon": "lon_dim"})
    .values
)

dlst_night_total_roll = (
    dlst_night_total.rolling({"lat": WINSIZE, "lon": WINSIZE}, center=True)
    .construct({"lat": "lat_dim", "lon": "lon_dim"})
    .values
)

det_total_roll = (
    det_total.rolling({"lat": WINSIZE, "lon": WINSIZE}, center=True)
    .construct({"lat": "lat_dim", "lon": "lon_dim"})
    .values
)
dalbedo_total_roll = (
    dalbedo_total.rolling({"lat": WINSIZE, "lon": WINSIZE}, center=True)
    .construct({"lat": "lat_dim", "lon": "lon_dim"})
    .values
)

changed_pixels_roll = (
    changed_pixels.rolling({"lat": WINSIZE, "lon": WINSIZE}, center=True)
    .construct({"lat": "lat_dim", "lon": "lon_dim"})
    .values
)

dlst_res = xr.full_like(changed_pixels, fill_value=np.nan, dtype=float)
dlst_day_res = xr.full_like(changed_pixels, fill_value=np.nan, dtype=float)
dlst_night_res = xr.full_like(changed_pixels, fill_value=np.nan, dtype=float)
det_res = xr.full_like(changed_pixels, fill_value=np.nan, dtype=float)
dalbedo_res = xr.full_like(changed_pixels, fill_value=np.nan, dtype=float)
for i in range(0, changed_pixels.shape[0]):
    for j in range(0, changed_pixels.shape[1]):
        print(i)
        if changed_pixels[i, j] == 0:
            continue
        mask = changed_pixels_roll[i, j]
        dist_m_mask = dist_m[np.where(mask == 0)]

        if len(dist_m_mask) == 0:
            continue

        dlst_tmp = dlst_total_roll[i, j]
        dlst_mask = dlst_tmp[np.where(mask == 0)]
        tmp_var1 = np.nansum(dlst_mask / dist_m_mask)
        dlst_res[i, j] = tmp_var1 / (np.nansum(dist_m_mask))

        dlst_day_tmp = dlst_day_total_roll[i, j]
        dlst_day_mask = dlst_day_tmp[np.where(mask == 0)]
        tmp_var2 = np.nansum(dlst_day_mask / dist_m_mask)
        dlst_day_res[i, j] = tmp_var2 / (np.nansum(dist_m_mask))

        dlst_night_tmp = dlst_night_total_roll[i, j]
        dlst_night_mask = dlst_night_tmp[np.where(mask == 0)]
        tmp_var3 = np.nansum(dlst_night_mask / dist_m_mask)
        dlst_night_res[i, j] = tmp_var3 / (np.nansum(dist_m_mask))

        det_tmp = det_total_roll[i, j]
        det_mask = det_tmp[np.where(mask == 0)]
        tmp_var4 = np.nansum(det_mask / dist_m_mask)
        det_res[i, j] = tmp_var4 / (np.nansum(dist_m_mask))

        dalbedo_tmp = dalbedo_total_roll[i, j]
        dalbedo_mask = dalbedo_tmp[np.where(mask == 0)]
        tmp_var5 = np.nansum(dalbedo_mask / dist_m_mask)
        dalbedo_res[i, j] = tmp_var5 / (np.nansum(dist_m_mask))

dlst_mean_percent_coverc = dlst_total - dlst_res
dlst_day_percent_coverc = dlst_day_total - dlst_day_res
dlst_night_percent_coverc = dlst_night_total - dlst_night_res
det_percent_coverc = det_total - det_res
dalbedo_percent_coverc = dalbedo_total - dalbedo_res

dlst_mean_percent_coverc.to_netcdf(out_dir + "dlst_mean_percent_coverc.nc")
dlst_total.to_netcdf(out_dir + "dlst_mean_total.nc")
dlst_res.to_netcdf(out_dir + "dlst_mean_nv.nc")

dlst_day_percent_coverc.to_netcdf(out_dir + "dlst_day_percent_coverc.nc")
dlst_day_total.to_netcdf(out_dir + "dlst_day_total.nc")
dlst_day_res.to_netcdf(out_dir + "dlst_day_nv.nc")

dlst_night_percent_coverc.to_netcdf(out_dir + "dlst_night_percent_coverc.nc")
dlst_night_total.to_netcdf(out_dir + "dlst_night_total.nc")
dlst_night_res.to_netcdf(out_dir + "dlst_night_nv.nc")

det_percent_coverc.to_netcdf(out_dir + "det_percent_coverc.nc")
det_total.to_netcdf(out_dir + "det_total.nc")
det_res.to_netcdf(out_dir + "det_nv.nc")

dalbedo_percent_coverc.to_netcdf(out_dir + "dalbedo_percent_coverc.nc")
dalbedo_total.to_netcdf(out_dir + "dalbedo_total.nc")
dalbedo_res.to_netcdf(out_dir + "dalbedo_nv.nc")

changed_pixels.to_netcdf(out_dir + "changed_pixels_mask.nc")
dluc.to_netcdf(out_dir + "dluc.nc")

# def my_fun(i, j):
#     # for i in range(0, changed_pixels.shape[0]):
#     #     for j in range(0, changed_pixels.shape[1]):
#     print(i)
#     if changed_pixels[i, j] == 0:
#         return
#     mask = changed_pixels_roll.isel(lat=i, lon=j)
#     dlst_tmp = dlst_total_roll.isel(lat=i, lon=j)
#     dlst_mask = dlst_tmp.where(mask == 1)
#     tmp_var1 = (dlst_mask / dist_m).sum()
#     dlst_res_tmp = tmp_var1 / (np.nansum(dist_m[np.where(mask == 1)]))
#     dlst_res[i, j] = dlst_res_tmp.values

#     det_tmp = det_total_roll.isel(lat=i, lon=j)
#     det_mask = det_tmp.where(mask == 1)
#     tmp_var2 = (det_mask / dist_m).sum()
#     det_res_tmp = tmp_var2 / (np.nansum(dist_m[np.where(mask == 1)]))
#     det_res[i, j] = det_res_tmp.values

#     dalbedo_tmp = dalbedo_total_roll.isel(lat=i, lon=j)
#     dalbedo_mask = dalbedo_tmp.where(mask == 1)
#     tmp_var3 = (dalbedo_mask / dist_m).sum()
#     dalbedo_res_tmp = tmp_var3 / (np.nansum(dist_m[np.where(mask == 1)]))
#     dalbedo_res[i, j] = dalbedo_res_tmp.values
# t1 = time.time()
# delayed_results = []
# for i in range(0, changed_pixels.shape[0]):
#     for j in range(0, changed_pixels.shape[1]):
#         delayed_my_fun = dask.delayed(my_fun)(i, j)
#         delayed_results.append(delayed_my_fun)
# with ProgressBar():
#     results = dask.compute(*delayed_results)
# t2 = time.time()
# print(f"Time passed is:{t2-t1}")
