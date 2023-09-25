# prepare the libraries and data paths
import xarray as xr
import numpy as np
import rioxarray
from rasterio.enums import Resampling
import glob
import pandas as pd
import time

dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/"

in_dir = dir + ("data/processed_data/landsat/mosaic/")
out_dir = dir + ("data/processed_data/landsat/resampled/")

target_image = xr.open_dataset(
    dir +
    "data/processed_data/noaa_nc/lai_fapar/resampled/lai_annual_resample_max_1984_2013.nc"
)["LAI"].isel(time=5)
target_image = target_image.rio.write_crs("EPSG:4326").rename({
    "latitude": "y",
    "longitude": "x"
})



#Resampling NDVI TM5 data
for i in np.arange(1984, 2012):
    print("TM5 year:" + str(i))
    source_image = xr.open_dataarray(
        in_dir + "mosaic_NDVI_" + str(i) + "_TM5.tif", engine="rasterio"
    )
    resampled = source_image.rio.reproject_match(
        target_image, resampling=Resampling.average
    )
    resampled.squeeze().to_netcdf(out_dir + "NDVI_resampled_" + str(i) + "_TM5.nc")
    del resampled

for i in np.arange(2011, 2014):
    print("ETM year:" + str(i))
    source_image = xr.open_dataarray(
        in_dir + "mosaic_NDVI_" + str(i) + "_ETM.tif", engine="rasterio"
    )/10000
    source_image_corr = (source_image+0.015)/1.095
    resampled = source_image_corr.rio.reproject_match(
        target_image, resampling=Resampling.average
    )
    resampled = resampled*10000
    resampled.squeeze().to_netcdf(out_dir + "NDVI_resampled_" + str(i) + "_ETM.nc")
    del resampled

fnames = []
for i in np.arange(1984,2011):
    if i<1999:
        fnames.append(dir+"data/processed_data/landsat/resampled/NDVI_resampled_"+str(i)+"_TM5.nc")
        continue
    fnames.append(dir+"data/processed_data/landsat/resampled/NDVI_resampled_"+str(i)+"_ETM.nc")

ds_ndvi = xr.concat([xr.open_dataset(f)["band_data"] for f in fnames],dim="time")
ds_ndvi["time"] = pd.date_range("1984","2010",freq = "A")
ds_ndvi = ds_ndvi.drop("band").rename({"x":"lon","y":"lat"})
ds_ndvi.to_netcdf(dir+"data/processed_data/landsat/resampled/ndvi_annual_max_correct_2010.nc")



# Resampling RED band TM5 data
# for i in np.arange(1984, 2012):
#     print("TM5 Red band year:" + str(i))
#     source_image = xr.open_dataarray(
#         in_dir + "mosaic_Red_" + str(i) + "_TM5.tif", engine="rasterio"
#     )
#     resampled = source_image.rio.reproject_match(
#         target_image, resampling=Resampling.average
#     )
#     resampled.squeeze().to_netcdf(out_dir + "Red_resampled_" + str(i) + "_TM5.nc")
#     del resampled

for i in np.arange(1999, 2014):
    print("ETM Red band year:" + str(i))
    source_image = xr.open_dataarray(
        in_dir + "mosaic_Red_" + str(i) + "_ETM.tif", engine="rasterio"
    )
    source_image_corr = (source_image+0.004)/1.052
    
    resampled = source_image.rio.reproject_match(
        target_image, resampling=Resampling.average
    )

    resampled.squeeze().to_netcdf(out_dir + "Red_resampled_" + str(i) + "_ETM.nc")
    del resampled

# Resampling NIR band TM5 data
# for i in np.arange(2011, 2012):
#     print("TM5 NIR band year:" + str(i))
#     source_image = xr.open_dataarray(
#         in_dir + "mosaic_NIR_" + str(i) + "_TM5.tif", engine="rasterio"
#     )
#     resampled = source_image.rio.reproject_match(
#         target_image, resampling=Resampling.average
#     )
#     resampled.squeeze().to_netcdf(out_dir + "NIR_resampled_" + str(i) + "_TM5.nc")
#     del resampled

# for i in np.arange(1985, 2014):
#     print("NIR band year:" + str(i))
#     if i < 1999:
#         source_image = xr.open_dataarray(in_dir + "mosaic_NIR_" + str(i) +
#                                          "_ETM.tif",
#                                          engine="rasterio")
#         tmp = source_image.where(source_image != -10000)
#         resampled = tmp.rio.reproject_match(target_image,
#                                             resampling=Resampling.average)
#         resampled.squeeze().to_netcdf(out_dir + "NIR_resampled_corrected_" +
#                                       str(i) + "_ETM.nc")
#         del resampled
#         continue
#     source_image = xr.open_dataarray(in_dir + "mosaic_NIR_" + str(i) +
#                                      "_ETM.tif",
#                                      engine="rasterio")
#     tmp = source_image.where(source_image != -10000)
#     resampled = tmp.rio.reproject_match(target_image,
#                                         resampling=Resampling.average)
#     resampled.squeeze().to_netcdf(out_dir + "NIR_resampled_corrected_" +
#                                   str(i) + "_ETM.nc")
#     del resampled

fnames_nir = []
for i in np.arange(1984, 2014):
    if i < 1999:
        fnames_nir.append(
            dir + "data/processed_data/landsat/resampled/NIR_resampled_" +
            str(i) + "_TM5.nc")
        continue
    fnames_nir.append(dir +
                      "data/processed_data/landsat/resampled/NIR_resampled_" +
                      str(i) + "_ETM.nc")
ds_nir = xr.concat([xr.open_dataset(f)["band_data"] for f in fnames_nir],
                   dim="time")
ds_nir['time'] = pd.date_range("1984", "2014", freq="A")
ds_nir = ds_nir.rename({"x": "lon", "y": "lat"}).drop("band")
ds_nir.to_netcdf(out_dir + "nir.nc")

for i in np.arange(1984, 2014):
    print("NIRV band year:" + str(i))
    if i < 1999:
        nir = xr.open_dataarray(in_dir + "mosaic_NIR_" + str(i) + "_TM5.tif",
                                engine="rasterio")
        # tmp = source_image.where(source_image != -10000)

        ndvi = xr.open_dataarray(in_dir + "mosaic_NDVI_" + str(i) + "_TM5.tif",
                                 engine="rasterio")
        NIRV = ndvi * nir

        resampled = NIRV.rio.reproject_match(target_image,
                                             resampling=Resampling.average)
        resampled.squeeze().to_netcdf(out_dir + "NIRV_resampled_" + str(i) +
                                      "_TM5.nc")
        del resampled
        continue
    nir = xr.open_dataarray(in_dir + "mosaic_NIR_" + str(i) + "_ETM.tif",
                            engine="rasterio")
    # tmp = source_image.where(source_image != -10000)

    ndvi = xr.open_dataarray(in_dir + "mosaic_NDVI_" + str(i) + "_ETM.tif",
                             engine="rasterio")
    NIRV = ndvi * nir

    resampled = NIRV.rio.reproject_match(target_image,
                                         resampling=Resampling.average)
    resampled.squeeze().to_netcdf(out_dir + "NIRV_resampled_" + str(i) +
                                  "_ETM.nc")
    del resampled

fnames_nirv = []
for i in np.arange(1984, 2014):
    if i < 1999:
        fnames_nirv.append(
            dir + "data/processed_data/landsat/resampled/NIRV_resampled_" +
            str(i) + "_TM5.nc")
        continue
    fnames_nirv.append(dir +
                      "data/processed_data/landsat/resampled/NIRV_resampled_" +
                      str(i) + "_ETM.nc")

ds_nirv = xr.concat([xr.open_dataset(f)["band_data"] for f in fnames_nirv],
                   dim="time")
ds_nirv['time'] = pd.date_range("1984", "2014", freq="A")
ds_nirv = ds_nirv.rename({"x": "lon", "y": "lat"}).drop("band")
ds_nirv.to_netcdf(out_dir + "nirv.nc")


# Resampling spring onset
# for i in np.arange(2000, 2014):
#     print(" band year:" + str(i))
#     source_image = xr.open_dataarray(in_dir + "mosaic_spr_" + str(i) + ".tif",
#                                      engine="rasterio")
#     resampled = source_image.rio.reproject_match(target_image,
#                                                  resampling=Resampling.average)
#     resampled.squeeze().to_netcdf(out_dir + "spr_resampled_" + str(i) + ".nc")
#     del resampled

# Resampling autmn onset
# for i in np.arange(2001, 2014):
#     print(" band year:" + str(i))
#     source_image = xr.open_dataarray(in_dir + "mosaic_aut_" + str(i) + ".tif",
#                                      engine="rasterio")
#     resampled = source_image.rio.reproject_match(target_image,
#                                                  resampling=Resampling.average)
#     resampled.squeeze().to_netcdf(out_dir + "aut_resampled_" + str(i) + ".nc")
#     del resampled





# fnames_aut = []
# fnames_spr = []
# for i in np.arange(1985, 2014):
#     fnames_aut.append(dir +
#                       "data/processed_data/landsat/resampled/aut_resampled_" +
#                       str(i) + ".nc")
#     fnames_spr.append(dir +
#                       "data/processed_data/landsat/resampled/spr_resampled_" +
#                       str(i) + ".nc")

# ds_aut = xr.concat([xr.open_dataset(f)["band_data"] for f in fnames_aut],
#                    dim="time")
# ds_aut['time'] = pd.date_range("1985", "2013", freq="A")
# ds_spr = xr.concat([xr.open_dataset(f)["band_data"] for f in fnames_spr],
#                    dim="time")
# ds_spr['time'] = pd.date_range("1985", "2013", freq="A")

# ds_spr = ds_spr.rename({"x": "lon", "y": "lat"})
# ds_aut = ds_aut.rename({"x": "lon", "y": "lat"})

# ds_aut.to_netcdf(dir + "data/processed_data/landsat/resampled/aut.nc")
# ds_spr.to_netcdf(dir + "data/processed_data/landsat/resampled/spr.nc")


# fnames_nirv = []
# for i in np.arange(1985, 2014):
#     if i < 1999:
#         fnames_nirv.append(
#             dir + "data/processed_data/landsat/resampled/NIRV_resampled_" +
#             str(i) + "_TM5.nc")
#         continue
#     fnames_nirv.append(dir +
#                       "data/processed_data/landsat/resampled/NIRV_resampled_" +
#                       str(i) + "_ETM.nc")
# ds_nirv = xr.concat([xr.open_dataset(f)["band_data"] for f in fnames_nirv],
#                    dim="time")
# ds_nirv['time'] = pd.date_range("1985", "2014", freq="A")
# ds_nirv = ds_nirv.rename({"x": "lon", "y": "lat"}).drop("band")
# ds_nirv.to_netcdf(out_dir + "nirv.nc")