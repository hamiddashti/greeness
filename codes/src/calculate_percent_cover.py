import os
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
import glob
import xarray as xr
import pandas as pd
import rioxarray

dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/"

# Create a sample tif file
sample = rioxarray.open_rasterio(
    dir
    + "data/raw_data/noaa_cdr/ndvi/clipped/clipped_AVHRR-Land_v005_AVH13C1_NOAA-19_20120425_c20170407181650.nc"
)["NDVI"]
# replace no data with nan
sample = sample.where(sample != -9999.0)
sample.rio.write_nodata(np.nan, inplace=True)
sample.rio.to_raster(dir + "data/raw_data/landcover/sample.tif")

years = pd.date_range(start="1984", end="2015", freq="Y").year
fname_all = []

for year in years:
    print("Reprojecting year:" + str(year))
    fname_mosaic = "data/raw_data/landcover/mosaic/mosaic_" + str(year) + ".tif"
    da = xr.open_rasterio(dir + fname_mosaic)
    da_reproj = da.rio.reproject("EPSG:4326")
    da_reproj.rio.to_raster(
        dir + "data/raw_data/landcover/mosaic/mosaic_reproject_" + str(year) + ".tif"
    )

    print("Calculating percent cover of year:" + str(year))
    fname = "data/raw_data/landcover/mosaic/mosaic_reproject_" + str(year) + ".tif"
    ds = gdal.Open(dir + fname)
    band = ds.GetRasterBand(1)
    class_ar = band.ReadAsArray()
    gt = ds.GetGeoTransform()
    pj = ds.GetProjection()
    ds = band = None  # close

    # Define the raster values for each class, to relate to each band
    class_ids = (np.arange(10) + 1).tolist()

    # Make a new bit rasters
    bit_name = (
        dir + "data/processed_data/percent_cover/" + str(year) + "_bit_raster.tif"
    )
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        bit_name,
        class_ar.shape[1],
        class_ar.shape[0],
        len(class_ids),
        gdal.GDT_Byte,
        ["NBITS=1", "COMPRESS=LZW", "INTERLEAVE=BAND"],
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(pj)
    for bidx in range(ds.RasterCount):

        band = ds.GetRasterBand(bidx + 1)
        # create boolean
        selection = class_ar == class_ids[bidx]
        band.WriteArray(selection.astype("B"))

    ds = band = None  # save, close

    # Open raster from step 1
    src_ds = gdal.Open(bit_name)

    # Open a template or copy array, for dimensions and NODATA mask
    cpy_ds = gdal.Open(dir + "data/raw_data/landcover/sample.tif")
    band = cpy_ds.GetRasterBand(1)

    # WARNING WARNING WARNING: MAKE SURE NODATA IS ACTUALY NAN
    if np.isnan(band.GetNoDataValue()):
        # cpy_mask = (band.ReadAsArray() == band.GetNoDataValue())
        cpy_mask = np.isnan(band.ReadAsArray())
        # basename = os.path.basename(f)
        outname = (
            dir
            + "data/processed_data/percent_cover/"
            + str(year)
            + "_percent_cover.tif"
        )
        # Result raster, with same resolution and position as the copy raster
        dst_ds = drv.Create(
            outname,
            cpy_ds.RasterXSize,
            cpy_ds.RasterYSize,
            len(class_ids),
            gdal.GDT_Float32,
            ["INTERLEAVE=BAND"],
        )
        dst_ds.SetGeoTransform(cpy_ds.GetGeoTransform())
        dst_ds.SetProjection(cpy_ds.GetProjection())

        # Do the same as gdalwarp -r average; this might take a while to finish
        gdal.ReprojectImage(src_ds, dst_ds, None, None, gdal.GRA_Average)

        # Convert all fractions to percent, and apply the same
        # NODATA mask from the copy raster
        NODATA = np.nan
        for bidx in range(dst_ds.RasterCount):

            band = dst_ds.GetRasterBand(bidx + 1)
            ar = band.ReadAsArray()
            ar[cpy_mask] = NODATA
            band.WriteArray(ar)
            # band.SetNoDataValue(NODATA)
        # Save and close all rasters
        src_ds = cpy_ds = dst_ds = band = None

        fname_all.append(outname)
    else:
        print("The No data value of the modis is not NAN!!")
        # break
    print(fname_all)

years = pd.date_range(start="1984", end="2015", freq="Y").year
fname_all = [
    dir + "data/processed_data/percent_cover/" + str(year) + "_percent_cover.tif"
    for year in years
]


chunks = {"y": 448, "x": 1348}
da = xr.concat([xr.open_rasterio(f, chunks=chunks) for f in fname_all], dim=years)
da = da.rename({"concat_dim": "time", "x": "lon", "y": "lat"})
da.attrs["bands"] = (
    "1:Evegreen forest; 2:Deciduous Forest; 3:Shrubland; 4:Herbaceous; 5:Sparsely Vegetated"
    "6:Barren; 7:Fen; 8:Bog; 9:Shallows/Littoral; 10:water"
)
da.to_netcdf(dir + "data/processed_data/percent_cover/percent_cover.nc")
