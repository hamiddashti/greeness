import xarray as xr
import pandas as pd
import rasterio
from rasterio.merge import merge
import glob
import numpy as np

# Read raw data

dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/"

in_dir = (
    "/data/ABOVE/ABoVE_Final_Data/landsat/phenology/orders/"
    "dc6b1f56b7619c37e1a4b6fec7ce3dcc/Annual_Seasonality_Greenness/data/")
out_dir = dir + "data/processed_data/landsat/mosaic/"

years = pd.date_range(start="1984", end="2015", freq="A").year

# Spring onset and autumn onset

# var_names = ["spr", "aut"]
# for i in var_names:
#     for j in np.arange(0, len(years)):

#         # Read all the file names
#         fnames = glob.glob(in_dir + "*pheno." + i + ".tif")

#         src_files_to_mosaic = []

#         # Open all files
#         for fp in fnames:
#             src = rasterio.open(fp)
#             src_files_to_mosaic.append(src)

#         # Start mosaicing
#         print(
#             "Mosaicing sesonal,year:"
#             + str(years[j])
#             + ",var:"
#             + i
#             + ",#scenes:"
#             + str(len(fnames))
#         )

#         mosaic, out_trans = merge(
#             src_files_to_mosaic, indexes=[j + 1]
#         )  # note bands start from 1
#         out_meta = src.meta.copy()
#         # Update the metadata
#         out_meta.update(
#             {
#                 "driver": "GTiff",
#                 "height": mosaic.shape[1],
#                 "width": mosaic.shape[2],
#                 "transform": out_trans,
#                 "crs": src.crs,
#                 "count": 1,
#             }
#         )

#         # Save the mosaic
#         with rasterio.open(
#             out_dir + "mosaic_" + i + "_" + str(years[j]) + ".tif",
#             "w",
#             **out_meta,
#             compress="lzw"
#         ) as dest:
#             dest.write(mosaic)

#         # Free memory
#         del mosaic, out_meta, src_files_to_mosaic

# b_lst = [1, 3, 4]  # Bands to be selected for mosaicing (i.e. ndvi,red, nir)
# b_lst_names = ["NDVI", "Red", "NIR"]
b_lst = [6]  # Bands to be selected for mosaicing (i.e. ndvi,red, nir)
b_lst_names = ["SWIR2"]

years_tm5 = pd.date_range(start="1984", end="2012", freq="A").year
for i in np.arange(0, len(years_tm5)):
    for j in np.arange(0, len(b_lst)):

        # Read all the file names
        fnames = glob.glob(in_dir + "*tm5." + str(years_tm5[i]) + ".tif")
        src_files_to_mosaic = []

        # Open all files
        for fp in fnames:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        # Start mosaicing
        print("Mosaicing et5,year:" + str(years_tm5[i]) + ",var:" +
              str(b_lst_names[j]) + ",#scenes:" + str(len(fnames)))

        mosaic, out_trans = merge(src_files_to_mosaic, indexes=[b_lst[j]])
        out_meta = src.meta.copy()
        # Update the metadata
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src.crs,
            "count": 1,
        })

        # Save the mosaic
        with rasterio.open(out_dir + "mosaic_" + b_lst_names[j] + "_" +
                           str(years_tm5[i]) + "_TM5.tif",
                           "w",
                           **out_meta,
                           compress="lzw",
                           BIGTIFF='YES') as dest:
            dest.write(mosaic)

        # Free memory
        del mosaic, out_meta, src_files_to_mosaic

years_etm = pd.date_range(start="1999", end="2015", freq="A").year
for i in np.arange(0, len(years_etm)):
    for j in np.arange(0, len(b_lst)):

        # Read all the file names
        fnames = glob.glob(in_dir + "*etm." + str(years_etm[i]) + ".tif")
        src_files_to_mosaic = []

        # Open all files
        for fp in fnames:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)
        # Start mosaicing
        print("Mosaicing ETM,year:" + str(years_etm[i]) + ",var:" +
              str(b_lst_names[j]) + ",#scenes:" + str(len(fnames)))

        mosaic, out_trans = merge(src_files_to_mosaic, indexes=[b_lst[j]])
        out_meta = src.meta.copy()
        # Update the metadata
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src.crs,
            "count": 1,
        })
        # Save the mosaic
        with rasterio.open(out_dir + "mosaic_" + b_lst_names[j] + "_" +
                           str(years_etm[i]) + "_ETM.tif",
                           "w",
                           **out_meta,
                           compress="lzw",
                           BIGTIFF='YES') as dest:
            dest.write(mosaic)
        del mosaic, out_meta, src_files_to_mosaic