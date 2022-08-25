import xarray as xr
import pandas as pd
import rasterio
from rasterio.merge import merge
import glob
import numpy as np


# Read the data 
dir ="/data/home/hamiddashti/hamid/nasa_above/greeness/data/"
out_dir = dir+"raw_data/landcover/mosaic/"
fnames = glob.glob(dir + "raw_data/landcover/raw/Annual_Landcover_ABoVE_1691/data/*Simplified*")

fnames =fnames[1:4]

src_files_to_mosaic = []
for fp in fnames:
    src = rasterio.open(fp)
    src_files_to_mosaic.append(src)
src_files_to_mosaic

# Each tif image has 31 layers with each layer (index=31) representing a year from 1984-2014
years = pd.date_range(start='1984', end='1986', freq='A').year
for i in range(1, len(years) + 1):
    print(f"mosaic year: {years[i-1]}")
    mosaic, out_trans = merge(src_files_to_mosaic, indexes=[i])
    out_meta = src.meta.copy()
    # Update the metadata
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src.crs,
        "count": 1
    })

    with rasterio.open(out_dir + "mosaic_" + str(years[i - 1]) + ".tif",
                       "w",
                       **out_meta,
                       compress='lzw') as dest:
        dest.write(mosaic)