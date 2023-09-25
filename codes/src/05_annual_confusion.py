# This script calculate the confusion table and associated LST, ET and albedo
# and save it in a netcdf file

import numpy as np
import rasterio
import fiona
import pandas as pd
import xarray as xr
from rasterio import features
from rasterio.mask import mask
import dask
from dask.diagnostics import ProgressBar
import geopandas as gpd
from datetime import datetime


def add_time_dim(xda):
    xda = xda.expand_dims(time=[datetime.now()])
    return xda


def mymask(tif, shp):
    # To mask landsat LUC pixels included in each MODIS pixel
    out_image, out_transform = rasterio.mask.mask(tif,
                                                  shp,
                                                  all_touched=False,
                                                  crop=True)
    # out_meta = tif.meta
    # return out_image,out_meta,out_transform
    return out_image, out_transform


def confusionmatrix(actual, predicted, unique, imap):
    """
    Generate a confusion matrix for multiple classification
    @params:
        actual      - a list of integers or strings for known classes
        predicted   - a list of integers or strings for predicted classes
        # normalize   - optional boolean for matrix normalization
        unique		- is the unique numbers assigned to each class
        imap		- mapping of classes

    @return:
        matrix      - a 2-dimensional list of pairwise counts
    """

    matrix = [[0 for _ in unique] for _ in unique]
    # Generate Confusion Matrix
    for p, a in list(zip(actual, predicted)):
        if (p > len(unique)) or (a > len(unique)):
            continue
        matrix[imap[p]][imap[a]] += 1
    # Matrix Normalization
    # if normalize:
    sigma = sum([sum(matrix[imap[i]]) for i in unique])
    matrix_normalized = [
        row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)
    ]
    return matrix, matrix_normalized


NUMBER_OF_CLASSES = 10  # [DF,DF,shrub,herb,sparse,wetland, water]
class_names = [
    "EF",
    "DF",
    "Shrub",
    "Herb",
    "Sparse",
    "Barren",
    "Fen",
    "Bog",
    "SL",
    "water",
]
conversion_type = []
for i in range(0, NUMBER_OF_CLASSES):
    for j in range(0, NUMBER_OF_CLASSES):
        # if (i==j):
        # 	continue
        tmp = class_names[i] + "_" + class_names[j]
        conversion_type.append(tmp)
dir = "/data/home/hamiddashti/hamid/nasa_above/greeness/"

luc_dir = dir + "data/raw_data/landcover/mosaic/"
out_dir = dir + "data/processed_data/confusion_tables/"

# changed_pixels_mask = xr.open_dataarray(
#     dir + "data/processed_data/noaa_nc/lai_fapar/trend/changed_pixels.nc"
# )

shape_file = dir + "data/shp_files/python_grid.shp"

with fiona.open(shape_file, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]
    # area = [feature["properties"]["area"] for feature in shapefile]
# Calculate area
shp_cart = gpd.read_file(shape_file)
shp_cart = shp_cart.set_crs(4326)
shp_cart = shp_cart.copy()
shp_cart = shp_cart.to_crs({"init": "epsg:3857"})
shp_cart.crs
area = shp_cart["geometry"].area / 10**6

arr = xr.open_dataarray(dir + "data/arr_id.nc")

##########

percent_cover = (
    xr.open_dataarray(dir +
                      "data/processed_data/percent_cover/percent_cover.nc") *
    100)
percent_cover = percent_cover.loc["1984":"2013"]
#percent_cover = percent_cover.round(4)
a = percent_cover.sel(time=1999)
b = percent_cover.sel(time=2000)
a[:, I[0], I[1]] / 100
percent_1
i = 15872
I = np.where(arr == i)

##########

for year in np.arange(1984, 2013):
    print(year + 1)

    luc1 = rasterio.open(luc_dir + "mosaic_reproject_" + str(year) + ".tif")
    luc2 = rasterio.open(luc_dir + "mosaic_reproject_" + str(year + 1) +
                         ".tif")
    # changed_pixels_mask_val = np.ravel(changed_pixels_mask.values, order="F")

    pix_index = []
    final_confusion = []
    final_normal_confusion = []
    final_area = []
    final_percent_1 = []
    final_percent_2 = []
    final_dlcc = []

    unique = np.arange(1, NUMBER_OF_CLASSES + 1)
    imap = {key: i for i, key in enumerate(unique)}
    for i in range(len(shapes)):
        # if changed_pixels_mask_val[i] == 0:
        #     continue
        luc1_masked = mymask(tif=luc1, shp=[shapes[i]])[0]
        luc2_masked = mymask(tif=luc2, shp=[shapes[i]])[0]
        try:
            conf_tmp, conf_normal_tmp = np.asarray(
                confusionmatrix(luc1_masked.ravel(), luc2_masked.ravel(),
                                unique, imap))
        except ZeroDivisionError:
            # This error mostly happens at the border of the study area,
            # where after clipping it with shapefile only left values are
            # 255 and 254 (i.e. nan values)
            print("ZeroDivisionError")
            continue
        count_1 = []
        count_2 = []
        for j in np.arange(1, 11):
            count_1_tmp = (luc1_masked == j).sum()
            count_1.append(count_1_tmp)
            count_2_tmp = (luc2_masked == j).sum()
            count_2.append(count_2_tmp)
        percent_1 = count_1 / (np.sum(count_1))
        percent_2 = count_2 / (np.sum(count_2))
        dlcc_val = percent_2 - percent_1
        # conf_tmp2 = np.ravel(conf_tmp, order="C")
        # conf_normal_tmp2 = np.ravel(conf_normal_tmp, order="C")
        final_confusion.append(conf_tmp)
        final_normal_confusion.append(conf_normal_tmp)

        pix_index.append(i)
        final_area.append(area[i])
        final_percent_1.append(percent_1)
        final_percent_2.append(percent_2)
        final_dlcc.append(dlcc_val)

    pix_index = np.array(pix_index)
    final_confusion = np.array(final_confusion)
    final_normal_confusion = np.array(final_normal_confusion)

    final_area = np.array(final_area)
    final_percent_1 = np.array(final_percent_1)
    final_percent_2 = np.array(final_percent_2)
    final_dlcc = np.array(final_dlcc)

    ds = xr.Dataset(
        data_vars={
            "CONFUSION": (("ID", "LC_t1", "LC_t2"), final_confusion),
            "NORMALIZED_CONFUSION":
            (("ID", "LC_t1", "LC_t2"), final_normal_confusion),
            "DLCC": (("ID", "LC"), final_dlcc),
            "LC_2003": (("ID", "LC"), final_percent_1),
            "LC_2013": (("ID", "LC"), final_percent_2),
            "PIX_INDEX": (("ID"), pix_index),
            "Area": (("ID"), final_area),
        },
        coords={
            "ID": range(len(final_area)),
            "LC_t1": range(1, 11),
            "LC_t2": range(1, 11),
            "LC": range(1, 11),
        },
    )

    ds.to_netcdf((dir + "/data/processed_data/confusion_tables/ct_corrected_" +
                  str(year + 1) + ".nc"))

# Collect all dataset in one along time dimension
data = xr.open_mfdataset(
    dir + "/data/processed_data/confusion_tables/ct_corrected_*",
    preprocess=add_time_dim,
)
t = pd.date_range(start="1985", end="2014", periods=None, freq="A-DEC")
data["time"] = t
data.to_netcdf(
    dir + "/data/processed_data/confusion_tables/ct_all_years_corrected.nc")
