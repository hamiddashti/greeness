library(Rbeast)
library(ncdf4) # package for netcdf manipulation
library(raster) # package for raster manipulation
library(rgdal) # package for geospatial analysis
library(ggplot2) # package for plotting
library(xts)

nc_data <- nc_open("/data/home/hamiddashti/hamid/nasa_above/greeness/data/processed_data/noaa_nc/lai_fapar/resampled/lai_growing.nc")
lon <- ncvar_get(nc_data, "longitude")
lat <- ncvar_get(nc_data, "latitude", verbose = F)


lai_growing <- ncvar_get(nc_data, "LAI") # store the data in a 3-dimensional array

metadata <- list()
metadata$isRegularOrdered <- TRUE # 'imagestack$ndvi' is an IRREGULAR input
metadata$whichDimIsTime <- 3 # Which dim of the input refer to time for 3D inputs?
# 1066 is the ts length, so dim is set to '3' here.
metadata$startTime <- c(1984, 1, 1)
metadata$deltaTime <- 1 # Aggregate the irregular ts at a monthly interval:1/12 Yr
metadata$period <- 1.0 # The period is 1 year: deltaTime*freq=1/12*12=1.0

mcmc <- list()
mcmc$seed <- 10

extra <- list()
extra$numParThreads <- 0 # If 0, total_num_threads=numThreadsPerCPU*num_of_cpu_core # nolint

season <- "none"

lai_trend_out <- beast123(lai_growing, metadata = metadata, mcmc = mcmc, extra = extra, season = season) # nolint

ncp <- lai_trend_out$trend$ncp_median

dim(ncp)
a <- which(ncp == max(ncp, na.rm = T), arr.ind = T)
a
which.max(apply(ncp, 3, max))
apply(ncp, 3, function(ncp) which(ncp == max(ncp), arr.ind = TRUE))
ncp <- flip(ncp, direction = "y")
image(ncp)
dim(lai_growing)
nc_data

r <- raster(t(ncp), xmn = min(lon), xmx = max(lon), ymn = min(lat), ymx = max(lat), crs = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs+ towgs84=0,0,0"))
r <- flip(r, direction = "y")
plot(r)
















data(imagestack)
dim(imagestack$ndvi) # Dim: 12 x 9 X 1066 (row x col x time)
imagestack$datestr # A character vector of 1066 date strings
metadata <- list()
metadata$isRegularOrdered <- FALSE # 'imagestack$ndvi' is an IRREGULAR input
metadata$whichDimIsTime <- 3 # Which dim of the input refer to time for 3D inputs?
# 1066 is the ts length, so dim is set to '3' here.
metadata$time$datestr <- imagestack$datestr
metadata$time$strfmt <- "LT05_018032_20080311.yyyy-mm-dd"
metadata$deltaTime <- 1 / 12 # Aggregate the irregular ts at a monthly interval:1/12 Yr
metadata$period <- 1.0 # The period is 1 year: deltaTime*freq=1/12*12=1.0
extra <- list()
extra$dumpInputData <- TRUE # Get a copy of aggregated input ts
extra$numThreadsPerCPU <- 2 # Each cpu core will be assigned 2 threads
extra$numParThreads <- 0 # If 0, total_num_threads=numThreadsPerCPU*num_of_cpu_core

mcmc <- list()
mcmc$seed <- 10

# if >0, used to specify the total number of threads
# Default values for missing parameters
ndvi <- imagestack$ndvi
o <- beast123(ndvi, metadata = metadata, mcmc = mcmc, extra = extra)
a <- o$trend$ncp
a[3, 3]

ndvi[6, 5, ] <- NA
o2 <- beast123(ndvi, metadata = metadata, mcmc = mcmc, extra = extra)
a2 <- o2$trend$ncp
a2[3, 3]




nc_data <- nc_open("/data/home/hamiddashti/hamid/nasa_above/greeness/data/processed_data/noaa_nc/lai_fapar/resampled/lai_growing.nc")
lai_growing <- ncvar_get(nc_data, "LAI") # store the data in a 3-dimensional array

mean_lai <- apply(lai_growing, 3, mean, na.rm = TRUE)

n <- 1984:2013

opt <- list()
# opt$period=1
# opt$start=1984
# # opt$deltat=1
# out <- beast(mean_lai,opt)
# plot(out,index=n )
# axis(1,at=n,labels=str(n))

sample_lai <- lai_growing[1026, 428, ]
opt <- list()
opt$period <- 24
opt$start <- 1984
opt$deltat <- 1
opt$season <- "none"
out <- beast(sample_lai, opt)
plot(out)




lai_growing[0, 0, 0]



test[11, 11, ]
