#-----------------------------------------------------------------
# 1.1 Set up python 

# install.packages("reticulate")
library(reticulate)

# path to where python is installed
use_python("C:/WBG/Anaconda/python.exe")

# path to environment with packages installed
use_condaenv("pkg")

# path to python code
repo_dir <- "C:/Users/wb514197/Repos/Spatial-Anonymization"
python_path <- file.path(repo_dir, 'src', 'point_displacement.py')
source_python(python_path)

#-----------------------------------------------------------------
# 1.2 Input/output data 

DATA = file.path("X:/data/surveys")

# ========================
# INPUT FILES FOR ETHIOPIA
# ========================
ETH_SURVEY = file.path(DATA, 'ETH', 'survey')
ETH_GEOVARS = file.path(DATA, 'ETH', 'w4_geovars')

# GPS coordinates file
ETH_COORDS = file.path(ETH_SURVEY, "HH level coordinates.dta")

# Admin regions (adm2) shapefile
ETH_ADM = file.path(ETH_GEOVARS, "spatial_data", "shp_adm2", "eth_admbnda_adm2_csa_bofed_20190827.shp")

# output directory
ETH_OUPUT_DIR = file.path(DATA, "ETH", "outputs")

#-----------------------------------------------------------------
# 1.3 Parameters and other global variables setup
# 
# Now, we setep parameters as well as helper global variables as follows:
#   
#   1.3.1  Displacement parameters
# All distance paramters are in meters
# 
# urb_max_dist. The maximum distance to displace a point in urban areas.
# rur_max_dist. The maximum distance to displace a point in rural areas.
# rur_max_dist_subset. The maximum distance to displace a point in rural areas for a subset of the households.
# urb_min_dist. Allowed minimum distance between original point and displaced point in urban areas.
# rur_min_dist. Allowed minimum distance between original point and displaced point in rural areas.
# apply_min_dist. whether to apply minimum distance threshold or not
# rur_subset_prop. proportion of rural households to apply special maximum distance
# 1.3.2  Processing variables/args
# geodetic_crs_epsg. The geographic CRS for which the latitude and longitude were collected. The default is WGS84 given as EPSG code. If you have the datum-name or CRS details but do not know the EPSG code, please use epsg.io
# lon_col. Column name containing longitude
# lat_col. Column name containing latitude
# uniq_id. Column name (if available uniquely identifying each row
# urb_rur_col. Column name indicating whether point is in urban or rural
# max_tries. In some cases where there are many coordinates and the confining polygon is small, finding displaced coordinates withing the polygon takes too long and we give up after these many tries.

# =====================
# SETUP PARAMS 
# ====================
# urb_max_dist
urb_max = 2000
# rur_max_dist
rur_max = 5000
# rur_max_dist_subset
rur_max_subset = 10000
# urb_min_dist
urb_min = 100
# rur_min_dist
rur_min = 1000
# apply_min_dist
apply_min_dist = FALSE
# rur_subset_prop = 10%
rur_sub_prop = 0.1
# max_tries
max_tries = 30

PARAMS <- dict(
  'urb_max_dist'= urb_max, 
  'rur_max_dist'= rur_max, 
  'rur_max_dist_subset'=rur_max_subset,
  'urb_min_dist'= urb_min, 
  'rur_min_dist'= rur_min, 
  'maximum_tries'= max_tries,
  'apply_min_dist'= apply_min_dist, 
  'rur_sub_prop'= rur_sub_prop
)

# ========================
# PREPARE PROCESSING ARGS
# =======================
# add EPSG code for geographic CRS if its not 
# default to WGS84
geo_crs = as.integer(4326)
lon ='saq19__Longitude'
lat ='saq19__Latitude'
hhid = 'household_id'
urb_rur = 'rural_urban'
rural_code = 1

#-----------------------------------------------------------------
# 1.4 Run the displacement

# Create displacer object
eth_displacer = SimpleDisplacer(PARAMS, input_coords=ETH_COORDS,
                               shp_adm_bounds=ETH_ADM, output_dir=ETH_OUPUT_DIR,
                               lon_col=lon, lat_col=lat, urb_rur_col=urb_rur, 
                               unique_id=hhid, rural_code=rural_code, 
                               geodetic_crs=geo_crs)

# Run the displacer
eth_displacer$run_displacer() 