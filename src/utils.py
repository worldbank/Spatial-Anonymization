"""
Contains miscellaneous functions for spatia data processing
Created on Mar 6, 2023
@author-1: Dunstan Matekenya
@author-2: Andres
"""

import math
import rasterio
import rasterio.mask

import geopandas as gpd
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime


def extract_impact_observ_lulc_tif(adm0_shp, output_dir, utm_zones, yr=None):
    """
    Extract URL of required TIF from https://registry.opendata.aws/io-lulc/ by overlaying with country polygon
    adm0_shp(str) - path to shapefile of country boundary (admin level 0)
    output_dir - Directory to temporarily hold downloaded TIFs
    utm_zones(list) - UTM zones of area of interest
    yr(int) - Year i format YYYY for which year of data to use

    """
    # ====================
    # SETUP S3 BUCKET
    # ===================
    base_url = 's3://io-10m-annual-lulc/'
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    my_bucket = s3.Bucket('io-10m-annual-lulc')

    # ===================================
    # EXTRACT ALL FILES FROM THE BUCKET
    # ===================================
    objs = []
    for my_bucket_object in my_bucket.objects.all():
        objs.append(my_bucket_object.key)

    # set the year
    all_yrs = []
    for obj in objs:
        try:
            this_yr = datetime.strptime(obj[4:12], '%Y%m%d')
            all_yrs.append(this_yr)
        except:
            pass

    yrs = sorted(set(all_yrs), reverse=True)

    if yr and (yr in [y.year for y in yrs]):
        target_yr = yr
    else:
        target_yr = yrs[0].year

    # ===================================
    # GET TIF FOR TARGET ZONES
    # ===================================
    objs_target_yr = [obj for obj in objs if obj[4:8] == str(target_yr)]

    # Load the shapefile
    gdf = gpd.read_file(adm0_shp)
    shape = gdf.iloc[0]['geometry']

    # keep filename of rasters we found here
    raster_list = []
    for z in utm_zones:
        # Get all objects for this zone
        obj_list = [i for i in objs_target_yr if z in i]

        for file in obj_list:
            # setup output file
            out_tif = output_dir.joinpath(file)
            if out_tif.exists():
                pass
            else:
                # first, download the file
                my_bucket.download_file(Key=file, Filename=str(out_tif))
            with rasterio.open(out_tif) as tile:
                try:
                    # Ensure raster and shape has same CRS
                    assert tile.crs.to_epsg() == gdf.crs.to_epsg()
                except:
                    gdf2 = gdf.to_crs(tile.crs.to_epsg())
                    shape = gdf2.iloc[0]['geometry']
                try:
                    out_image, out_transform = rasterio.mask.mask(tile, shape, crop=True)
                    raster_list.append(output_dir.joinpath(file))
                    break
                except ValueError:
                    print(f"No overlap found for {tile.files[0]}. Try the next one ..")
                    continue

    return raster_list


def convert_wgs_coordinates_to_utm(lon, lat):
    """
    Given latitude, longitude coordinates based on WGS84, convert to
    UTM zone and EPSG code. Shamelessly copied from here:
    https://stackoverflow.com/questions/40132542/
    Parameters
    ----------
    lon - input longitude
    lat - input latitude

    Returns EPSG code
    -------

    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code


def get_utm_zones_from_geographic_dataframe(gdf, datum='WGS 84'):
    """
    Returns estimated UTM CRS from the geopandas dataframe
    Parameters
    ----------
    gdf - input geopandas dataframe

    Returns
    -------

    """

    # get layer bounds
    min_lat, min_lon, max_lat, max_lon = gdf.total_bounds

    utm_crs_list = query_utm_crs_info(
        datum_name=datum,
        area_of_interest=AreaOfInterest(
            west_lon_degree=min_lon,
            south_lat_degree=min_lat,
            east_lon_degree=max_lon,
            north_lat_degree=max_lat,
        ),
    )
    utm_crs_code_list = [i[1] for i in utm_crs_list]

    return utm_crs_code_list
