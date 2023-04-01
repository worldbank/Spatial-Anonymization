"""
Contains miscellaneous functions for spatia data processing
Created on Mar 6, 2023
@author-1: Dunstan Matekenya
@author-2: Andres
"""
from pathlib import Path
import shutil
import zipfile
from datetime import datetime
import json
import requests
import math
import rasterio
from rasterio.mask import mask

import skimage.graph as graph
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import shape
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import boto3
from botocore import UNSIGNED
from botocore.config import Config

import market_access as ma
import point_displacement as pod


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
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds

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


def distance_between_points(pt1=None, pt2=None):
    """
    Calculate the Haversine distance betweeen 2 points.

    Parameters
    ----------
    pt1 : tuple of float
        (lat, long)
    pt2 : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> pt1 = (48.1372, 11.5756)  # Munich
    >>> pt2 = (52.5186, 13.4083)  # Berlin
    >>> round(distance_between_points(pt1, pt2), 1)
    504.2
    """
    lat1, lon1 = pt1
    lat2, lon2 = pt2
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


def distance_matrix(xy_list=None):
    """
    Return distance matrix from a dictlist of xy coordinates
    :param xy_list:
    :return: a dataframe style of distance matrix
    """
    df = pd.DataFrame([dict(d._asdict()) for d in xy_list])

    for pt in xy_list:
        pt_id = pt.point_id
        colname = 'to_' + str(pt_id)
        df[colname] = df.apply(lambda x: distance_between_points(
            pt1=(x['x'], x['y']), pt2=(pt.x, pt.y)), axis=1)

    return df


def distance_to_points(pts_list, target_pt, output='dist_list'):    
    """
    Measure distance from target_pt to all points in pts_list.
    Parameters
    ----------
    pts_list(list) - A list of points provided as lat,lon tuples like this: [(lat,lon)]
    target_pt(tupl) - A target point to measure distance from provided as a (lat,lon)
    output - Whether to output the distance list, minimum distance or maximum

    Returns A list containing floats
    -------
    """
    df = pd.DataFrame(pts_list, columns=['lat', 'lon'])
    df['dist'] = df.apply(lambda x: distance_between_points(pt1=(x['lat'], x['lon']), 
                                                            pt2=(target_pt[0], target_pt[1])), axis=1)
    
    if output == 'nearest':
        return df['dist'].min()
    elif output == 'dist_list':
        return list(df['dist'].values)
    else:
        return list(df['dist'].values)


def load_csv_into_geopandas(csv_with_coords, lat, lon):
    """
    Helper function to convert a CSV file with lat, lon into Geopandas Geodataframe
    
    Parameters
    ----------
    csv_with_coords(str) - Full path of CSV file with coordinates
    lat(str) - Column name with latitude
    lon(str) - Column name with longitude
    
    Returns
    -------
    A Geopandas GeoDataframe
    """
    df = pd.read_csv(csv_with_coords)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=4326)
    
    return gdf


def clip_raster(input_raster, clip_polygon, out_file):
    ''' 
    Clip input raster using shapefile copied from:
    https://github.com/worldbank/GOST/blob/Scripts/GOSTRocks/rasterMisc.py
    
    Parameters
    ----------
    input_raster(rasterio object) - Raster file to clip
    clip_polygon (Geopandas object) -  Polygon of extents to clip to
    out_file (str) - Full path with extension of output clipped TIF

    Returns
    -------
    Saves TIF file to provided path (out_file)
    '''
    if clip_polygon.crs != input_raster.crs:
        clip_polygon = clip_polygon.to_crs(input_raster.crs)
    out_meta = input_raster.meta.copy()
    def getFeatures(gdf):
        #Function to parse features from GeoDataFrame in such a manner that rasterio wants them
        return [json.loads(gdf.to_json())['features'][0]['geometry']]
    tD = gpd.GeoDataFrame([[1]], geometry=[clip_polygon.unary_union])
    coords = getFeatures(tD)
    out_img, out_transform = mask(input_raster, shapes=coords, crop=True)
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform})
    with rasterio.open(out_file, "w", **out_meta) as dest:
        dest.write(out_img)


def admin_bounds_from_geoboundaries(co_iso='MWI', admin_level=0):
    """
    Retrieves admin boundaries for country from https://www.geoboundaries.org/api.html API.
    
    Parameters
    ----------
    co_iso(str) - Country ISO3 code, for example MWI
    admin_level(int) - Admin level provided as an integer
    
    Returns
    -------
    A Geopandas GeoDataframe
    """

    assert len(co_iso) == 3, 'PLEASE PROVIDE A VALID ISO CODE FOR THE COUNTRY'
    assert admin_level < 10, 'PLEASE PROVIDE A VALID ADMIN LEVEL NUMBER'
    
    # ==========================================
    # LOAD GEOBOUNDARY JSON OBJECT
    # ==========================================
    admin_lev_str = 'ADM{}'.format(admin_level)
    r = requests.get("https://www.geoboundaries.org/gbRequest.html?ISO={}&ADM={}".format(co_iso.upper(), admin_lev_str.upper()))
    dl_path = r.json()[0]['gjDownloadURL']
    geom_json = requests.get(dl_path).json()
    
    # ==========================================
    # CONVERT JSON INTO GEOPANDAS DATAFRAME
    # ========================================== 
    # For now just extract WGS84
    # Also, just default to WGS84
    # TO-DO: find a better way to convert from OGC CRS to EPSG
    try:
        crs_name = geom_json['crs']['properties']['name'].split(":")[-1]
        if crs_name == 'CRS84':
            crs_epsg_code = 4326
        else:
            crs_epsg_code = 4326
            print('Please manually check CRS, now defaulting to WGS-84')
    except:
        print('Please manually check CRS, now defaulting to WGS-84')
        crs_epsg_code = 4326
    
    data = []
    for feature in geom_json['features']:
        properties = feature['properties']
        geometry = shape(feature['geometry'])
        properties['geometry'] = geometry
        data.append(properties)
    
    gdf = gpd.GeoDataFrame(data, crs=crs_epsg_code)
    
    return gdf

def generate_travel_time_raster(friction_raster=None, dest=None, outdir=None, out_raster=None):
    """
    Calculates travel time (in minutes) to destinations (points).
    This function was adapated from code in this notebook: 
    https://github.com/worldbank/INFRA_SAP/blob/master/Notebooks/Implementations/MA_DECAT_B_RasterMarketAccess.ipynb
    Parameters
    ----------
    friction_raster (str) : Full path to friction surface raster layer (country level).
    dest (gpd.GeoDataFrame) : A GeoPandas Datafreme object with target destination points.
    outdir (str) : Directory to hold outputs, some of them temporary.
    out_raster (str) : Preferred file name with extension ".tif" of output travel surface, it will be saved in outdir.
    
    Returns
    -------
    str
        Full path where travel time (in minutes) raster was saved.
    """

    # Ensure output filename has extension .tif
    tif = '.tif'
    try:
        assert out_raster.split(".")[1] == "tif"
    except (AssertionError, IndexError):
        print('PLEASE ADD EXTENSION {} TO OUTPUT RASTER FILE NAME'.format(tif))
    
    # ===========================================
    # CREATE MCP OBJECT
    # ===========================================
    inG = rasterio.open(friction_raster)
    inG_data = inG.read() * 1000
    # Correct no data values
    inG_data[inG_data < 0] = 99999999
    mcp = graph.MCP_Geometric(inG_data[0,:,:])

    
    # ============================================
    # CALCULATE TRAVEL TIME AND SAVE OUTPUT RASTER
    # ============================================
    out_file = Path(outdir).joinpath(out_raster)
    facility_cells = ma.get_mcp_dests(inG, dest)
    costs, traceback = mcp.find_costs(facility_cells)  
    costs[np.isinf(costs)] = 0
    costs[np.isnan(costs)] = 0
    meta = inG.meta.copy()
    meta.update(dtype=costs.dtype)
    with rasterio.open(out_file, 'w', **meta) as out:
        out.write_band(1, costs)
    
    return out_file

def terminal_coordinates(origin_coords, bearing, distance, return_degrees=True):
    """Calculates end coordinates in decimal degrees given initial/origin coordinates in decimal degrees.

    Parameters
    ----------
    origin_coords : tuple
        Initial coordinates in decimal degrees given as a lon,lat tuple.
    bearing : float
        Bearings or azimuths (in degrees) start with 0 degrees toward true north, 90 degrees east, 
        180 degrees south, and 270 degrees west (clockwise rotation)
    distance : int
        Distance in meters.
    return_degrees : Boolean, default = True
        Whether to return new point in degrees or meters
    
    Returns
    ----------
    Tuple
        New coordinates as a (lon,lat) tuple
    """
    # ==========================
    # DO CONVERSIONS
    # ==========================
    # Convert the angle from degrees to radians
    angle_radian = bearing * (math.pi / 180)

    # Convert decimal degrees to meters
    x, y = pod.degrees_to_meters(origin_coords[0], origin_coords[1])

    # get projected CRS EPSG code
    utm_crs_epsg = int(convert_wgs_coordinates_to_utm(origin_coords[0], origin_coords[1]))

    # ==========================
    # GENERATE NEW POINT
    # ==========================
    # Generate the offset by applying trig formulas (law of cosines) using the distance as
    # the hypotenuse solving for the other sides
    x_offset = math.sin(angle_radian) * distance
    y_offset = math.cos(angle_radian) * distance

    # Add the offset to the original coordinates (in meters)
    x_new = x + x_offset
    y_new = y + y_offset

    # Convert back to degrees
    if return_degrees:
        x_degrees, y_degrees = pod.meters_to_degrees(x_new, y_new, utm_crs_epsg=utm_crs_epsg)
        return x_degrees, y_degrees
    else:
        return x_new, y_new

def download_osm_shapefiles(region, country, outdir):
    """Downloads OSM latest shapefile from http://download.geofabrik.de/

    Parameters
    ----------
    region : str
        Continent/region where country is located as used on Geofrabrik website. For example, Africa, Asia, 
        South America (south-america)
    country : str
        Country name in full _description_
    outdir : str
        Directory to save the data.
    
    Returns
    --------
    Saves and unzips downloaded files in directory: outdir + country-latest-free-shp
    """
    start = datetime.now()
    geofabrick_url = 'http://download.geofabrik.de/{}/{}-latest-free.shp.zip'.format(region.lower(), country.lower())
    r = requests.get(geofabrick_url, allow_redirects=True)
    try:
        assert r.status_code == 200
    except AssertionError:
        print('ENSURE THERE IS INTERNET AND/OR REGION AND COUNTRY NAMES ARE CORRECT')
        return 
    

    assert outdir.exists(), 'ENSURE OUTPUT DIRECTORY EXISTS'
    outfile = Path(outdir).joinpath(geofabrick_url.split("/")[-1])
    print('Saving file .............')
    open(outfile, 'wb').write(r.content)
    
    print()
    print('Unzipping file .............')
    extract_outdir = Path(outdir).joinpath(outfile.parts[-1].split(".")[0] + "-shp").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outfile,"r") as zip_ref:
        zip_ref.extractall(extract_outdir)
    
    # Delete zipfile
    file_size = outfile .stat()[6]/1000000
    shutil.rmtree(outfile)

    end = datetime.now()
    time_taken = (end - start).total_seconds/60

    print()
    print('Downloading {} MB took {} minutes'.format(int(file_size)), int(time_taken))

    
    return extract_outdir



    