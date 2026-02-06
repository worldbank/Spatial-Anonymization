"""
Module with miscellaneous functions for geospatial data processing
Created on Mar 6, 2023
@author-1: Dunstan Matekenya
@author-2: Andres

For the sake of convinience and to reduce dependencies, some functions have 
been copied from the following repos/packages:
1. GOSTnets
2. https://github.com/worldbank/INFRA_SAP/blob/master/infrasap/market_access.py
"""
import os
from pathlib import Path
import shutil
import zipfile
from datetime import datetime
import json
import requests
import math
from rasterstats import gen_point_query, gen_zonal_stats
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

import skimage.graph as graph
os.environ["USE_PYGEOS"] = "0"
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import shape
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from . import point_displacement as pod
from math import asin, atan2, cos, degrees, radians, sin


def tif_urls_from_impact_observ(adm0_shp, output_dir, utm_zones, yr=None):
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
        obj_list = [i for i in objs_target_yr if i.split("_")[0][:2]==z]
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
                    out_image, out_transform = rasterio.mask.mask(
                        tile, [shape], crop=True)
                    raster_list.append(output_dir.joinpath(file))
                except Exception as e:
                    print(e)
                    print(
                        f"No overlap found for {tile.files[0]}. Try the next one ..")
                    continue
    # ===================================
    # CLEAN-UP
    # ===================================
    # delete everything except the tifs we
    # need
    for tif in output_dir.iterdir():
        if tif.suffix == ".tif":
            if tif in raster_list:
                continue
            else:
                tif.unlink()

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


def utm_zones_from_geodataframe(gdf, datum='WGS 84', return_zones=False):
    """
    Returns estimated UTM CRS from the geopandas dataframe
    Parameters
    ----------
    gdf: gpd.GeoDataframe: Geopandas dataframe
    return_zones: Boolean : Whether to return UTM zones or EPSG code.

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
    if return_zones:
        return [i[-2:] for i in utm_crs_code_list]
    else:
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


def distance_to_points(dest_pts, target_pt, lon_col='lon', lat_col='lat', 
                       id_col='id', output='nearest'):
    """
    Measure distance from target_pt to all points in dest_pts.
    Parameters
    ----------
    dest_pts - Either a list or a Pandas Dataframe. If a list, it should be a lilike this:  [(lat,lon)] or [(lat,lon,id)]
    target_pt(tuple) - A target point to measure distance from provided as a (lat,lon)
    lon_col(str) - Column with longitude values
    lat_col(str) - Column with latitude values
    id_col (str) - Column with row id if interested in returning nearest point id
    output - nearest-output distance to nearest point; nearest_id-output id of nearest point; dist_list-output distance list.

    Returns A list containing floats
    -------
    """
    if isinstance(dest_pts, pd.DataFrame):
        df = dest_pts.copy(deep=True)
    else:
        if len(dest_pts[0]) == 3:
            df = pd.DataFrame(dest_pts, columns=[lat_col, lon_col, id_col])
        else:
            df = pd.DataFrame(dest_pts, columns=[lat_col, lon_col])
    df['dist'] = df.apply(lambda x: distance_between_points(pt1=(x[lat_col], x[lon_col]),
                                                            pt2=(target_pt[0], target_pt[1])), axis=1)
    if output == 'nearest':
        return df['dist'].min()
    elif output == 'nearest_id':
        try:
            return df.sort_values(by='dist',ascending=True).iloc[0][id_col]
        except:
            print('Make sure you provided ID column')
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
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=4326)

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
        # Function to parse features from GeoDataFrame in such a manner that rasterio wants them
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
    r = requests.get("https://www.geoboundaries.org/gbRequest.html?ISO={}&ADM={}".format(
        co_iso.upper(), admin_lev_str.upper()))
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


def get_mcp_dests(inH, destinations):
    ''' Get indices from inH for use in mcp.find_costs
    INPUT
        inH[rasterio] - object from which to extract geographic coordinates
        destinations[geopandas geodataframe] - point geodataframe of destinations
    RETURN
        [list of indices]
    '''
    cities = list(set([inH.index(x.x, x.y) for x in destinations['geometry']]))
    cities = [x for x in cities if ((x[0] > 0) and (x[1] > 0) and
                                    (x[0] <= inH.shape[0]) and (x[1] <= inH.shape[1]))]
    return(cities)


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
    mcp = graph.MCP_Geometric(inG_data[0, :, :])

    # ============================================
    # CALCULATE TRAVEL TIME AND SAVE OUTPUT RASTER
    # ============================================
    out_file = Path(outdir).joinpath(out_raster)
    facility_cells = get_mcp_dests(inG, dest)
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
    utm_crs_epsg = int(convert_wgs_coordinates_to_utm(
        origin_coords[0], origin_coords[1]))

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
        x_degrees, y_degrees = pod.meters_to_degrees(
            x_new, y_new, utm_crs_epsg=utm_crs_epsg)
        return x_degrees, y_degrees
    else:
        return x_new, y_new


def terminal_coordinates_faster(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    """
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
    lon2 = lon1 + atan2(
        sin(a) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2)
    )
    return degrees(lon2), degrees(lat2)


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
    geofabrick_url = 'http://download.geofabrik.de/{}/{}-latest-free.shp.zip'.format(
        region.lower(), country.lower())
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
    extract_outdir = Path(outdir).joinpath(
        outfile.parts[-1].split(".")[0] + "-shp").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outfile, "r") as zip_ref:
        zip_ref.extractall(extract_outdir)

    # Delete zipfile
    file_size = outfile .stat()[6]/1000000
    shutil.rmtree(outfile)

    end = datetime.now()
    time_taken = (end - start).total_seconds/60

    print()
    print('Downloading {} MB took {} minutes'.format(
        int(file_size)), int(time_taken))

    return extract_outdir


def reproject_tif(in_tif, out_tif, dst_crs='EPSG:4326'):
    """Use rasterio to reproject raster.

    Parameters
    ----------
    in_tif : str
        Full path to input raster (tif)
    out_tif : str
        Full path to output raster (tif)
    dst_crs : str, optional
        Destination CRS in EPSG format
    """
    with rasterio.open(in_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(out_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def tif_from_other(in_tif, out_tif, dst_crs='EPSG:4326'):
    """Use rasterio to save as TIF from other formats

    Parameters
    ----------
    in_tif : str
        Full path to input raster (tif)
    out_tif : str
        Full path to output raster (tif)
    dst_crs : str, optional
        Destination CRS in EPSG format
    """
    with rasterio.open(in_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(out_tif, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def point_query_raster(in_raster, point_shp, statcol_name, class_pct_code=None,
                            id_col=None,stat='mean', buffer=False, buff_dist=None, utm_epsg=None,
                            xcol=None, ycol=None, all_touch=False):
    """Extracts statistics from raster.

    Extraction happens in two possible ways:
    1. Single value at point
    2. Zonal statistics within buffer zone defined by buff_dist

    Parameters
    ----------
    in_raster : str
        Full path to inout raster
    point_shp : gpd.Geopandas.GeoDataFrame
        Geopandas dataframe with point features.
    statcol_name : str
        Column name for the value.
    class_pct_code : int
        Integer value of category for which to calculate class percent. For example, percent with agriculture.
    id_col : str, optional
        Unique identifier(e.g., household ID), by default None
    stat : str, optional
        For buffer zone, summary statistics to use, by default 'mean'
    buffer : bool, optional
        Whether to extract zonal statistics, by default False
    buff_dist : int, optional
        If doing zonal stats, buffer distance in meters, by default 1000
    xcol : _type_, optional
        Column name with x-coordinates (e.g., longitude), by default None
    ycol : _type_, optional
        Column name with y-coordinates (e.g., longitude), by default None
    all_touch: Boolean
            Whether to use all_touched rasterization strategy

    Returns
    -------
    gpd.GeoDataframe
        Same Geodataframe as input (point_shp) with two columns: id_col + statcol_name
    """
   
    # ==============================
    # DO SOME CHECKS
    # ==============================
    rst_obj = rasterio.open(in_raster)
    try:
        assert rst_obj.crs is not None
    except:
        print('RASTER SHOULD HAVE CRS, SETTING CRS TO WGS84')
        rst_obj = rasterio.open(in_raster, mode='r+')
        rst_obj.crs = CRS.from_epsg(4326)
    
    raster_nodata_val = rst_obj.meta['nodata']
    if not utm_epsg:
        utm_epsg_lst = utm_zones_from_geodataframe(point_shp, return_zones=False)
        utm_epsg = int(utm_epsg_lst[0])
    
    # Ensure there are no NaN and duplicates 
    try:
        point_shp.dropna(subset=[xcol, ycol], inplace=True)
        point_shp.drop_duplicates(subset=[xcol, ycol], inplace=True)
    except:
        pass

    # ==============================
    # EXTRACT STATS
    # ==============================
    if buffer or buff_dist:
        point_shp_proj = point_shp.to_crs(utm_epsg)
        point_shp_proj_buff = point_shp_proj.buffer(buff_dist)
        
        if rst_obj.crs != point_shp_proj_buff.crs:
            point_shp_proj_buff = point_shp_proj_buff.to_crs(rst_obj.crs)
        
        if stat == 'majority':
            res = gen_zonal_stats(point_shp_proj_buff, in_raster, stats='majority', categorical=True, nodata=raster_nodata_val,
                                  aall_touched=all_touch)
            point_shp_proj[statcol_name] = [i[stat] for i in res]
            try:
                assert point_shp_proj[statcol_name].isna().sum() == 0, 'THERE SHOULD BE NO NULLS'
            except AssertionError:
                print('Number of nulls: {}'.format(point_shp_proj[statcol_name].isna().sum()))
                pass
            df_out = point_shp_proj[[id_col, statcol_name]]
        elif stat == 'class-pct':
            res = gen_zonal_stats(point_shp_proj_buff, in_raster, all_touched=all_touch, categorical=True, nodata=raster_nodata_val)
            pct_data = []
            for i in res:
                if i:
                    tot_pixels = sum(i.values())
                    sum_target_pixels = np.sum([i.get(class_pct_code[k], 0) for k in range(len(class_pct_code))])
                    class_pct  = sum_target_pixels/tot_pixels*100
                    pct_data.append(class_pct)
                else:
                    pct_data.append(np.NaN)
            point_shp_proj[statcol_name] = pct_data
            try:
                assert point_shp_proj[statcol_name].isna().sum() == 0, 'THERE SHOULD BE NO NULLS'
            except:
                print('Number of nulls: {}'.format(point_shp_proj[statcol_name].isna().sum()))
                pass

            df_out = point_shp_proj[[id_col, statcol_name]]

        else:
            res = gen_zonal_stats(point_shp_proj_buff, in_raster, stats=stat, all_touched=all_touch, nodata=raster_nodata_val)
            point_shp_proj[statcol_name] = [i[stat] for i in res]
            try:
                assert point_shp_proj[statcol_name].isna().sum() == 0, 'THERE SHOULD BE NO NULLS'
            except:
                print('Number of nulls: {}'.format(point_shp_proj[statcol_name].isna().sum()))
                pass

            df_out = point_shp_proj[[id_col, statcol_name]]
        
    else:
        if rst_obj.crs != point_shp.crs:
            point_shp = point_shp.to_crs(rst_obj.crs)
       
        point_shp[statcol_name] = list(gen_point_query(point_shp, in_raster, interpolate='nearest'))
        try:
            assert point_shp[statcol_name].isna().sum() == 0, 'THERE SHOULD BE NO NULLS'
        except:
            print('Number of nulls: {}'.format(point_shp[statcol_name].isna().sum()))
            pass
        df_out = point_shp[[id_col, statcol_name]]
        
    return df_out


def replace_missing_with_nearest(df, target_col, idcol, lon, lat):
    """Replace missing values in column with values in nearest location.

    Given an attribute with values for each location identified by idcol 
    (e.g., clusterId for cluster centroid). Replace any missing values with values
    of attribute in nearest location.

    Parameters
    ----------
    df : pd.Dataframe
        Pandas Dataframe where target_col has missinng values/nulls.
    target_col : str
        Column containing missing values/nulls
    idcol : str
        Column uniquely identifying each row
    lon : str
        Column with longitude values
    lat    : str 
        Column with longitude values
    """
    # ===================================================
    # SEPARATE INTO TWO DATAFRAMES WITH NULLS AND WITHOUT
    # ===================================================
    df_null = df[df[target_col].isnull()][[idcol, lon, lat]]
    df_other = df[~df[target_col].isnull()][[target_col, idcol, lon, lat]]

    # =====================================
    # FIND NEAREST NEIGHBORS IN NULLS
    # =====================================
    df_null['nearest'] = df_null.apply(lambda x: distance_to_points(dest_pts=df_other, 
                        target_pt=(x[lat],x[lon]), lon_col=lon, lat_col=lat, 
                        id_col=idcol, output='nearest_id'), axis=1)
    df_null[target_col] = df_null.nearest.apply(lambda x: 
                                            df_other.query('{}=="{}"'.format(idcol, x))[target_col].values[0])
    
    # merge back the dataframes 
    keep_cols = [target_col, idcol, lon, lat]
    df = df_other[keep_cols].append(df_null[keep_cols])

    return df
    
