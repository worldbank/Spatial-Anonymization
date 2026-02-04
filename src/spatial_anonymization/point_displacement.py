"""
Created on Mar 6, 2023
@author-1: Dunstan Matekenya
@author-2: Andres
"""
import os
os.environ["USE_PYGEOS"] = "1"
from . import utils
from pathlib import Path
from datetime import datetime

import random
import math
import numpy as np

import pandas as pd
import geopandas as gpd

from pyproj import CRS, Proj, Transformer, Geod
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry import Point, LineString
from rasterstats import gen_zonal_stats
import rasterio
from collections import Counter


class SimpleDisplacer(object):
    """
    A  set of functions to displace household survey GPS coordinates
    in random direction and distance based on the DHS methodology.
    """
    PARAM_DESCRIPTIONS = {
        'urb_max_dist': 'Maximum distance(m) for anonymizing region for coordinates in urban areas',
        'rur_max_dist': 'Maximum distance(m) for anonymizing region for coordinates in rural areas',
        'rur_max_dist_subset': 'Maximum distance(m) for anonymizing region for a subset of coordinates in rural areas',
        'urb_min_dist': 'Minimum distance(m) from original coordinates in urban areas',
        'rur_min_dist': 'Minimum distance(m) from original coordinates in urban areas',
        'maximum_tries': 'Number of times to keep trying when geenrated displaced coordinates do not meet requirements',
        'apply_min_dist': 'Whether to apply minimum distance or not',
        'rur_sub_prop': 'Proportion of rural cooridnates to apply large anonymizing zone',
        'admin_reg': 'Admin region for constraining displaced coordinates',
        'min_pop': 'Minimum number of people withing anonymizing region to avoid re-identification',
        'check_waterbodies': 'Check to ensure generated coordinates dont fall on large water bodies',
        'pop_check': 'Whether generated coordinates meet the minimum population threshold',
        'max_runs': 'Maximum number of re-runs when some coordinates fail to get generated',
        'utm_epsg_code': 'While some counties have multiple UTM zones, use this as default'
        }

    def __init__(self, params, inputs, output_dir, lon_col, lat_col,
                 urb_rur_col=None, rural_code=None, geodetic_crs='EPSG:4326', 
                 unique_id=None):
        """
        Initializes object attributes
        Parameters
        ----------
        params(dict object)- provides displacement parameters such as maximum_distance, min_distance
        inputs(dict) - Dict object containing the following inputs: 
        {'pop_tif': (str/Path object) Full path to raster file with population counts, 
        'shp_waterbodies': (str/Path object) Full path to shapefile of major water bodies,
        'shp_adm_bounds': (str/Path object) Full path to shapefile of admin boundaries to use as constraining regions,
        'input_coords': (str/Path object/Dataframe) Either pd.Dataframe or full path of coordinates to be anonymized
        output_dir - directory where to save results
        lon_col (str) - Column in the input file which has longitude (decimal degrees)
        lat_col (str) - Column in the input file which has latitude (decimal degrees)
        urb_rur_col (str) - Column in the input file with urban/rural classification
        urban_code - which code represent urban
        geodetic_crs - Geographic CRS as EPSG code for which the latitude and longitude were collected.
        The default is WGS84, code 4326.
        unique_id - Unique identifier for each row (e.g., household-ID)
        """
        # ensure that paths to files and files are are valid
        input_coords = inputs['input_coords']
        if isinstance(input_coords, str) or (input_coords).exists():
            self.data_file = Path(input_coords)
        elif isinstance(input_coords, pd.core.frame.DataFrame):
            self.data_file = input_coords
        else:
            print('PLEASE CHECK THE INPUT COORDS FILE/OBJECT')

        shp_adm_bounds = inputs['shp_adm_bounds']
        if Path(shp_adm_bounds).exists():
            self.adm_file = Path(shp_adm_bounds)
        else:
            print('PLEASE ENSURE THIS PATH: {} IS VALID'.format(shp_adm_bounds))
        
        if Path(output_dir).exists():
            self.output_dir = Path(output_dir)
        else:
            print('CREATING OUTPUT DIRECTORY: {}'.format(output_dir))
            self.output_dir = Path(output_dir).mkdir()
        # set column names
        self.lon = lon_col
        self.lat = lat_col
        self.x = 'x'  # column name for original coordinates in meters
        self.y = 'y'  # column name for original coordinates in meters
        self.disp_x = 'disp_x'  # column name for displaced coordinates in meters
        self.disp_y = 'disp_y'  # column name for displaced coordinates in meters
        self.disp_lon = 'disp_lon'  # column name for displaced coordinates in degrees
        self.disp_lat = 'disp_lat'  # column name for displaced coordinates in degrees

        self.urb_rur_col = urb_rur_col
        self.rural_val = rural_code
        self.id = unique_id

        self.geodetic_epsg_code = geodetic_crs
        self.urb_max_dist = params['urb_max_dist']
        self.rur_max_dist = params['rur_max_dist']
        self.rur_sub_prop = params['rur_sub_prop']
        self.rur_max_dist_sub = params['rur_max_dist_subset']
        self.urb_min_dist = params['urb_min_dist']
        self.rur_min_dist = params['rur_min_dist']
        self.maximum_tries = params['maximum_tries']
        self.apply_min_dist = params['apply_min_dist']
        self.max_runs = params['max_runs']
        self.default_utm_epsg_code = params['utm_epsg_code']

        # Lets hold the coordinates and the admin regions
        # in these variables in case we need to use them
        self.coords_data = None  # Running dataframe as we do processing
        # Original dataframe, except for dropped null values
        # Keep this for sanity checks
        self.original_coords = None
        self.adm_data = None
        self.dropped_coords_data = None
        self.coords_data_disp = None

        # set datum name for the lat/lon coordinates
        if 'datum' in params:
            self.datum_name = params['datum_name']
        else:
            self.datum_name = 'WGS 84'
        # column name for UTM CRS associated with each Lat/Lon
        self.proj_crs_col = 'utm_epsg_code'
        self.pop_tif = inputs.get('pop_tif', None)
        self.waterbodies_shp = inputs.get('shp_waterbodies', None)
        self.min_pop = params.get('min_pop', None)
        self.params = params
        # temporary holding place as displaced coordinates
        # are being genrated
        self.temp_disp_coords = None

    def print_params(self, out_fname):
        """Print and save parameters
        _extended_summary_
        """
        num = 1
        params = self.params
        if self.pop_tif:
            params['pop_check'] = True
        else:
            params['pop_check'] = False
        
        if self.waterbodies_shp:
            params['check_waterbodies'] = True
        else:
            params['check_waterbodies'] = False
        
        print('='*50)
        print(' Running displacement with the following params')
        print('='*50)
        for k, v in params.items():
            desc = self.PARAM_DESCRIPTIONS[k]
            print('{}.{}-{}-{}'.format(num, k, v, desc))
            num += 1
        df = pd.DataFrame([params, self.PARAM_DESCRIPTIONS]).T.reset_index().rename(
                            columns={0: 'Value', 'index':'Param/input', 1:'Description'})
        df.to_csv(out_fname, index=False)
        short_output_fpath = "/".join(out_fname.parts[6:])
        print()
        print('Params file saved to file: \n{}'.format(short_output_fpath))
        print('-'*100)
        
    def load_coordinates_data(self):
        """
        Load coordinates into a dataframe and do some preprocessing
        Returns
        -------

        """
        # =========================================
        # LOAD DATA AND DO PRELIMINARY CLEAN-UP
        # =========================================
        if isinstance(self.data_file, pd.core.frame.DataFrame):
            df_hh_coords = self.data_file
        else:
            df_hh_coords = None
            if self.data_file.suffix == ".csv":
                df_hh_coords = pd.read_csv(self.data_file)
            elif self.data_file.suffix == ".dta":
                df_hh_coords = pd.read_stata(self.data_file, convert_categoricals=False)
            else:
                print('PLEASE CONVERT THE COORDINATES FILE TO CSV OR STATA FILE')

        # set unique-id as string to avoid number issues
        df_hh_coords[self.id] = df_hh_coords[self.id].astype(str)

        # drop any null values in lat/lon columns
        df_hh_coords[[self.lon, self.lat]] = df_hh_coords[[self.lon, self.lat]].astype(float)
        num_nulls = df_hh_coords[[self.lon, self.lat]].isna().sum().max()
        if num_nulls > 0:
            df_hh_coords2 = df_hh_coords.dropna(subset=[self.lon, self.lat])
            df_na = df_hh_coords.loc[df_hh_coords.index.difference(df_hh_coords2.index)]
            assert len(df_na) == num_nulls
            self.dropped_coords_data = df_na
            # Save dropped observations for auditing
            output_csv_filename = "{}_dropped.csv".format(self.data_file.parts[-1].split(".")[0])
            output_csv_fpath = self.output_dir.joinpath(output_csv_filename)
            df_na.to_csv(output_csv_fpath, index=False)
            print()
            print('{:,} null lat/lon rows dropped out of {:,}'.format(num_nulls, len(df_hh_coords)))
            print()
            print('Dropped observations saved to file: \n {}'.format(output_csv_fpath))

            self.coords_data = df_hh_coords2
            self.original_coords = df_hh_coords2
        else:
            self.coords_data = df_hh_coords
            self.original_coords = df_hh_coords

        # ====================================================
        # COLLAPSE TO EA LEVEL/ANOTHER LEVEL IF NEED TO
        # ====================================================
        df = self.coords_data
        if df[self.id].nunique() == len(df):
            pass
        elif df[self.id].value_counts().mean() > 5:
            print()
            print('Grouping to {} level and calculating average coordinates'.format(self.id))
            print()
            df_ea = df.groupby(['ea_id']).agg(ea_ctr_lat=(self.lat, 'mean'),
                                              ea_ctr_lon=(self.lon, 'mean'),
                                              urb_rur=(self.urb_rur_col,
                                                       pd.Series.mode)).reset_index()
            # change column names
            self.lon = 'ea_ctr_lon'
            self.lat = 'ea_ctr_lat'
            self.urb_rur_col = 'urb_rur'

            # update data
            self.coords_data = df_ea
        else:
            pass

        # =========================================
        # FIX URBAN/RURAL CODE
        # =========================================
        df_hh_coords = self.coords_data
        if isinstance(self.rural_val, str):
            df_hh_coords[self.urb_rur_col] = df_hh_coords[self.urb_rur_col].apply(lambda x: x.lower())
            df_hh_coords[self.urb_rur_col] = df_hh_coords[self.urb_rur_col].map({'rural': 1, 'urban': 2})
            self.rural_val = 1

        # ====================================================
        # INFER PROJECTED CRS (EPSG CODE) FROM LAT/LON VALUES
        # AND ADD COORDINATES IN METERS
        # ====================================================
        # infer UTM CRS and get updated coordinates data
        df_hh_coords = self.infer_projected_crs()

        # add coordinates in meters
        df_hh_coords['tmp'] = df_hh_coords.apply(lambda x:
                                                 degrees_to_meters(x[self.lon], x[self.lat],
                                                                   utm_crs_epsg=x[self.proj_crs_col]),
                                                 axis=1)
        df_hh_coords[self.x] = df_hh_coords.tmp.apply(lambda x: x[0])
        df_hh_coords[self.y] = df_hh_coords.tmp.apply(lambda x: x[1])
        df_hh_coords.drop(columns=['tmp'], inplace=True)

        # update coordinates data
        self.coords_data = df_hh_coords

    def add_dynamic_distance(self):
        """
        Adds a maximum and minimum distance for displacement based
        on the following:
        1. Whether household is in urban or rural
        2. Whether household is randomly picked to be in special rural pool for
        larger distance displacement
        Returns
        -------

        """
        # ============================================
        # RETRIEVE CURRENT DATAFRAME AND ASSIGN GENERIC
        # URBAN/RURAL MAXIMUM DISTANCES
        # ============================================
        df_hh_coords = self.coords_data
        df_hh_coords['max_dist'] = df_hh_coords[self.urb_rur_col].apply(
            lambda x: self.rur_max_dist if x == self.rural_val else self.urb_max_dist)
        if self.apply_min_dist:
            df_hh_coords['min_dist'] = df_hh_coords[self.urb_rur_col].apply(
                lambda x: self.rur_min_dist if x == self.rural_val else self.urb_min_dist)
        else:
            df_hh_coords['min_dist'] = 0

        # ================================================
        # SELECT P% OF RURAL COORDINATES AND ASSIGN LARGER
        # RURAL MAXIMUM DISTANCE
        # ================================================
        # Check if dataframe has unique ID
        try:
            assert df_hh_coords[self.id].nunique() == len(df_hh_coords)
            id_col = self.id
        except AssertionError:
            id_col = "id"
            df_hh_coords[id_col] = ['tmp_{}'.format(i) for i in range(1, len(df_hh_coords) + 1)]

        # get rural hh/coordinates
        df_rur = df_hh_coords.query('{} == {}'.format(self.urb_rur_col, self.rural_val))

        # set required number of rural households to select using the proportion param
        rural_sub_size = int(len(df_rur) * self.rur_sub_prop)

        # Select rural_sub_size random hhs
        selected_ids = np.random.choice(df_rur[id_col].values, size=rural_sub_size, replace=False)

        # Assign higher maximum distance to rural households/coordinates selected above
        df_hh_coords['max_dist'] = df_hh_coords.apply(
            lambda x: self.rur_max_dist_sub if x[id_col] in selected_ids else x['max_dist'], axis=1)

        # update coords data
        self.coords_data = df_hh_coords

    def prepare_data(self):
        """
        Preprocess coordinates file and admin regions files in preparation
        for displacement.
        Returns  A Geopandas dataframe with coordinates converted to planar (meters) and each coordinate associateed
        with admin region.
        -------

        """
        # =======================================
        # GET COORDINATES DATA
        # =======================================
        self.load_coordinates_data()

        # =======================================
        # ADD DISTANCE COLUMN BASED ON URBAN/RURAL
        # AND INCLUSION IN SPECIAL SUBSET
        # =======================================
        self.add_dynamic_distance()

        # =======================================
        # LOAD ADMIN SHAPEFILE AND RUN CHECKS
        # =======================================
        # Get admin bounds
        assert self.adm_file.suffix == ".shp", 'PLEASE USE SHAPEFILE FOR ADMIN REGIONS'
        gdf_adm = gpd.read_file(self.adm_file)
        self.adm_data = gdf_adm

        # Ensure that admin geodataframe has geodetic CRS
        # same as the one used for lon/lat
        try:
            assert gdf_adm.crs.to_epsg() == self.geodetic_epsg_code
        except:
            if gdf_adm.crs is None:
                print('Please provide CRS for the admin boundaries')
            else:
                gdf_adm = gdf_adm.to_crs(self.geodetic_epsg_code)

        # =======================================
        # JOIN THE HH COORDS WITH ADMIN SHAPEFILE
        # =======================================
        # get updated coords data
        df_hh_coords = self.coords_data

        # create coordinates geodataframe based on lat/lon
        gdf_hh_coords = gpd.GeoDataFrame(data=df_hh_coords,
                                         geometry=gpd.points_from_xy(df_hh_coords[self.lon],
                                                                     df_hh_coords[self.lat]),
                                         crs=self.geodetic_epsg_code)

        # join the hh coordinates with admin bounds
        gdf_adm_hh_coords = gpd.sjoin(right_df=gdf_hh_coords, left_df=gdf_adm, op='contains')

        # Check that the two matched perfectly
        try:
            assert len(gdf_adm_hh_coords) == len(df_hh_coords)
        except AssertionError:
            outstanding_id = set(list(df_hh_coords[self.id].values)) - set(list(gdf_adm_hh_coords[self.id].values))
            print('SOME OF THE COORDINATES TO BE DISPLACED DID NOT MATCH WITH ADMIN REGIONS: {}'.format(list(outstanding_id)))

        return gdf_adm_hh_coords

    def add_displaced_coordinates(self, row, max_tries):
        """
        Helper function to apply across each row
        in the Geopandas dataframe from prepare_data(self) to generate displaced coordinates
        Parameters
        ----------
        row - A row in a Geopandas dataframe
        max_tries - Number of times to try generating perturbed coordinates which are within a given admin region.
        After this, we stop.

        Returns x_meters, y_meters, x_degrees, y_degrees. If num_tries exceeded, returns (-1, -1)
        -------

        """
        # Create Rasterio Object
        if self.pop_tif: 
            rd = rasterio.open(self.pop_tif)
            transform = rd.meta['transform']
            array = rd.read(1)
            nodata_val = rd.meta['nodata']
            array_data = np.where(array!=nodata_val, array, 0)
        
        # coordinates of original point
        orig_x, orig_y = row[self.lon], row[self.lat]
        
        # polygon of admin region where the orig point falls
        polygon = row['geometry']
        
        max_dist = row['max_dist']
        min_dist = row['min_dist']
        pop_threshold = self.min_pop
        num_tries = 0
        failed_attempts = []
        while True:
            new_lon, new_lat = displace_point_faster(x=orig_x, y=orig_y, max_distance=max_dist, min_distance=min_dist)
            new_pt = Point(new_lon, new_lat)
            gdf_new_pt = gpd.GeoDataFrame([{'geometry': new_pt, 'id': 'p1'}], crs=self.geodetic_epsg_code)
            
            # Admin zone check: checks whether newly created point falls within constraining
            # admin zone
            if polygon.contains(new_pt):
                within_adm_zone = True
            else:
                within_adm_zone = False
                failed_attempts.append('adm')
                continue

             # Waterbody check: whether this newly geenrated point doesnt fall on a large waterbody
            if self.waterbodies_shp:
                gdf_water = gpd.read_file(self.waterbodies_shp)
                possible_matches = gpd.sjoin(gdf_new_pt, gdf_water, op='within')

                if len(possible_matches) == 0:
                    water_check = True
                else:
                    water_check = False
                    failed_attempts.append('water')
                    continue
            else:
                water_check = True
            
            # Population check: checks whether population withing n-km (where n = max_dist)
            # is larger than pop_threshold (e.g., 500 people)
            if self.pop_tif:
                # create buffer 
                point_shp_proj = gdf_new_pt.to_crs(self.default_utm_epsg_code)
                point_shp_proj_buff = point_shp_proj.buffer(max_dist)
                point_shp_proj_buff_wg84 = point_shp_proj_buff.to_crs(self.geodetic_epsg_code)

                # Get intersection between buffer and admin region
                geom_intersection = point_shp_proj_buff_wg84.intersection(polygon)
                
                pop = list(gen_zonal_stats(vectors=geom_intersection, raster=array_data, 
                                               affine=transform, stats=['sum']))[0]['sum']
                assert pop > 0, 'POPULATION ESTIMATES SHOULD BE ABOVE ZERO'
                print(pop)
                if pop > pop_threshold:
                    pop_check = True
                else:
                    pop_check = False
                    failed_attempts.append('pop')
            else:
                pop_check = True
                pop = np.NaN
            
            if (within_adm_zone and pop_check and water_check):
                return new_lon, new_lat, pop
            else:
                pass
            
            num_tries += 1
            if num_tries >= max_tries:
                if max_tries > 25:
                    failed = Counter(failed_attempts)
                    print('Failed to generate displacements for : {}-{},{}'.format(row[self.id], orig_x, orig_y))
                    print(failed)
                return -1, -1, -1

    def infer_projected_crs(self):
        """
        Automatically determine projected CRS in the form of EPSG code
        from the coordinates in the dataframe
        Returns Updates the coordinates data in the class attribute
        -------

        """
        df_hh_coords = self.coords_data

        min_lat = df_hh_coords[self.lat].min()
        max_lat = df_hh_coords[self.lat].max()
        min_lon = df_hh_coords[self.lon].min()
        max_lon = df_hh_coords[self.lon].max()

        utm_crs_list = query_utm_crs_info(
            datum_name=self.datum_name,
            area_of_interest=AreaOfInterest(
                west_lon_degree=min_lon,
                south_lat_degree=min_lat,
                east_lon_degree=max_lon,
                north_lat_degree=max_lat,
            ),
        )
        utm_crs_code_list = [i[1] for i in utm_crs_list]
        if len(utm_crs_code_list) > 1:
            # EPSG code to each row
            df_hh_coords[self.proj_crs_col] = df_hh_coords.apply(lambda x:
                                                                 utils.convert_wgs_coordinates_to_utm(
                                                                     x[self.lon], x[self.lat]), axis=1)
            proj_crs_list = list(df_hh_coords[self.proj_crs_col].unique())
            # Ensure that the two methods found the same codes
            assert sorted(utm_crs_code_list) == sorted(proj_crs_list), 'Check the UTM CRS code'
            print('Found the following projected CRS: {}'.format(proj_crs_list))
        else:
            df_hh_coords[self.proj_crs_col] = utm_crs_list[0][1]

        return df_hh_coords

    def run_displacement_process(self, df_input, max_runs=5, max_tries=25):
        """Helper to run displacemet process until all coordinates have been generated or give-up after max_runs

        Parameters
        ----------
        max_runs - When some coordinates come with no displacements after exhausting maximum_num tries, 
        this param determines how many times to try again for those coordinates
        max_tries - Maximum number of tries.
        """
        print()
        print('Run-{} ......'.format(max_runs))
        df_input['tmp_new_xy'] = df_input.apply(lambda x: self.add_displaced_coordinates(x, max_tries=max_tries), axis=1)
        df_input[self.disp_lon] = df_input.tmp_new_xy.apply(lambda x: x[0])
        df_input[self.disp_lat] = df_input.tmp_new_xy.apply(lambda x: x[1])
        df_input['pop'] = df_input.tmp_new_xy.apply(lambda x: x[2])
        df_failed = df_input[df_input[self.disp_lon] == -1]
        df_success = df_input[df_input[self.disp_lon] != -1]
        # initialize temporary dataframe to hold completed displacements
        if max_runs == self.max_runs:
            self.temp_disp_coords = pd.DataFrame(columns=df_input.columns)
        if len(df_failed) == 0:
            self.temp_disp_coords = self.temp_disp_coords.append(df_success)
            return df_failed
        elif max_runs == 0:
            print()
            print('Exhausted number of tries, existing without finishing all coordinates')
            print('{} points couldnt be displaced'.format(len(df_failed)))
            self.temp_disp_coords = self.temp_disp_coords.append(df_success)
            print(df_failed[[self.id, self.urb_rur_col, self.lon, self.lat]])
            return df_failed 
        else:
            print('{} coords failed to be generated, re-running'.format(len(df_failed)))
            self.temp_disp_coords = self.temp_disp_coords.append(df_success)
            max_tries += 5
            return self.run_displacement_process(df_input=df_failed, max_runs=max_runs-1, max_tries=max_tries)

    def run_displacer(self, output_file):
        """
        Calls all the required functions to generate the displacements.

        Parameters
        ----------
        output_file (str) - Full path for output file.
        If not given, will save file to given output directory with same name as input_coords file
        and suffix "disp".

        Returns - Saves CSV file with four new columns as follows:
        'disp_x', 'disp_y'- new displaced coordinates in meters.
        'disp_lon', 'disp_lat' - new displaced coordinates  in decimal degrees
        -------

        """
        # ========================
        # PREPARE THE DATA
        # ========================
        output_params = self.output_dir.joinpath('disp-params-{}.csv'.format(output_file.parts[-1][:-4]))
        self.print_params(out_fname=output_params)
        print()
        print('Preparing data for displacement...')

        start = datetime.now()
        df = self.prepare_data()
        
        # =========================
        # ADD DISPLACED COORDINATES
        # ==========================
        print('Adding displacements to {:,} points ....'.format(len(df)))
        df_failed = self.run_displacement_process(df_input=df, max_runs=self.max_runs, max_tries=self.maximum_tries)
        
        # Get coordinates which managed to generate displacements
        df_success = self.temp_disp_coords
        df_success.drop(columns=['geometry', 'tmp_new_xy'], inplace=True)
        
        # ======================================================
        # SAVE THE CSV FILE-COORDS WITH NO DISPLACED COORDS
        # ======================================================
        print('Saving outputs...')
        out_failed = self.output_dir.joinpath('{}-failed.csv'.format(output_file.parts[-1][:-4]))
        if len(df_failed) > 0:
            df_failed = df_failed[[self.id, self.urb_rur_col, self.lon, self.lat]]
            df_failed.to_csv(out_failed, index=False)
            short_fpath = "/".join(out_failed.parts[6:])
            print('Coordinates with no displacement saved to file: \n {}'.format(short_fpath))
        else:
            print('Displacements generated for all coordinates in input')
            pass

        # ======================================================
        # SAVE THE CSV FILE-COORDS WITH SUCCESSFUL DISPLACEMENT
        # ======================================================
        if Path(output_file).parents[0].exists():
            output_csv_fpath = output_file
        else:
            output_csv_filename = "{}_disp.csv".format(self.data_file.parts[-1].split(".")[0])
            output_csv_fpath = self.output_dir.joinpath(output_csv_filename)

        # Remove unnecessary columns before saving
        keep_cols = [self.id, self.urb_rur_col, self.lon, self.lat, self.disp_lon, 
                     self.disp_lat, 'max_dist', 'pop']
        df = df_success[keep_cols]

        df.to_csv(output_csv_fpath, index=False)
        short_fpath = "/".join(output_csv_fpath.parts[6:])
        print('Output CSV file with anonymized coordinates saved to file: \n {}'.format(short_fpath))

    
        end = datetime.now()
        time_taken = (end - start).total_seconds()/60
        print()
        print('Processing {:,} coordinates with {} maximum tries per coordinate took {} minutes'.format(
            len(df), self.maximum_tries, int(time_taken)))
        print("-" * 100)
        # update coordinates data with this dataframe
        self.coords_data_disp = df

        # =========================
        # RUN SUMMARY STATS
        # =========================
        print()
        print('='*60)
        print(' Generating summary statistics to check the displacement')
        print('='*60)
        self.generate_displacement_stats(stats=['mean', 'median', 'max', 'min'], output_fname=output_file)

        return df

    def generate_displacement_stats(self, stats=None, output_fname=None):
        """
        Run diagnostic summary statistics on distance between original
        and displaced point
        Parameters
        ----------
        stats - List of summary statistics to run. For example, mean, median, min, maximamum

        Returns Saves summary statistics as a CSV file
        -------

        """
        # ===================================
        # CALCULATE DISTANCES
        # ===================================
        dist_col = 'dist2origin'
        df_res = self.coords_data_disp
        df_res[dist_col] = df_res.apply(lambda x:
                                          distance_latlon_points((x[self.lon], x[self.lat]),
                                                                 (x[self.disp_lon], x[self.disp_lat])), axis=1)

        # update results based off distance calculations
        self.coords_data_disp = df_res

        # ===================================
        # GET SUMMARY STATISTICS
        # ===================================
        df_rur = df_res.query('{} == {}'.format(self.urb_rur_col, self.rural_val))
        print()
        print('Urban/rural distribution')
        print('-'*25)
        print(df_res[self.urb_rur_col].value_counts())
        urb_code = list(set(df_res[self.urb_rur_col].unique()) - {self.rural_val})[0]
        df_urb = df_res.query('{} == {}'.format(self.urb_rur_col, urb_code))

        
        res = []
        if 'mean' in stats:
            urb_mean = df_urb[dist_col].mean()
            rur_mean = df_rur[dist_col].mean()
            res.append({'stat': 'mean', 'area': 'urban', 'value': urb_mean})
            res.append({'stat': 'mean', 'area': 'rural', 'value': rur_mean})
        if 'median' in stats:
            urb_med = df_urb[dist_col].median()
            rur_med = df_rur[dist_col].median()
            res.append({'stat': 'median', 'area': 'urban', 'value': urb_med})
            res.append({'stat': 'median', 'area': 'rural', 'value': rur_med})

        if 'min' in stats or 'minimum' in stats:
            urb_min = df_urb[dist_col].min()
            rur_min = df_rur[dist_col].min()
            res.append({'stat': 'minimum', 'area': 'urban', 'value': urb_min})
            res.append({'stat': 'minimum', 'area': 'rural', 'value': rur_min})

        if 'max' in stats or 'maximum' in stats:
            urb_max = df_urb[dist_col].max()
            rur_max = df_rur[dist_col].max()
            res.append({'stat': 'maximum', 'area': 'urban', 'value': urb_max})
            res.append({'stat': 'maximum', 'area': 'rural', 'value': rur_max})

        # ===================================
        # PRINT AND SAVE SUMMARY STATISTICS
        # ===================================
        df_stats = pd.DataFrame(res)
        print()
        print('Stats for distance between original & displaced coords-Urban')
        print("-" * 62)
        print(df_stats.query('area == "urban"'))
        
        print()
        print('Stats for distance between original & displaced coords-Rural')
        print("-" * 62)
        print(df_stats.query('area == "rural"'))

        output_csv_filename = "{}-disp-stats.csv".format(output_fname.parts[-1][:-4])
        output_csv_fpath = self.output_dir.joinpath(output_csv_filename)

        print()
        print('Summary displacement statistics saved to: \n {}'.format("/".join(output_csv_fpath.parts[6:])))
        print('-'*100)
        df_stats.to_csv(output_csv_fpath, index=False)

def run_sanity_check(self):
        df = self.original_coords

        df[self.id] = df[self.id].astype(str)
        df_proc_coords = self.coords_data
        df_proc_coords[self.id] = df_proc_coords[self.id].astype(str)

        df_tmp = df.merge(df_proc_coords, on=self.id, how='inner', indicator=True)

        assert len(df_tmp) == len(df_tmp), 'SHOULD HAVE SAME LENGTH'


def displace_point_with_angle(x, y, distance, angle_degrees):
    """
    calculates new point up to a given distance away
    from original point given an angle.
    Parameters
    ----------
    x - Coordinate (Eastings) of point given in meters
    y - Coordinnate (Northings) of point given in meters
    mdistance - Maximum allowed distance (meters) to displace point

    Returns coordinates of the new displaced point
    -------

    """
    # Convert the random angle from degrees to radians
    angle_radian = angle_degrees * (math.pi / 180)

    # Generate the offset by applying trig formulas (law of cosines) using the distance as
    # the hypotenuse solving for the other sides
    x_offset = math.sin(angle_radian) * distance
    y_offset = math.cos(angle_radian) * distance

    # Add the offset to the original coordinates (in meters)
    x_new = x + x_offset
    y_new = y + y_offset

    return x_new, y_new


def displace_point(x, y, max_distance, min_distance=0):
    """
    calculates new point up to a given distance away
    from original point.  All values should be provided
    in meters
    Parameters
    ----------
    x - Coordinate (Eastings) of point given in meters
    y - Coordinnate (Northings) of point given in meters
    min_distance - minimum allowed distance (meters) to displace point
    max_distance - Maximum allowed distance (meters) to displace point

    Returns coordinates of the new displaced point
    -------

    """
    # Generate a random angle between 0 and 360
    angle_degree = random.randint(0, 360)

    # Convert the random angle from degrees to radians
    angle_radian = angle_degree * (math.pi / 180)

    # Generate a random distance by multiplying the max distance by a random number between 0 and 1
    while True:
        distance = random.random() * max_distance
        if distance > min_distance:
            break

    # Generate the offset by applying trig formulas (law of cosines) using the distance as
    # the hypotenuse solving for the other sides
    x_offset = math.sin(angle_radian) * distance
    y_offset = math.cos(angle_radian) * distance

    # Add the offset to the original coordinates (in meters)
    x_new = x + x_offset
    y_new = y + y_offset

    return x_new, y_new


def displace_point_faster(x, y, max_distance, min_distance=0):
    """
    calculates new point up to a given distance away
    from original point.  All values should be provided
    in meters
    Parameters
    ----------
    x - Coordinate (Longitude)) of point given in meters
    y - Coordinnate (Latitude) of point given in meters
    min_distance - minimum allowed distance (meters) to displace point
    max_distance - Maximum allowed distance (meters) to displace point

    Returns coordinates of the new displaced point
    -------

    """
    # Generate a random angle between 0 and 360
    angle_degree = random.randint(0, 360)

    # Generate a random distance by multiplying the max distance by a random number between 0 and 1
    while True:
        distance = random.random() * max_distance
        if distance > min_distance:
            break

    disp_lon, disp_lat = utils.terminal_coordinates_faster(lat1=y, lon1=x, d=distance/1000, bearing=angle_degree, R=6371)

    return disp_lon, disp_lat


def distance_planar_points(pt1, pt2):
    """
        Calculates planar distance between two points
        given in meters
        Parameters
        ----------
        pt1 - (x,y) coordinates in meters for first point
        pt2 - (x,y) coordinates in meters for second point

        Returns distance in kilometers
        -------

        """
    sum_sq = np.sum(np.square(np.array(pt1) - np.array(pt2)))
    return np.sqrt(sum_sq) / 1000


def distance_latlon_points(pt1, pt2, ellipsoid="WGS84"):
    """
        Calculates distance between two points given by longitude and latitude
        using pyproj Geod
        Parameters
        ----------
        pt1 - (x,y) coordinates in decimal degrees
        pt2 - (x,y) coordinates in decimal degrees
        ellipsoid - whether its WGS84 or other datum

        Returns distance in kilometers
        -------

        """
    line_string = LineString([Point(pt1[0], pt1[1]), Point(pt2[0], pt2[1])])
    wgs84_geod = Geod(ellps=ellipsoid)
    total_length = wgs84_geod.geometry_length(line_string)
    return total_length / 1000


def degrees_to_meters(x_degrees, y_degrees, utm_crs_epsg=None):
    """
    Convert coordinates in decimal degrees to meters
    Parameters
    ----------
    x_degrees - x coordinate in decimal degrees
    y_degrees - y coordinate in decimal degrees
    utm_crs_epsg - target projected CRS as EPSG code

    Returns x, y coordinates in meters
    For example: (714931.710372255, 8262655.189366714) = degrees_to_meters(35.005599, -15.705196, projected_crs=32736)
    -------
    """
    # Get projected CRS EPSG code if not given
    if not utm_crs_epsg:
         utm_crs_epsg = utils.convert_wgs_coordinates_to_utm(lon=x_degrees, lat=y_degrees)

    # Ensure lon and lat values are valid
    assert (360 > x_degrees > -360), 'longitude outside of wrapping bounds'
    assert (180 > y_degrees > -180), 'latitude outside of wrapping bounds'

    projected_crs = CRS.from_epsg(utm_crs_epsg)
    proj_from_crs = Proj(projected_crs)
    x, y = proj_from_crs(x_degrees, y_degrees)

    return x, y


def meters_to_degrees(x, y, utm_crs_epsg, geodetic_crs_epsg=4326):
    """
    Convert coordinates in meters to decimal degrees
    Parameters
    ----------
    x_degrees - x coordinate in decimal degrees
    y_degrees - y coordinate in decimal degrees
    utm_crs_epsg - projected crs info as EPSG code to convert from
    geodetic_crs_epsg - target geodetic CRS as EPSG code (e.g., 4326 for WGS 84)

    Returns x, y coordinates in degrees
    (35.005599000000004, -15.705196) = meters_to_degrees(714931.710372255,8262655.189366714,
    projected_crs=32736, geodetic_crs=4326)
    -------
    """
    transformer = Transformer.from_crs(utm_crs_epsg, geodetic_crs_epsg)
    lat, lon = transformer.transform(x, y)

    # Ensure lon and lat values are valid
    try:
        assert (360 >= lon >= -360)
        assert (180 >= lat >= -180)
    except AssertionError:
        print('LONGITUDE AND/OR LATITUDE OUTSIDE OF WRAPPING BOUNDS')

    return lon, lat


def generate_displacement_stats(df_disp, x, y, x_disp, y_disp, urb_rur_col, rural_val, stats=None):
    """
    Run diagnostic summary statistics on distance between original
    and displaced point
    Parameters
    ----------
    df_disp - Dataframe with original and displaced coordinates
    x - Column with original longitude degrees which were input into displacement
    y - Column with original latitude degrees which were input into displacement
    urb_rur_col - Column with urban/rural information
    rural_val - Value for rural
    stats - List of summary statistics to run. For example, mean, median, min, maximamum

    Returns Saves summary statistics as a CSV file
    -------

    """
    # ===================================
    # CALCULATE DISTANCES
    # ===================================
    df = df_disp.copy(deep=True)
    df['dist'] = df.apply(lambda row: distance_latlon_points((row[x], row[y]),
                                                             (row[x_disp], row[y_disp])), axis=1)

    # ===================================
    # GET SUMMARY STATISTICS
    # ===================================
    if isinstance(rural_val, str):
        df[urb_rur_col] = df[urb_rur_col].apply(lambda x: x.lower())
        df[urb_rur_col] = df[urb_rur_col].map({'rural': 1, 'urban': 2})
        rural_val = 1

    df_rur = df.query('{} == {}'.format(urb_rur_col, rural_val))
    assert len(df[urb_rur_col].unique()) == 2, 'THERE SHOULD BE ONLY TWO URBAN/RURAL VALUES'
    urb_code = list(set(df[urb_rur_col].unique()) - {rural_val})[0]
    df_urb = df.query('{} == {}'.format(urb_rur_col, urb_code))

    res = []
    if 'mean' in stats:
        urb_mean = df_urb['dist'].mean()
        rur_mean = df_rur['dist'].mean()
        res.append({'stat': 'mean', 'area': 'urban', 'value': urb_mean})
        res.append({'stat': 'mean', 'area': 'rural', 'value': rur_mean})
    if 'median' in stats:
        urb_med = df_urb['dist'].median()
        rur_med = df_rur['dist'].median()
        res.append({'stat': 'median', 'area': 'urban', 'value': urb_med})
        res.append({'stat': 'median', 'area': 'rural', 'value': rur_med})

    if 'min' in stats or 'minimum' in stats:
        urb_min = df_urb['dist'].min()
        rur_min = df_rur['dist'].min()
        res.append({'stat': 'minimum', 'area': 'urban', 'value': urb_min})
        res.append({'stat': 'minimum', 'area': 'rural', 'value': rur_min})

    if 'max' in stats or 'maximum' in stats:
        urb_max = df_urb['dist'].max()
        rur_max = df_rur['dist'].max()
        res.append({'stat': 'maximum', 'area': 'urban', 'value': urb_max})
        res.append({'stat': 'maximum', 'area': 'rural', 'value': rur_max})

    # ===================================
    # PRINT AND SAVE SUMMARY STATISTICS
    # ===================================
    df_stats = pd.DataFrame(res)
    print()
    print("=" * 65)
    print(' Stats based off distance between original & displaced coords')
    print("=" * 65)
    print(df_stats)

    return df_stats
