"""
Input data management for spatial anonymization.

This module defines InputsManager, a lightweight helper responsible
for acquiring, validating, and preparing spatial input data required
for anonymization workflows.

InputsManager does not perform anonymization. It prepares inputs
for SurveyDisplacer and related components.
"""

from typing import Optional
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
import requests
import tempfile
import os
from pathlib import Path
import warnings


class InputsManager:
    """
    Manages spatial input data required for anonymization.

    Inputs may be provided directly by the user or resolved
    through helper methods (e.g., downloading default layers).
    """

    def __init__(
        self,
        *,
        country_iso3: Optional[str] = None,
        admin_gdf: Optional[gpd.GeoDataFrame] = None,
        admin_level: int = 2,
        population_raster_path: Optional[str] = None,
        water_gdf: Optional[gpd.GeoDataFrame] = None,
        roads_gdf: Optional[gpd.GeoDataFrame] = None,
        utm_epsg: Optional[int] = None,
        auto_download: bool = True,
        download_dir: Optional[str] = None,
    ):
        """
        Args:
            country_iso3 (str, optional):
                ISO 3166-1 alpha-3 country code (e.g., 'MWI' for Malawi).
                If provided and admin_gdf is None, will download admin boundaries.
            admin_gdf (GeoDataFrame, optional):
                Administrative boundary geometries in WGS84 (EPSG:4326).
                Either this or country_iso3 must be provided.
            admin_level (int):
                Administrative level represented by admin_gdf
                (e.g., 0=country, 1=region, 2=district). Default: 2.
            population_raster_path (str, optional):
                Path to population raster (e.g., WorldPop).
            water_gdf (GeoDataFrame, optional):
                Water body geometries.
            roads_gdf (GeoDataFrame, optional):
                Road geometries.
            utm_epsg (int, optional):
                Projected CRS EPSG code used for distance calculations.
            auto_download (bool):
                Whether to automatically download missing data. Default: True.
            download_dir (str, optional):
                Directory for downloaded files. If None, uses system temp directory.
        """
        # Validate that we have either admin_gdf or country_iso3
        if admin_gdf is None and country_iso3 is None:
            raise ValueError("Must provide either admin_gdf or country_iso3")

        self.country_iso3 = country_iso3
        self.admin_level = admin_level
        self.auto_download = auto_download
        self.download_dir = download_dir or tempfile.gettempdir()
        
        # Download admin boundaries if needed
        if admin_gdf is None and country_iso3 is not None:
            if not auto_download:
                raise ValueError(
                    "auto_download is False but admin_gdf not provided. "
                    "Either provide admin_gdf or set auto_download=True."
                )
            admin_gdf = self._download_admin_boundaries()
        
        self.admin_gdf = admin_gdf
        self.population_raster_path = population_raster_path
        self.water_gdf = water_gdf
        self.roads_gdf = roads_gdf
        self.utm_epsg = utm_epsg

        self._validate_inputs()

    # ==========================================================
    # INPUT VALIDATION
    # ==========================================================

    def _validate_inputs(self):
        if self.admin_gdf.crs is None:
            raise ValueError("admin_gdf must have a CRS defined")

        if self.admin_gdf.crs.to_epsg() != 4326:
            raise ValueError("admin_gdf must be in WGS84 (EPSG:4326)")

        if not isinstance(self.admin_level, int) or self.admin_level < 0:
            raise ValueError("admin_level must be a non-negative integer")

        if self.water_gdf is not None and self.water_gdf.crs != self.admin_gdf.crs:
            self.water_gdf = self.water_gdf.to_crs(self.admin_gdf.crs)

        if self.roads_gdf is not None and self.roads_gdf.crs != self.admin_gdf.crs:
            self.roads_gdf = self.roads_gdf.to_crs(self.admin_gdf.crs)

    # ==========================================================
    # DATA DOWNLOAD METHODS
    # ==========================================================

    def _download_admin_boundaries(self) -> gpd.GeoDataFrame:
        """
        Download administrative boundaries from geoBoundaries or GADM.
        Tries geoBoundaries first, falls back to GADM if that fails.
        """
        if self.country_iso3 is None:
            raise ValueError("country_iso3 required for downloading admin boundaries")
        
        # Try geoBoundaries first
        try:
            return self._download_from_geoboundaries()
        except Exception as e:
            warnings.warn(
                f"geoBoundaries download failed: {e}. Trying GADM..."
            )
            # Fall back to GADM
            try:
                return self._download_from_gadm()
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to download admin boundaries from both geoBoundaries and GADM. "
                    f"geoBoundaries error: {e}, GADM error: {e2}"
                )

    def _download_from_geoboundaries(self) -> gpd.GeoDataFrame:
        """
        Download administrative boundaries from geoBoundaries API.
        Downloads data and saves as shapefile in download_dir.
        """
        # geoBoundaries API endpoint
        # ADM levels: ADM0 (country), ADM1 (state/region), ADM2 (district), etc.
        adm_code = f"ADM{self.admin_level}"
        url = f"https://www.geoboundaries.org/api/current/gbOpen/{self.country_iso3}/{adm_code}/"
        
        print(f"Downloading admin boundaries from geoBoundaries: {self.country_iso3} {adm_code}...")
        
        # Get the metadata
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        metadata = response.json()
        
        # Get the shapefile URL
        shapefile_url = metadata.get('gjDownloadURL') or metadata.get('simplifiedGeometryGeoJSON')
        if not shapefile_url:
            raise ValueError("No download URL found in geoBoundaries response")
        
        # Download and read the GeoJSON
        gdf = gpd.read_file(shapefile_url)
        
        # Ensure WGS84
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        
        # Save as shapefile
        output_path = os.path.join(
            self.download_dir, 
            f"{self.country_iso3}_admin_level{self.admin_level}_geoboundaries.shp"
        )
        gdf.to_file(output_path)
        print(f"Saved admin boundaries to {output_path}")
        
        print(f"Successfully downloaded {len(gdf)} admin boundaries from geoBoundaries")
        return gdf

    def _download_from_gadm(self) -> gpd.GeoDataFrame:
        """
        Download administrative boundaries from GADM.
        Downloads data and saves as shapefile in download_dir.
        """
        # GADM download URL pattern
        url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{self.country_iso3}_{self.admin_level}.json"
        
        print(f"Downloading admin boundaries from GADM: {self.country_iso3} level {self.admin_level}...")
        
        # Download and read the GeoJSON
        gdf = gpd.read_file(url)
        
        # Ensure WGS84
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        
        # Save as shapefile
        output_path = os.path.join(
            self.download_dir, 
            f"{self.country_iso3}_admin_level{self.admin_level}_gadm.shp"
        )
        gdf.to_file(output_path,  driver="ESRI Shapefile")
        print(f"Saved admin boundaries to {output_path}")
        
        print(f"Successfully downloaded {len(gdf)} admin boundaries from GADM")
        return gdf

    def download_population_raster(self, year: int = 2025) -> str:
        """
        Download WorldPop population raster for the country.
        Uses WorldPop Global 2015-2030 dataset (R2025A v1, constrained, 100m resolution).
        
        Args:
            year (int): Year for population data (2015-2030). Default: 2025.
            
        Returns:
            str: Path to downloaded raster file.
        """
        if self.country_iso3 is None:
            raise ValueError("country_iso3 required for downloading population raster")
        
        # WorldPop URL pattern (R2025A v1, constrained, 100m)
        # https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/{ISO3}/v1/100m/constrained/{iso3}_pop_{year}_CN_100m_R2025A_v1.tif
        country_lower = self.country_iso3.lower()
        country_upper = self.country_iso3.upper()
        url = f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{year}/{country_upper}/v1/100m/constrained/{country_lower}_pop_{year}_CN_100m_R2025A_v1.tif"
        
        print(f"Downloading WorldPop raster for {self.country_iso3} ({year}) from R2025A v1...")
        
        # Download file
        output_path = os.path.join(self.download_dir, f"{country_lower}_worldpop_{year}.tif")
        
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded WorldPop raster to {output_path}")
        self.population_raster_path = output_path
        return output_path

    def download_water_bodies(self) -> gpd.GeoDataFrame:
        """
        Download water bodies from OpenStreetMap using Overpass API.
        Downloads major water bodies (natural=water, waterway=*, etc.) within country bounds.
        
        Note: This method may fail due to Overpass API timeouts or rate limits.
        Returns None if download fails.
        """
        if self.admin_gdf is None:
            raise ValueError("Admin boundaries must be loaded before downloading water bodies")
        
        print(f"Downloading water bodies from OpenStreetMap for {self.country_iso3}...")
        
        # Get bounding box from admin boundaries
        bounds = self.admin_gdf.total_bounds  # (minx, miny, maxx, maxy)
        bbox_str = f"{bounds[1]},{bounds[0]},{bounds[3]},{bounds[2]}"  # (south,west,north,east)
        
        # Overpass API query for water bodies
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:90][bbox:{bbox_str}];
        (
          way["natural"="water"];
          way["waterway"~"river|stream|canal"];
          relation["natural"="water"];
          relation["waterway"~"river|stream|canal"];
        );
        out geom;
        """
        
        try:
            response = requests.post(overpass_url, data={'data': query}, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            warnings.warn(
                f"Failed to download water bodies from OpenStreetMap: {e}. "
                "This is often due to API timeouts or rate limits. Continuing without water data."
            )
            return None
        
        # Parse OSM data to GeoDataFrame
        import json
        from shapely.geometry import Polygon, LineString, MultiPolygon
        
        osm_data = response.json()
        geometries = []
        properties = []
        
        for element in osm_data.get('elements', []):
            if element.get('type') == 'way' and 'geometry' in element:
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                if len(coords) > 2:
                    if coords[0] == coords[-1]:  # Closed way = polygon
                        geometries.append(Polygon(coords))
                    else:  # Open way = linestring
                        geometries.append(LineString(coords))
                    properties.append(element.get('tags', {}))
        
        if not geometries:
            warnings.warn("No water bodies found in OSM data")
            return None
        
        water_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
        
        # Save as shapefile
        output_path = os.path.join(
            self.download_dir,
            f"{self.country_iso3}_water_bodies_osm.shp"
        )
        water_gdf.to_file(output_path, driver="ESRI Shapefile")
        print(f"Saved water bodies to {output_path}")
        
        print(f"Successfully downloaded {len(water_gdf)} water features from OSM")
        
        self.water_gdf = water_gdf
        return water_gdf

    # ==========================================================
    # ACCESSORS
    # ==========================================================

    def get_admin_gdf(self) -> gpd.GeoDataFrame:
        """
        Returns the administrative boundary GeoDataFrame.
        """
        return self.admin_gdf

    def get_admin_polygon(self) -> Polygon:
        """
        Returns the unified administrative boundary polygon.
        """
        return self.admin_gdf.unary_union

    def get_admin_level(self) -> int:
        """
        Returns the administrative level of the boundary.
        """
        return self.admin_level

    def get_water_bodies(self) -> Optional[gpd.GeoDataFrame]:
        """
        Returns water body geometries if available.
        """
        return self.water_gdf

    def get_water_gdf(self) -> Optional[gpd.GeoDataFrame]:
        """
        Returns water body geometries if available (alias for get_water_bodies).
        """
        return self.water_gdf

    def get_roads(self) -> Optional[gpd.GeoDataFrame]:
        """
        Returns road geometries if available.
        """
        return self.roads_gdf

    def get_roads_gdf(self) -> Optional[gpd.GeoDataFrame]:
        """
        Returns road geometries if available (alias for get_roads).
        """
        return self.roads_gdf

    def get_population_dataset(self):
        """
        Opens and returns the population raster using rasterio.
        """
        if self.population_raster_path is None:
            return None

        return rasterio.open(self.population_raster_path)

    def open_population_raster(self):
        """
        Opens the population raster using rasterio (alias for get_population_dataset).
        """
        return self.get_population_dataset()

    def get_utm_epsg(self) -> Optional[int]:
        """
        Returns UTM EPSG code. If not provided by user, infers from admin boundary centroid.
        """
        if self.utm_epsg is not None:
            return self.utm_epsg

        # Infer from admin boundary centroid
        centroid = self.admin_gdf.unary_union.centroid
        lon, lat = centroid.x, centroid.y

        # Calculate UTM zone from longitude
        zone = int((lon + 180) / 6) + 1
        
        # Northern or Southern hemisphere
        if lat >= 0:
            epsg = 32600 + zone  # North
        else:
            epsg = 32700 + zone  # South

        return epsg

    # ==========================================================
    # CONVENIENCE FLAGS
    # ==========================================================

    def has_admin(self) -> bool:
        """
        Returns True if administrative boundary is available.
        """
        return self.admin_gdf is not None

    def has_population(self) -> bool:
        """
        Returns True if a population raster is available.
        """
        return self.population_raster_path is not None

    def has_water(self) -> bool:
        """
        Returns True if water body geometries are available.
        """
        return self.water_gdf is not None

    def has_roads(self) -> bool:
        """
        Returns True if road geometries are available.
        """
        return self.roads_gdf is not None
