"""
High-level user-facing API for spatial anonymization.

This module provides simplified convenience functions for common
anonymization workflows, abstracting away the complexity of
creating InputsManager and AnonymizationParams objects.
"""

from pathlib import Path
from typing import Optional, Union, Literal

import geopandas as gpd
import pandas as pd

from spatial_anonymization.core.anonymizer import SurveyAnonymizer
from spatial_anonymization.config.params import AnonymizationParams
from spatial_anonymization.data.inputs_manager import InputsManager


def anonymize_survey_data(
    df: pd.DataFrame,
    admin_boundaries: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    *,
    country_iso3: Optional[str] = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    method: Literal["dhs", "donut", "population_adaptive"] = "dhs",
    unit_type: Literal["household", "cluster"] = "household",
    max_distance: float = 2000.0,
    min_distance: float = 0.0,
    urban_col: Optional[str] = None,
    urban_value: Optional[str] = "urban",
    rural_value: Optional[str] = "rural",
    urban_max_distance: Optional[float] = None,
    rural_max_distance: Optional[float] = None,
    admin_level: int = 2,
    population_raster: Optional[Union[str, Path]] = None,
    auto_download_population: bool = False,
    population_year: int = 2025,
    water_boundaries: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    auto_download_water: bool = False,
    roads: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    enforce_admin_boundary: bool = True,
    enforce_water_constraint: bool = True,
    enforce_road_constraint: bool = False,
    utm_epsg: Optional[int] = None,
    max_attempts: int = 50,
    random_seed: Optional[int] = None,
    out_lon_col: str = "lon_anon",
    out_lat_col: str = "lat_anon",
    group_id_col: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Anonymize survey GPS coordinates with a simplified interface.

    This function provides a streamlined way to anonymize survey data
    without manually creating InputsManager and AnonymizationParams objects.

    Args:
        df (DataFrame):
            Input survey data with GPS coordinates.
        admin_boundaries (str, Path, or GeoDataFrame, optional):
            Administrative boundaries as shapefile path or GeoDataFrame.
            Either this or country_iso3 must be provided.
        country_iso3 (str, optional):
            ISO 3166-1 alpha-3 country code (e.g., 'MWI' for Malawi).
            If provided, admin boundaries will be auto-downloaded.
            Either this or admin_boundaries must be provided.
        lon_col (str):
            Name of longitude column. Default: "lon".
        lat_col (str):
            Name of latitude column. Default: "lat".
        method (str):
            Anonymization method: "dhs", "donut", or "population_adaptive".
            Default: "dhs".
        unit_type (str):
            Type of spatial unit: "household" or "cluster". Default: "household".
        max_distance (float):
            Maximum displacement distance in meters. Default: 2000.0.
        min_distance (float):
            Minimum displacement distance in meters (for donut method). Default: 0.0.
        urban_col (str, optional):
            Column name indicating urban/rural classification.
        urban_value (str):
            Value in urban_col representing urban areas. Default: "urban".
        rural_value (str):
            Value in urban_col representing rural areas. Default: "rural".
        urban_max_distance (float, optional):
            Maximum distance for urban areas. If None, uses max_distance.
        rural_max_distance (float, optional):
            Maximum distance for rural areas. If None, uses max_distance.
        admin_level (int):
            Administrative level (0=country, 1=region, 2=district, etc.). Default: 2.
        population_raster (str, Path, optional):
            Path to population raster file (e.g., WorldPop) for population-adaptive method.
        auto_download_population (bool):
            Auto-download WorldPop raster if population_raster not provided. Default: False.
        population_year (int):
            Year for WorldPop data when auto-downloading (2015-2030). Default: 2025.
        water_boundaries (str, Path, or GeoDataFrame, optional):
            Water bodies to avoid when displacing points.
        auto_download_water (bool):
            Auto-download water bodies from OSM if water_boundaries not provided. Default: False.
        roads (str, Path, or GeoDataFrame, optional):
            Road network to avoid when displacing points.
        enforce_admin_boundary (bool):
            Whether displaced points must stay within admin boundaries. Default: True.
        enforce_water_constraint (bool):
            Whether to avoid water bodies. Default: True.
        enforce_road_constraint (bool):
            Whether to avoid roads. Default: False.
        utm_epsg (int, optional):
            UTM EPSG code for distance calculations. If None, inferred from admin boundaries.
        max_attempts (int):
            Maximum retry attempts per point. Default: 50.
        random_seed (int, optional):
            Random seed for reproducibility.
        out_lon_col (str):
            Output column name for anonymized longitude. Default: "lon_anon".
        out_lat_col (str):
            Output column name for anonymized latitude. Default: "lat_anon".
        group_id_col (str, optional):
            Column identifying clusters/groups for cluster-level anonymization.
        working_dir (str, optional):
            Working directory for downloaded files and outputs. If None, uses system temp directory.

    Returns:
        DataFrame: Input data with added anonymized coordinate columns.

    Examples:
        Basic usage with local shapefile:
        >>> df_anon = anonymize_survey_data(
        ...     df=survey_data,
        ...     admin_boundaries="data/districts.shp",
        ...     lon_col="longitude",
        ...     lat_col="latitude",
        ...     max_distance=2000
        ... )

        Auto-download admin boundaries using country code:
        >>> df_anon = anonymize_survey_data(
        ...     df=survey_data,
        ...     country_iso3="MWI",  # Malawi
        ...     admin_level=2,
        ...     max_distance=2000
        ... )

        Urban/rural differentiation:
        >>> df_anon = anonymize_survey_data(
        ...     df=survey_data,
        ...     admin_boundaries="data/districts.shp",
        ...     urban_col="ur_class",
        ...     urban_max_distance=1000,
        ...     rural_max_distance=5000
        ... )

        Donut geomasking with minimum distance:
        >>> df_anon = anonymize_survey_data(
        ...     df=survey_data,
        ...     country_iso3="KEN",  # Kenya
        ...     method="donut",
        ...     min_distance=500,
        ...     max_distance=2000
        ... )

        Population-adaptive with auto-downloaded data:
        >>> df_anon = anonymize_survey_data(
        ...     df=survey_data,
        ...     country_iso3="TZA",  # Tanzania
        ...     method="population_adaptive",
        ...     auto_download_population=True,
        ...     auto_download_water=True,
        ...     population_year=2020
        ... )
    """
    # Validate inputs
    if admin_boundaries is None and country_iso3 is None:
        raise ValueError("Must provide either admin_boundaries or country_iso3")
    # Validate inputs
    if admin_boundaries is None and country_iso3 is None:
        raise ValueError("Must provide either admin_boundaries or country_iso3")
    
    # Load admin boundaries if path provided
    admin_gdf = None
    if admin_boundaries is not None:
        if isinstance(admin_boundaries, (str, Path)):
            admin_gdf = gpd.read_file(admin_boundaries)
        else:
            admin_gdf = admin_boundaries.copy()

        # Ensure admin boundaries are in WGS84
        if admin_gdf.crs.to_epsg() != 4326:
            admin_gdf = admin_gdf.to_crs("EPSG:4326")

    # Load water boundaries if provided
    water_gdf = None
    if water_boundaries is not None:
        if isinstance(water_boundaries, (str, Path)):
            water_gdf = gpd.read_file(water_boundaries)
        else:
            water_gdf = water_boundaries.copy()

    # Load roads if provided
    roads_gdf = None
    if roads is not None:
        if isinstance(roads, (str, Path)):
            roads_gdf = gpd.read_file(roads)
        else:
            roads_gdf = roads.copy()

    # Convert population_raster to string if Path
    pop_raster_path = str(population_raster) if population_raster else None

    # Create InputsManager with download support
    inputs = InputsManager(
        country_iso3=country_iso3,
        admin_gdf=admin_gdf,
        admin_level=admin_level,
        population_raster_path=pop_raster_path,
        water_gdf=water_gdf,
        roads_gdf=roads_gdf,
        utm_epsg=utm_epsg,
        auto_download=True,
        download_dir=working_dir,
    )

    # Auto-download population raster if requested
    if auto_download_population and pop_raster_path is None:
        try:
            inputs.download_population_raster(year=population_year)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to download population raster: {e}")

    # Auto-download water bodies if requested
    if auto_download_water and water_gdf is None:
        try:
            inputs.download_water_bodies()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to download water bodies: {e}")

    # Create AnonymizationParams
    config = AnonymizationParams(
        method=method,
        unit_type=unit_type,
        max_distance=max_distance,
        min_distance=min_distance,
        urban_max_distance=urban_max_distance,
        rural_max_distance=rural_max_distance,
        enforce_admin_boundary=enforce_admin_boundary,
        enforce_water_constraint=enforce_water_constraint,
        enforce_road_constraint=enforce_road_constraint,
    )

    # Create anonymizer
    anonymizer = SurveyAnonymizer(
        config=config,
        inputs=inputs,
        max_attempts=max_attempts,
        random_seed=random_seed,
    )

    # Run anonymization
    return anonymizer.anonymize_dataframe(
        df,
        lon_col=lon_col,
        lat_col=lat_col,
        group_id_col=group_id_col,
        urban_col=urban_col,
        out_lon_col=out_lon_col,
        out_lat_col=out_lat_col,
    )


def anonymize_survey_data_from_config(
    df: pd.DataFrame,
    config: AnonymizationParams,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    urban_rural_col: Optional[str] = None,
    urban_value: Optional[str] = "urban",
    rural_value: Optional[str] = "rural",
    admin_boundaries: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    country_iso3: Optional[str] = None,
    population_raster: Optional[Union[str, Path]] = None,
    water_boundaries: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    roads: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    utm_epsg: Optional[int] = None,
    max_attempts: int = 50,
    random_seed: Optional[int] = None,
    group_id_col: Optional[str] = None,
    out_lon_col: str = "lon_anon",
    out_lat_col: str = "lat_anon",
) -> pd.DataFrame:
    """Anonymize survey GPS coordinates using a complete AnonymizationParams object.

    This function provides full control over anonymization policy and data sources
    by accepting a pre-configured AnonymizationParams object (which includes
    DataSourcesConfig). Use this for:
    - Fine-grained control over all anonymization parameters
    - Reproducible workflows using saved YAML/JSON configurations
    - Custom data source providers (e.g., LandScan instead of WorldPop)

    For simpler use cases, see anonymize_survey_data().

    Args:
        df (DataFrame):
            Input survey data with GPS coordinates.
        config (AnonymizationParams):
            Complete configuration object including anonymization policy
            and data sources (config.data_sources).
        lon_col (str):
            Name of longitude column. Default: "lon".
        lat_col (str):
            Name of latitude column. Default: "lat".
        urban_rural_col (str, optional):
            Column name indicating urban/rural classification.
        urban_value (str):
            Value in urban_rural_col representing urban areas. Default: "urban".
        rural_value (str):
            Value in urban_rural_col representing rural areas. Default: "rural".
        admin_boundaries (str, Path, or GeoDataFrame, optional):
            Administrative boundaries. If None, uses country_iso3 and
            config.data_sources settings to download.
        country_iso3 (str, optional):
            ISO 3166-1 alpha-3 country code for auto-downloading data.
        population_raster (str, Path, optional):
            Path to population raster. If None and method requires population,
            downloads using config.data_sources settings.
        water_boundaries (str, Path, or GeoDataFrame, optional):
            Water bodies. If None and config.enforce_water_constraint=True,
            downloads using config.data_sources settings.
        roads (str, Path, or GeoDataFrame, optional):
            Roads. If None and config.enforce_road_constraint=True,
            downloads using config.data_sources settings.
        utm_epsg (int, optional):
            UTM EPSG code. If None, inferred from admin boundaries.
        max_attempts (int):
            Maximum retry attempts per point. Default: 50.
        random_seed (int, optional):
            Random seed for reproducibility.
        group_id_col (str, optional):
            Column identifying clusters/groups for cluster-level anonymization.
        out_lon_col (str):
            Output column name for anonymized longitude. Default: "lon_anon".
        out_lat_col (str):
            Output column name for anonymized latitude. Default: "lat_anon".

    Returns:
        DataFrame: Input data with added anonymized coordinate columns.

    Examples:
        Load configuration from YAML file:
        >>> config = AnonymizationParams.from_yaml("my_config.yaml")
        >>> df_anon = anonymize_survey_data_from_config(
        ...     df=survey_data,
        ...     config=config,
        ...     country_iso3="MWI"
        ... )

        Custom data sources configuration:
        >>> from spatial_anonymization import DataSourcesConfig
        >>> data_sources = DataSourcesConfig(
        ...     admin_boundary_source="gadm",
        ...     population_source="worldpop",
        ...     population_year=2020,
        ...     water_source="none"
        ... )
        >>> config = AnonymizationParams(
        ...     method="dhs",
        ...     unit_type="household",
        ...     urban_max_distance=2000,
        ...     rural_max_distance=5000,
        ...     data_sources=data_sources
        ... )
        >>> df_anon = anonymize_survey_data_from_config(
        ...     df=survey_data,
        ...     config=config,
        ...     country_iso3="TZA"
        ... )
    """
    # Validate inputs
    if admin_boundaries is None and country_iso3 is None:
        raise ValueError("Must provide either admin_boundaries or country_iso3")

    # Load admin boundaries if path provided
    admin_gdf = None
    if admin_boundaries is not None:
        if isinstance(admin_boundaries, (str, Path)):
            admin_gdf = gpd.read_file(admin_boundaries)
        else:
            admin_gdf = admin_boundaries.copy()

        # Ensure admin boundaries are in WGS84
        if admin_gdf.crs.to_epsg() != 4326:
            admin_gdf = admin_gdf.to_crs("EPSG:4326")

    # Load water boundaries if provided
    water_gdf = None
    if water_boundaries is not None:
        if isinstance(water_boundaries, (str, Path)):
            water_gdf = gpd.read_file(water_boundaries)
        else:
            water_gdf = water_boundaries.copy()

    # Load roads if provided
    roads_gdf = None
    if roads is not None:
        if isinstance(roads, (str, Path)):
            roads_gdf = gpd.read_file(roads)
        else:
            roads_gdf = roads.copy()

    # Convert population_raster to string if Path
    pop_raster_path = str(population_raster) if population_raster else None

    # Create InputsManager using config.data_sources
    inputs = InputsManager(
        country_iso3=country_iso3,
        admin_gdf=admin_gdf,
        admin_level=config.admin_level,
        population_raster_path=pop_raster_path,
        water_gdf=water_gdf,
        roads_gdf=roads_gdf,
        utm_epsg=utm_epsg,
        auto_download=config.data_sources.auto_download,
        download_dir=config.data_sources.working_dir,
    )

    # Auto-download population if needed and not provided
    if config.method == "population_adaptive" and pop_raster_path is None:
        if config.data_sources.population_source == "worldpop":
            try:
                inputs.download_population_raster(
                    year=config.data_sources.population_year
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to download population raster: {e}")

    # Auto-download water if needed and not provided
    if config.enforce_water_constraint and water_gdf is None:
        if config.data_sources.water_source == "osm":
            try:
                inputs.download_water_bodies()
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to download water bodies: {e}")

    # Create anonymizer
    anonymizer = SurveyAnonymizer(
        config=config,
        inputs=inputs,
        max_attempts=max_attempts,
        random_seed=random_seed,
    )

    # Run anonymization using column names from function parameters
    return anonymizer.anonymize_dataframe(
        df,
        lon_col=lon_col,
        lat_col=lat_col,
        group_id_col=group_id_col,
        urban_col=urban_rural_col,
        out_lon_col=out_lon_col,
        out_lat_col=out_lat_col,
    )
