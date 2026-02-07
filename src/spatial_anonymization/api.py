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


def anonymize_survey_coordinates(
    df: pd.DataFrame,
    country_iso3: str,
    *,
    config: Optional[AnonymizationParams] = None,
    # Data locations
    admin_boundaries: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    population_raster: Optional[Union[str, Path]] = None,
    water_boundaries: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    roads: Optional[Union[str, Path, gpd.GeoDataFrame]] = None,
    # Input data schema
    lon_col: str = "lon",
    lat_col: str = "lat",
    urban_rural_col: Optional[str] = None,
    urban_value: Optional[str] = "urban",
    rural_value: Optional[str] = "rural",
    # Individual anonymization params (used if config is None)
    method: Literal["dhs", "donut", "population_adaptive"] = "dhs",
    unit_type: Literal["household", "cluster"] = "household",
    max_distance: float = 2000.0,
    min_distance: float = 0.0,
    urban_max_distance: Optional[float] = None,
    rural_max_distance: Optional[float] = None,
    rural_extra_max_distance: Optional[float] = None,
    rural_extra_proportion: float = 0.0,
    admin_level: int = 2,
    # Download and constraint options
    population_year: int = 2025,
    enforce_admin_boundary: bool = True,
    enforce_water_constraint: bool = True,
    enforce_road_constraint: bool = False,
    # Execution options
    utm_epsg: Optional[int] = None,
    max_attempts: int = 50,
    random_seed: Optional[int] = None,
    out_lon_col: str = "lon_anon",
    out_lat_col: str = "lat_anon",
    group_id_col: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Anonymize survey GPS coordinates.

    This function supports two usage modes:
    1. Provide a pre-configured AnonymizationParams object via `config`
    2. Specify anonymization parameters individually (method, max_distance, etc.)

    Args:
        df (DataFrame):
            Input survey data with GPS coordinates.
        country_iso3 (str):
            ISO 3166-1 alpha-3 country code (e.g., 'MWI' for Malawi).
            Required for auto-downloading admin boundaries and other spatial data.
        config (AnonymizationParams, optional):
            Pre-configured anonymization policy. If provided, individual
            anonymization parameters (method, max_distance, etc.) are ignored.
            Use this for reproducible workflows with saved configurations.
        admin_boundaries (str, Path, or GeoDataFrame, optional):
            Administrative boundaries as shapefile path or GeoDataFrame.
            If None, automatically downloads using country_iso3.
        population_raster (str, Path, optional):
            Path to population raster file (e.g., WorldPop).
            If None and method="population_adaptive", auto-downloads.
        water_boundaries (str, Path, or GeoDataFrame, optional):
            Water bodies to avoid when displacing points.
            If None and enforce_water_constraint=True, auto-downloads from OSM.
        roads (str, Path, or GeoDataFrame, optional):
            Road network to avoid when displacing points.
            If None and enforce_road_constraint=True, auto-downloads from OSM.
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
        method (str):
            Anonymization method: "dhs", "donut", or "population_adaptive".
            Only used if config is None. Default: "dhs".
        unit_type (str):
            Type of spatial unit: "household" or "cluster".
            Only used if config is None. Default: "household".
        max_distance (float):
            Maximum displacement distance in meters.
            Only used if config is None. Default: 2000.0.
        min_distance (float):
            Minimum displacement distance in meters (for donut method).
            Only used if config is None. Default: 0.0.
        urban_max_distance (float, optional):
            Maximum distance for urban areas.
            Only used if config is None.
        rural_max_distance (float, optional):
            Maximum distance for rural areas.
            Only used if config is None.
        rural_extra_max_distance (float, optional):
            Larger maximum distance for rural subset (DHS-style).
            Only used if config is None.
        rural_extra_proportion (float):
            Proportion of rural points receiving extra displacement.
            Only used if config is None. Default: 0.0.
        admin_level (int):
            Administrative level (0=country, 1=region, 2=district, etc.).
            Only used if config is None. Default: 2.
        population_year (int):
            Year for WorldPop data when auto-downloading (2015-2030).
            Only used if config is None or when auto-downloading. Default: 2025.
        enforce_admin_boundary (bool):
            Whether displaced points must stay within admin boundaries.
            Only used if config is None. Default: True.
        enforce_water_constraint (bool):
            Whether to avoid water bodies. If True and water_boundaries is None,
            auto-downloads from OSM. Only used if config is None. Default: True.
        enforce_road_constraint (bool):
            Whether to avoid roads. If True and roads is None,
            auto-downloads from OSM. Only used if config is None. Default: False.
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
            Working directory for downloaded files and outputs.
            Only used if config is None.

    Returns:
        DataFrame: Input data with added anonymized coordinate columns.

    Examples:
        Using individual parameters (admin boundaries auto-downloaded):
        >>> df_anon = anonymize_survey_coordinates(
        ...     df=survey_data,
        ...     country_iso3="MWI",
        ...     method="dhs",
        ...     max_distance=2000,
        ...     lon_col="longitude",
        ...     lat_col="latitude"
        ... )

        Using a preset configuration:
        >>> from spatial_anonymization.config.presets import DHS
        >>> df_anon = anonymize_survey_coordinates(
        ...     df=survey_data,
        ...     country_iso3="MWI",
        ...     config=DHS,
        ...     lon_col="longitude",
        ...     lat_col="latitude"
        ... )

        Provide your own admin boundaries:
        >>> df_anon = anonymize_survey_coordinates(
        ...     df=survey_data,
        ...     country_iso3="MWI",
        ...     admin_boundaries="path/to/districts.shp",
        ...     urban_rural_col="ur_class",
        ...     urban_max_distance=1000,
        ...     rural_max_distance=5000
        ... )

        Loading config from file:
        >>> config = AnonymizationParams.from_yaml("my_config.yaml")
        >>> df_anon = anonymize_survey_coordinates(
        ...     df=survey_data,
        ...     country_iso3="TZA",
        ...     config=config
        ... )
    """
    # Create or use provided config
    if config is None:
        # Build config from individual parameters
        config = AnonymizationParams(
            method=method,
            unit_type=unit_type,
            max_distance=max_distance,
            min_distance=min_distance,
            urban_max_distance=urban_max_distance,
            rural_max_distance=rural_max_distance,
            rural_extra_max_distance=rural_extra_max_distance,
            rural_extra_proportion=rural_extra_proportion,
            admin_level=admin_level,
            enforce_admin_boundary=enforce_admin_boundary,
            enforce_water_constraint=enforce_water_constraint,
            enforce_road_constraint=enforce_road_constraint,
        )
    
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
        auto_download=config.data_sources.auto_download if hasattr(config, 'data_sources') else True,
        download_dir=config.data_sources.working_dir if hasattr(config, 'data_sources') else working_dir,
    )

    # Auto-download population if needed and not provided
    if config.method == "population_adaptive" and pop_raster_path is None:
        # Auto-download if using individual params or if config specifies worldpop
        if config.data_sources.population_source == "worldpop":
            try:
                year = config.data_sources.population_year if hasattr(config, 'data_sources') else population_year
                inputs.download_population_raster(year=year)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to download population raster: {e}")

    # Auto-download water if needed, not provided, and constraint is enabled
    if config.enforce_water_constraint and water_gdf is None:
        # Auto-download if using individual params or if config specifies OSM
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


import pandas as pd
from typing import Hashable


def validate_input_schema(
    df: pd.DataFrame,
    *,
    lat_col: Hashable,
    lon_col: Hashable,
    urban_rural_col: Hashable | None = None,
    urban_value: Hashable | None = None,
    rural_value: Hashable | None = None,
) -> None:
    """
    Validate that input survey coordinates meet minimal requirements
    for spatial anonymization.

    This function performs lightweight checks to ensure that:
    - latitude and longitude columns exist, are numeric, and have no nulls
    - the urban/rural column (if provided) contains exactly two values
    - urban and rural labels are consistent in type and match expectations

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing survey coordinates.
    lat_col : Hashable
        Column name for latitude values.
    lon_col : Hashable
        Column name for longitude values.
    urban_rural_col : Hashable, optional
        Column name for urban/rural classification.
    urban_value : Hashable, optional
        Value representing urban areas.
    rural_value : Hashable, optional
        Value representing rural areas.

    Raises
    ------
    ValueError
        If any validation check fails.
    """

    # --------------------------------------------------
    # BASIC COLUMN EXISTENCE
    # --------------------------------------------------
    for col in (lat_col, lon_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # --------------------------------------------------
    # LAT / LON NUMERIC AND NON-NULL
    # --------------------------------------------------
    for col in (lat_col, lon_col):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' must be numeric (decimal degrees)."
            )

        if df[col].isna().any():
            raise ValueError(
                f"Column '{col}' contains missing (null) values. "
                "Please clean the data before anonymization."
            )

    # --------------------------------------------------
    # URBAN / RURAL VALIDATION (OPTIONAL)
    # --------------------------------------------------
    if urban_rural_col is not None:
        if urban_rural_col not in df.columns:
            raise ValueError(
                f"Urban/rural column '{urban_rural_col}' not found in dataframe."
            )

        unique_vals = df[urban_rural_col].dropna().unique()

        if len(unique_vals) != 2:
            raise ValueError(
                f"Urban/rural column '{urban_rural_col}' must contain exactly "
                f"two distinct values. Found: {list(unique_vals)}"
            )

        if urban_value is not None and rural_value is not None:
            if set(unique_vals) != {urban_value, rural_value}:
                raise ValueError(
                    f"Urban/rural values do not match expected labels.\n"
                    f"Expected: {{'{urban_value}', '{rural_value}'}}\n"
                    f"Found: {set(unique_vals)}"
                )

            if type(urban_value) is not type(rural_value):
                raise ValueError(
                    "Urban and rural labels must have the same data type "
                    "(both strings or both numeric)."
                )
    print("All Input Data Schema Validations Passed!")

def export_anonymized_data(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    *,
    format: Literal["shapefile", "geojson", "gpkg", "csv"] = "shapefile",
    lon_col: str = "lon_anon",
    lat_col: str = "lat_anon",
    crs: str = "EPSG:4326",
    drop_original_coords: bool = False,
    columns: Optional[list] = None,
    **kwargs
) -> Path:
    """
    Export anonymized survey data to geospatial format.

    Similar to geopandas.to_file(), this function provides a unified interface
    for exporting anonymized coordinates to various geospatial formats.

    Args:
        df (DataFrame):
            Anonymized survey data with coordinate columns.
        output_path (str or Path):
            Output file path. Extension is optional (will be added based on format).
        format (str):
            Output format. Options: "shapefile", "geojson", "gpkg", "csv".
            Default: "shapefile".
        lon_col (str):
            Column name containing anonymized longitude. Default: "lon_anon".
        lat_col (str):
            Column name containing anonymized latitude. Default: "lat_anon".
        crs (str):
            Coordinate reference system for output. Default: "EPSG:4326" (WGS84).
        drop_original_coords (bool):
            If True, removes original (non-anonymized) coordinate columns before export.
            Useful for ensuring original locations are not included in output files.
            Default: False.
        columns (list, optional):
            Specific columns to include in export. If None, exports all columns.
            Geometry column is always included for geospatial formats.
        **kwargs:
            Additional arguments passed to the underlying export function.
            For shapefile: driver-specific options
            For GeoJSON: indent, precision (default: 6)
            For GPKG: layer name

    Returns:
        Path: Path to the exported file.

    Raises:
        ValueError: If format is not supported or required columns are missing.
        FileNotFoundError: If output directory does not exist.

    Examples:
        Export to shapefile:
        >>> df_anon = anonymize_survey_coordinates(df, "MWI", method="dhs")
        >>> export_anonymized_data(
        ...     df_anon,
        ...     "output/anonymized.shp",
        ...     format="shapefile"
        ... )

        Export to GeoJSON with specific columns:
        >>> export_anonymized_data(
        ...     df_anon,
        ...     "output/anonymized.geojson",
        ...     format="geojson",
        ...     columns=["hh_id", "urb_rur", "lon_anon", "lat_anon"],
        ...     drop_original_coords=True
        ... )

        Export to GeoPackage:
        >>> export_anonymized_data(
        ...     df_anon,
        ...     "output/data.gpkg",
        ...     format="gpkg",
        ...     layer="anonymized_survey"
        ... )

        Export to CSV (coordinates as columns):
        >>> export_anonymized_data(
        ...     df_anon,
        ...     "output/anonymized.csv",
        ...     format="csv"
        ... )
    """
    from shapely.geometry import Point
    
    # Validate inputs
    if format not in ["shapefile", "geojson", "gpkg", "csv"]:
        raise ValueError(
            f"Unsupported format: '{format}'. "
            f"Supported formats: shapefile, geojson, gpkg, csv"
        )
    
    # Check required columns exist
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(
            f"Required coordinate columns '{lon_col}' and '{lat_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Prepare output path
    output_path = Path(output_path)
    
    # Auto-add extension if not present
    format_extensions = {
        "shapefile": ".shp",
        "geojson": ".geojson",
        "gpkg": ".gpkg",
        "csv": ".csv"
    }
    
    if output_path.suffix.lower() not in [format_extensions[format], ""]:
        import warnings
        warnings.warn(
            f"File extension '{output_path.suffix}' does not match format '{format}'. "
            f"Expected: '{format_extensions[format]}'"
        )
    
    if not output_path.suffix:
        output_path = output_path.with_suffix(format_extensions[format])
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare dataframe
    df_export = df.copy()
    
    # Drop original coordinates if requested
    if drop_original_coords:
        # Try to identify original coordinate columns (common patterns)
        original_coord_patterns = [
            'lat', 'lon', 'latitude', 'longitude',
            'y', 'x', 'lat_orig', 'lon_orig'
        ]
        
        cols_to_drop = []
        for col in df_export.columns:
            col_lower = col.lower()
            # Don't drop the anonymized columns
            if col == lon_col or col == lat_col:
                continue
            # Drop if matches pattern
            if any(pattern in col_lower for pattern in original_coord_patterns):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df_export = df_export.drop(columns=cols_to_drop)
            import warnings
            warnings.warn(f"Dropped original coordinate columns: {cols_to_drop}")
    
    # Select specific columns if requested
    if columns is not None:
        # Ensure coordinate columns are included
        required_cols = {lon_col, lat_col}
        missing_cols = required_cols - set(columns)
        if missing_cols:
            columns = columns + list(missing_cols)
        
        df_export = df_export[columns]
    
    # Remove rows with null coordinates
    df_export = df_export.dropna(subset=[lon_col, lat_col])
    
    if len(df_export) == 0:
        raise ValueError("No valid coordinates found after removing null values")
    
    # Export based on format
    if format == "csv":
        # Simple CSV export (no geometry conversion needed)
        df_export.to_csv(output_path, index=False, **kwargs)
        
    else:
        # Convert to GeoDataFrame for geospatial formats
        geometry = [Point(xy) for xy in zip(df_export[lon_col], df_export[lat_col])]
        gdf = gpd.GeoDataFrame(df_export, geometry=geometry, crs=crs)
        
        # Drop coordinate columns to avoid duplication (geometry has them)
        gdf = gdf.drop(columns=[lon_col, lat_col])
        
        # Export to appropriate format
        if format == "shapefile":
            driver = kwargs.pop('driver', 'ESRI Shapefile')
            gdf.to_file(output_path, driver=driver, **kwargs)
            
        elif format == "geojson":
            # Set default precision for GeoJSON
            if 'precision' not in kwargs:
                kwargs['precision'] = 6
            gdf.to_file(output_path, driver='GeoJSON', **kwargs)
            
        elif format == "gpkg":
            layer = kwargs.pop('layer', 'anonymized_data')
            gdf.to_file(output_path, driver='GPKG', layer=layer, **kwargs)
    
    print(f"✓ Exported {len(df_export)} records to {output_path} ({format})")
    return output_path