"""
Population providers for spatial anonymization.

This module defines a standard interface for estimating population
within a given radius around a geographic point. The goal is to
decouple population logic (e.g. raster-based estimation) from
anonymization logic, making the system easier to test, extend,
and reason about.

Typical usage:
- radius selection logic calls a PopulationProvider
- SurveyDisplacer orchestrates how population is used
"""
from typing import Callable
from abc import ABC, abstractmethod
from typing import Optional

import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point


class PopulationProvider(ABC):
    """
    Abstract base class for population estimation.

    Implementations must provide a method to estimate population
    within a circular buffer around a geographic point.
    """

    @abstractmethod
    def population_within_radius(
        self,
        lon: float,
        lat: float,
        radius_m: float,
    ) -> int:
        """
        Estimate population within a radius around a point.

        Args:
            lon (float): Longitude in decimal degrees.
            lat (float): Latitude in decimal degrees.
            radius_m (float): Buffer radius in meters.

        Returns:
            int: Estimated population count.
        """
        raise NotImplementedError

class RasterPopulationProvider(PopulationProvider):
    """
    Population provider backed by a population raster.

    This implementation uses a raster dataset (e.g. WorldPop,
    HRSL, GPW) and zonal statistics to estimate population within
    a buffer around a point.
    """

    def __init__(
        self,
        raster_path: str,
        raster_crs: Optional[str] = None,
    ):
        """
        Initialize the raster population provider.

        Args:
            raster_path (str): Path to population raster (GeoTIFF).
            raster_crs (str, optional): CRS override if raster has
                missing or incorrect CRS metadata.
        """
        self.raster_path = raster_path
        self.raster_crs = raster_crs

    def population_within_radius(
        self,
        lon: float,
        lat: float,
        radius_m: float,
    ) -> int:
        """
        Estimate population within a radius using raster zonal stats.

        Args:
            lon (float): Longitude in decimal degrees.
            lat (float): Latitude in decimal degrees.
            radius_m (float): Buffer radius in meters.

        Returns:
            int: Estimated population count.
        """
        # Create point GeoDataFrame (WGS84)
        point_gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)],
            crs="EPSG:4326",
        )

        with rasterio.open(self.raster_path) as src:
            if src.crs is None and self.raster_crs:
                raster_crs = self.raster_crs
            else:
                raster_crs = src.crs

            # Project point to raster CRS
            point_proj = point_gdf.to_crs(raster_crs)

            # Buffer in meters (valid in projected CRS)
            buffer_geom = point_proj.buffer(radius_m)

            stats = zonal_stats(
                buffer_geom,
                self.raster_path,
                stats=["sum"],
                nodata=src.nodata,
            )

        pop_sum = stats[0].get("sum", 0)

        if pop_sum is None:
            return 0

        return int(pop_sum)


def find_radius_for_population(
    population_fn: Callable[[float, float, float], int],
    lon: float,
    lat: float,
    population_threshold: int,
    min_radius_m: float,
    max_radius_m: float,
    step_m: float = 100.0,
) -> float:
    """Find the smallest radius that satisfies a population threshold.

    This helper incrementally increases a radius around a point until
    the population within that radius meets or exceeds a required
    threshold. The population is provided via a callable, allowing
    raster-backed, API-backed, or synthetic population sources.

    Args:
        population_fn (Callable): Function that returns population given
            (lon, lat, radius_m).
        lon (float): Longitude of the point (decimal degrees).
        lat (float): Latitude of the point (decimal degrees).
        population_threshold (int): Minimum required population.
        min_radius_m (float): Minimum search radius in meters.
        max_radius_m (float): Maximum allowed search radius in meters.
        step_m (float, optional): Increment step in meters when expanding
            the radius. Defaults to 100.0.

    Returns:
        float: Radius in meters at which the population threshold is met.

    Raises:
        ValueError: If the population threshold cannot be reached within
            the maximum radius.
    """
    if min_radius_m <= 0:
        raise ValueError("min_radius_m must be positive")

    if max_radius_m <= min_radius_m:
        raise ValueError("max_radius_m must be greater than min_radius_m")

    if step_m <= 0:
        raise ValueError("step_m must be positive")

    radius = min_radius_m

    while radius <= max_radius_m:
        population = population_fn(lon, lat, radius)

        if population >= population_threshold:
            return radius

        radius += step_m

    raise ValueError(
        "Population threshold could not be reached within max_radius_m"
    )
