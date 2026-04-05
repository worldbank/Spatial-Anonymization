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
from typing import Callable, List, Tuple
from abc import ABC, abstractmethod
from typing import Optional


import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point
from shapely.ops import transform
import pyproj
from spatial_anonymization.geoprocessing.geo_utils import get_utm_epsg


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

    Raster metadata (CRS, nodata) and the array are read once at
    init to avoid reopening the file on every call. Points are
    projected to a metric CRS (UTM) for buffering in meters, then
    buffers are reprojected back to the raster CRS for zonal stats.
    """

    def __init__(
        self,
        raster_path: str,
        raster_crs: Optional[str] = None,
        utm_epsg: Optional[int] = None,
    ):
        """
        Initialize the raster population provider.

        Args:
            raster_path (str): Path to population raster (GeoTIFF).
            raster_crs (str, optional): CRS override if raster has
                missing or incorrect CRS metadata.
            utm_epsg (int, optional): EPSG code for a metric CRS used
                for buffering. If None, estimated from the raster centroid.
        """
        self.raster_path = raster_path

        # Cache raster data once
        with rasterio.open(raster_path) as src:
            self._raster_crs = raster_crs or src.crs
            self._nodata = src.nodata
            self._affine = src.transform
            self._raster_array = src.read(1)

            # Determine a metric CRS for buffering
            if utm_epsg is not None:
                self._metric_crs = f"EPSG:{utm_epsg}"
            else:
                # Estimate UTM zone from raster centroid
                bounds = src.bounds
                center_lon = (bounds.left + bounds.right) / 2
                center_lat = (bounds.bottom + bounds.top) / 2
                self._metric_crs = f"EPSG:{get_utm_epsg(center_lon, center_lat)}"

        # Cache transformers for performance
        self._to_metric = pyproj.Transformer.from_crs("EPSG:4326", self._metric_crs, always_xy=True).transform
        self._to_raster = pyproj.Transformer.from_crs(self._metric_crs, self._raster_crs, always_xy=True).transform

    def population_within_radius(
        self,
        lon: float,
        lat: float,
        radius_m: float,
    ) -> int:
        """
        Estimate population within a radius using raster zonal stats.

        Projects the point to a metric CRS for buffering, then
        reprojects the buffer to the raster CRS for extraction.
        """
        # Project point to metric CRS
        pt_metric = transform(self._to_metric, Point(lon, lat))
        buffer_metric = pt_metric.buffer(radius_m)
        buffer_raster_crs = transform(self._to_raster, buffer_metric)

        stats = zonal_stats(
            [buffer_raster_crs],
            self._raster_array,
            affine=self._affine,
            stats=["sum"],
            nodata=self._nodata,
        )

        pop_sum = stats[0].get("sum", 0)
        return int(pop_sum) if pop_sum is not None else 0

    def batch_population_within_radius(
        self,
        points: List[Tuple[float, float]],
        radius_m: float,
    ) -> np.ndarray:
        """
        Estimate population for many points in a single zonal_stats call.

        Projects points to a metric CRS for buffering in meters,
        then reprojects buffers to the raster CRS for extraction.
        The raster array and affine are passed directly to avoid
        reopening the file.

        Args:
            points: List of (lon, lat) tuples in decimal degrees.
            radius_m: Buffer radius in meters (same for all points).

        Returns:
            numpy array of int population values, one per input point.
        """
        if not points:
            return np.array([], dtype=int)

        buffers_raster_crs = []
        for lon, lat in points:
            pt_metric = transform(self._to_metric, Point(lon, lat))
            buffer_metric = pt_metric.buffer(radius_m)
            buffer_raster = transform(self._to_raster, buffer_metric)
            buffers_raster_crs.append(buffer_raster)

        stats = zonal_stats(
            buffers_raster_crs,
            self._raster_array,
            affine=self._affine,
            stats=["sum"],
            nodata=self._nodata,
        )

        return np.array(
            [int(s.get("sum", 0)) if s.get("sum") is not None else 0
             for s in stats],
            dtype=int,
        )


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
