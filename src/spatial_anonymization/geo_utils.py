"""
Geodesic and spatial utility functions.

This module provides low-level geographic helpers used throughout the
spatial anonymization package.

Conventions:
- Coordinates are always provided as (lon, lat) in decimal degrees.
- Distances are provided in meters (input) and kilometers (output).
- Internal trigonometric calculations use radians.
"""

import random
from typing import Tuple, Optional
from math import radians, degrees, asin, atan2, sin, cos

from shapely.geometry import LineString, Point
from pyproj import Geod


# ---------------------------------------------------------------------
# DISTANCE UTILITIES
# ---------------------------------------------------------------------

def distance_points(
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    ellipsoid: str = "WGS84",
) -> float:
    """
    Compute geodesic distance between two points.

    Args:
        pt1 (tuple[float, float]): First point as (lon, lat) in degrees.
        pt2 (tuple[float, float]): Second point as (lon, lat) in degrees.
        ellipsoid (str): Reference ellipsoid. Default is "WGS84".

    Returns:
        float: Distance between points in kilometers.
    """
    line = LineString(
        [Point(pt1[0], pt1[1]), Point(pt2[0], pt2[1])]
    )
    geod = Geod(ellps=ellipsoid)
    return geod.geometry_length(line) / 1000.0


# ---------------------------------------------------------------------
# GEODESIC DISPLACEMENT
# ---------------------------------------------------------------------

def displace_point_geodesic(
    lon: float,
    lat: float,
    distance_m: float,
    bearing: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Displace a geographic point along a geodesic by a fixed distance.

    Args:
        lon (float): Longitude of the original point (decimal degrees).
        lat (float): Latitude of the original point (decimal degrees).
        distance_m (float): Displacement distance in meters.
        bearing (float, optional): Bearing in degrees (0° = North,
            90° = East). If None, a random bearing is used.

    Returns:
        tuple[float, float]: Displaced point as (lon, lat) in degrees.

    Raises:
        ValueError: If distance_m is negative.
    """
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")

    R_km = 6371.0
    distance_km = distance_m / 1000.0

    if bearing is None:
        bearing = random.uniform(0.0, 360.0)

    lat1 = radians(lat)
    lon1 = radians(lon)
    bearing_rad = radians(bearing)

    delta = distance_km / R_km

    lat2 = asin(
        sin(lat1) * cos(delta)
        + cos(lat1) * sin(delta) * cos(bearing_rad)
    )

    lon2 = lon1 + atan2(
        sin(bearing_rad) * sin(delta) * cos(lat1),
        cos(delta) - sin(lat1) * sin(lat2),
    )

    return degrees(lon2), degrees(lat2)


# ---------------------------------------------------------------------
# PLANAR DISPLACEMENT
# ---------------------------------------------------------------------
def displace_point_planar(
    x: float,
    y: float,
    distance_m: float,
    bearing: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Displace a point in a planar (projected) coordinate system.

    This function assumes the input coordinates are in a projected CRS
    with units in meters (e.g. UTM). The displacement is performed using
    simple trigonometry and does not account for Earth curvature.

    Args:
        x (float): X-coordinate (e.g. Easting) in meters.
        y (float): Y-coordinate (e.g. Northing) in meters.
        distance_m (float): Displacement distance in meters.
        bearing (float, optional): Bearing in degrees (0° = North,
            90° = East). If None, a random bearing is used.

    Returns:
        tuple[float, float]: Displaced point as (x, y) in meters.

    Raises:
        ValueError: If distance_m is negative.

    Notes:
        - Use this only when working in a projected CRS.
        - Suitable for small-area operations and constraint checks.
        - Faster than geodesic displacement but less globally accurate.
    """
    if distance_m < 0:
        raise ValueError("distance_m must be non-negative")

    if bearing is None:
        bearing = random.uniform(0.0, 360.0)

    bearing_rad = radians(bearing)

    dx = distance_m * sin(bearing_rad)
    dy = distance_m * cos(bearing_rad)

    return x + dx, y + dy

