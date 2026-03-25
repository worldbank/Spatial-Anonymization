"""
Geometric constraints for spatial anonymization.

This module defines simple, atomic predicate functions that evaluate
whether a candidate anonymized point satisfies specific spatial
constraints such as administrative boundaries, water bodies, and roads.

These predicates are designed to be:
- deterministic
- independently testable
- composable in higher-level anonymization workflows
"""

from typing import Optional

import geopandas as gpd
from shapely.geometry import Point, Polygon


# ==============================================================
# ATOMIC CONSTRAINT PREDICATES
# ==============================================================

def is_within_admin(
    point: Point,
    admin_geom: Polygon,
) -> bool:
    """Check whether a point lies within an administrative boundary.

    Args:
        point (Point): Candidate anonymized point.
        admin_geom (Polygon): Administrative boundary geometry.

    Returns:
        bool: True if point is inside the admin boundary.
    """
    return admin_geom.contains(point)


def is_in_water(
    point: Point,
    water_gdf: Optional[gpd.GeoDataFrame],
) -> bool:
    """Check whether a point falls inside a water body.

    Uses spatial index for fast O(log n) lookup instead of brute-force scan.

    Args:
        point (Point): Candidate anonymized point.
        water_gdf (GeoDataFrame, optional): Water body geometries.

    Returns:
        bool: True if point intersects a water body.
    """
    if water_gdf is None or water_gdf.empty:
        return False

    hits = water_gdf.sindex.query(point, predicate="intersects")
    return len(hits) > 0


def is_on_road(
    point: Point,
    roads_gdf: Optional[gpd.GeoDataFrame],
    buffer_m: float,
) -> bool:
    """Check whether a point lies on or near a road.

    A buffer is applied to road geometries to allow proximity-based
    exclusion.

    Args:
        point (Point): Candidate anonymized point.
        roads_gdf (GeoDataFrame, optional): Road geometries.
        buffer_m (float): Buffer distance around roads.

    Returns:
        bool: True if point lies within buffered road geometry.
    """
    if roads_gdf is None or roads_gdf.empty:
        return False

    return bool(roads_gdf.buffer(buffer_m).contains(point).any())


# ==============================================================
# COMBINED CONSTRAINT EVALUATOR (OPTIONAL CONVENIENCE)
# ==============================================================

def passes_constraints(
    point: Point,
    *,
    admin_geom: Optional[Polygon] = None,
    prepared_admin=None,
    water_gdf: Optional[gpd.GeoDataFrame] = None,
    roads_gdf: Optional[gpd.GeoDataFrame] = None,
    road_buffer_m: float = 0.0,
    allow_roads: bool = True,
) -> bool:
    """Evaluate whether a point satisfies all enabled spatial constraints.

    This function composes individual constraint predicates and is
    intended for use inside anonymization loops.

    Args:
        point (Point): Candidate anonymized point.
        admin_geom (Polygon, optional): Admin boundary constraint.
        water_gdf (GeoDataFrame, optional): Water body constraint.
        roads_gdf (GeoDataFrame, optional): Road geometries.
        road_buffer_m (float): Buffer distance for road exclusion.
        allow_roads (bool): Whether points on roads are permitted.

    Returns:
        bool: True if all enabled constraints are satisfied.
    """
    if prepared_admin is not None:
        if not prepared_admin.contains(point):
            return False
    elif admin_geom is not None and not is_within_admin(point, admin_geom):
        return False

    if water_gdf is not None and is_in_water(point, water_gdf):
        return False

    if (
        roads_gdf is not None
        and not allow_roads
        and is_on_road(point, roads_gdf, road_buffer_m)
    ):
        return False

    return True
