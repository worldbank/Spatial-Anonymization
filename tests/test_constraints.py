"""
Tests for spatial anonymization geometric constraints.

These tests use synthetic geometries constructed with Shapely and
GeoPandas to mirror how constraints are applied in production, where
candidate points and constraint layers are handled as GeoDataFrames.

No real-world data or files are required.
"""

import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString

from spatial_anonymization.constraints import (
    is_within_admin,
    is_in_water,
    is_on_road,
    passes_constraints,
)


# =====================================================================
# FIXTURES: SYNTHETIC GEOMETRIES IN GEODATAFRAME FORM
# =====================================================================

def make_admin_gdf():
    """Square admin boundary from (0,0) to (10,10)."""
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


def make_water_gdf():
    """Small water body inside admin boundary."""
    return gpd.GeoDataFrame(
        geometry=[Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])],
        crs="EPSG:4326",
    )


def make_roads_gdf():
    """Single straight road crossing the admin boundary."""
    return gpd.GeoDataFrame(
        geometry=[LineString([(0, 5), (10, 5)])],
        crs="EPSG:4326",
    )


# =====================================================================
# ADMIN BOUNDARY CONSTRAINT TESTS
# =====================================================================

def test_is_within_admin_true():
    """
    Verify that a point inside the admin boundary passes the admin check.
    """
    admin = make_admin_gdf()
    point = Point(5, 5)

    assert is_within_admin(point, admin) is True


def test_is_within_admin_false():
    """
    Verify that a point outside the admin boundary fails the admin check.
    """
    admin = make_admin_gdf()
    point = Point(15, 5)

    assert is_within_admin(point, admin) is False


# =====================================================================
# WATER BODY CONSTRAINT TESTS
# =====================================================================

def test_is_in_water_false_for_land_point():
    """
    Verify that a point outside all water bodies is not flagged as water.
    """
    water = make_water_gdf()
    point = Point(1, 1)

    assert is_in_water(point, water) is False


def test_is_in_water_true_for_water_point():
    """
    Verify that a point inside a water polygon is correctly detected.
    """
    water = make_water_gdf()
    point = Point(3, 3)

    assert is_in_water(point, water) is True


# =====================================================================
# ROAD CONSTRAINT TESTS
# =====================================================================

def test_is_on_road_false_when_far():
    """
    Verify that a point far from a road is not flagged as on-road.
    """
    roads = make_roads_gdf()
    point = Point(5, 2)

    assert is_on_road(point, roads, buffer_m=0.5) is False


def test_is_on_road_true_when_on_line():
    """
    Verify that a point lying directly on a road geometry is detected.
    """
    roads = make_roads_gdf()
    point = Point(5, 5)

    assert is_on_road(point, roads, buffer_m=0.5) is True


# ==============================================================
# COMBINED CONSTRAINT TESTS
# ==============================================================

def test_passes_constraints_all_clear():
    """
    Verify that a point satisfying all enabled constraints passes.
    """
    admin = make_admin_gdf()
    water = make_water_gdf()
    roads = make_roads_gdf()

    point = Point(1, 1)

    assert passes_constraints(
        point,
        admin_geom=admin,
        water_gdf=water,
        roads_gdf=roads,
        allow_roads=True,
    ) is True


def test_passes_constraints_fails_water():
    """
    Verify that a point inside water fails combined constraints.
    """
    admin = make_admin_gdf()
    water = make_water_gdf()

    point = Point(3, 3)

    assert passes_constraints(
        point,
        admin_geom=admin,
        water_gdf=water,
    ) is False


def test_passes_constraints_fails_road_when_disallowed():
    """
    Verify that a point on a road fails when roads are disallowed.
    """
    admin = make_admin_gdf()
    roads = make_roads_gdf()

    point = Point(5, 5)

    assert passes_constraints(
        point,
        admin_geom=admin,
        roads_gdf=roads,
        road_buffer_m=0.5,
        allow_roads=False,
    ) is False
