import tempfile
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, LineString

from spatial_anonymization.inputs_manager import InputsManager


# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture
def admin_gdf_wgs84():
    """Simple square admin boundary in WGS84."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")


@pytest.fixture
def admin_gdf_projected():
    """Admin boundary in projected CRS (to test reprojection)."""
    poly = Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)])
    return gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")


@pytest.fixture
def water_gdf():
    """Simple water polygon."""
    poly = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
    return gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")


@pytest.fixture
def roads_gdf():
    """Simple road line."""
    line = LineString([(0, 5), (10, 5)])
    return gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs="EPSG:4326")


@pytest.fixture
def population_raster_file(tmp_path):
    """Create a tiny synthetic population raster."""
    raster_path = tmp_path / "population.tif"

    data = np.ones((10, 10), dtype=np.float32)
    transform = from_origin(0, 10, 1, 1)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return raster_path


# ==========================================================
# BASIC VALIDATION TESTS
# ==========================================================

def test_invalid_admin_level_raises(admin_gdf_wgs84):
    with pytest.raises(ValueError):
        InputsManager(
            admin_gdf=admin_gdf_wgs84,
            admin_level=-1,
        )


def test_missing_crs_raises():
    """Test that admin_gdf without CRS raises error."""
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf_no_crs = gpd.GeoDataFrame({"id": [1]}, geometry=[poly])
    
    with pytest.raises(ValueError, match="must have a CRS"):
        InputsManager(
            admin_gdf=gdf_no_crs,
            admin_level=2,
        )


def test_wrong_crs_raises():
    """Test that admin_gdf not in WGS84 raises error."""
    poly = Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)])
    gdf_projected = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")
    
    with pytest.raises(ValueError, match="must be in WGS84"):
        InputsManager(
            admin_gdf=gdf_projected,
            admin_level=2,
        )


# ==========================================================
# USER-PROVIDED INPUTS
# ==========================================================

def test_user_provided_admin_gdf_loaded(admin_gdf_wgs84):
    im = InputsManager(
        admin_gdf=admin_gdf_wgs84,
        admin_level=2,
    )

    assert im.has_admin()
    assert im.get_admin_gdf().crs.to_epsg() == 4326


def test_user_provided_water_and_roads(admin_gdf_wgs84, water_gdf, roads_gdf):
    im = InputsManager(
        admin_gdf=admin_gdf_wgs84,
        admin_level=2,
        water_gdf=water_gdf,
        roads_gdf=roads_gdf,
    )

    assert im.has_water()
    assert im.has_roads()


def test_population_raster_path_loaded(admin_gdf_wgs84, population_raster_file):
    im = InputsManager(
        admin_gdf=admin_gdf_wgs84,
        admin_level=2,
        population_raster_path=str(population_raster_file),
    )

    pop = im.get_population_dataset()
    assert pop is not None
    assert pop.count == 1


# ==========================================================
# UTM INFERENCE
# ==========================================================

def test_utm_epsg_inferred_from_admin(admin_gdf_wgs84):
    im = InputsManager(
        admin_gdf=admin_gdf_wgs84,
        admin_level=2,
    )

    utm = im.get_utm_epsg()
    assert utm is not None
    assert str(utm).startswith(("326", "327"))


def test_user_provided_utm_epsg_overrides(admin_gdf_wgs84):
    im = InputsManager(
        admin_gdf=admin_gdf_wgs84,
        admin_level=2,
        utm_epsg=32736,
    )

    assert im.get_utm_epsg() == 32736


# ==========================================================
# CRS REPROJECTION
# ==========================================================

def test_water_reprojection_to_match_admin(admin_gdf_wgs84):
    """Test that water_gdf in different CRS is reprojected to match admin."""
    # Create water in a different CRS
    poly = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
    water_projected = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:3857")
    
    im = InputsManager(
        admin_gdf=admin_gdf_wgs84,
        admin_level=2,
        water_gdf=water_projected,
    )
    
    # Water should be reprojected to WGS84
    assert im.water_gdf.crs.to_epsg() == 4326


def test_roads_reprojection_to_match_admin(admin_gdf_wgs84):
    """Test that roads_gdf in different CRS is reprojected to match admin."""
    # Create roads in a different CRS
    line = LineString([(0, 5), (10, 5)])
    roads_projected = gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs="EPSG:3857")
    
    im = InputsManager(
        admin_gdf=admin_gdf_wgs84,
        admin_level=2,
        roads_gdf=roads_projected,
    )
    
    # Roads should be reprojected to WGS84
    assert im.roads_gdf.crs.to_epsg() == 4326
