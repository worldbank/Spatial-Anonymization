"""
Integration tests for download functionality.

These tests require internet connection and download real data.
Run with: pytest -m download
Skip with: pytest -m "not download"
"""

import pytest
from pathlib import Path
import tempfile

from spatial_anonymization.data.inputs_manager import InputsManager


# Path to download-test directory
DOWNLOAD_DIR = Path(__file__).parent.parent / "data" / "download-test"


@pytest.fixture
def download_dir():
    """Fixture providing the download directory."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return str(DOWNLOAD_DIR)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.download
def test_admin_boundaries_geoboundaries(download_dir):
    """Test downloading admin boundaries from geoBoundaries."""
    # Test with Malawi (small country, fast download)
    inputs = InputsManager(
        country_iso3="MWI",
        admin_level=2,
        auto_download=True,
        download_dir=download_dir,
    )
    
    admin_gdf = inputs.get_admin_gdf()
    
    # Assertions
    assert admin_gdf is not None, "Admin GeoDataFrame should not be None"
    assert len(admin_gdf) > 0, "Should have at least one admin boundary"
    assert admin_gdf.crs.to_epsg() == 4326, "Should be in WGS84"
    assert inputs.has_admin(), "has_admin() should return True"
    
    # Check we got district-level data
    assert len(admin_gdf) > 10, "Malawi should have multiple districts"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.download
def test_admin_boundaries_different_levels(download_dir):
    """Test downloading different admin levels."""
    # Test country level (ADM0)
    inputs_adm0 = InputsManager(
        country_iso3="MWI",
        admin_level=0,
        auto_download=True,
        download_dir=download_dir,
    )
    
    admin_gdf_adm0 = inputs_adm0.get_admin_gdf()
    assert len(admin_gdf_adm0) == 1, "Country level should have 1 feature"
    
    # Test region level (ADM1)
    inputs_adm1 = InputsManager(
        country_iso3="MWI",
        admin_level=1,
        auto_download=True,
        download_dir=download_dir,
    )
    
    admin_gdf_adm1 = inputs_adm1.get_admin_gdf()
    assert len(admin_gdf_adm1) > 1, "Region level should have multiple features"
    assert len(admin_gdf_adm1) > len(admin_gdf_adm0), "Should have more regions than country level"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.download
def test_population_raster_download(download_dir):
    """Test downloading WorldPop population raster."""
    inputs = InputsManager(
        country_iso3="MWI",
        admin_level=0,
        auto_download=True,
        download_dir=download_dir,
    )
    
    # Download population raster
    raster_path = inputs.download_population_raster(year=2025)
    
    # Assertions
    assert raster_path is not None, "Raster path should not be None"
    assert Path(raster_path).exists(), "Downloaded raster file should exist"
    assert Path(raster_path).stat().st_size > 0, "Raster file should not be empty"
    assert inputs.has_population(), "has_population() should return True"
    
    # Verify we can open it
    dataset = inputs.get_population_dataset()
    assert dataset is not None, "Should be able to open raster"
    assert dataset.width > 0 and dataset.height > 0, "Raster should have dimensions"
    dataset.close()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.download
def test_water_bodies_download(download_dir):
    """Test downloading water bodies from OpenStreetMap."""
    # First get admin boundaries
    inputs = InputsManager(
        country_iso3="MWI",
        admin_level=0,  # Country level for faster download
        auto_download=True,
        download_dir=download_dir,
    )
    
    # Download water bodies
    water_gdf = inputs.download_water_bodies()
    
    # Assertions (may be None if no water bodies found)
    if water_gdf is not None:
        assert len(water_gdf) > 0, "Should have at least one water feature"
        assert water_gdf.crs.to_epsg() == 4326, "Should be in WGS84"
        assert inputs.has_water(), "has_water() should return True"
        
        # Check geometry types are valid
        valid_types = {"Polygon", "LineString", "MultiPolygon", "MultiLineString"}
        assert all(geom_type in valid_types for geom_type in water_gdf.geometry.type.unique()), \
            "All geometries should be valid types"
    else:
        # It's OK if no water bodies found, but check the flag
        assert not inputs.has_water(), "has_water() should return False if no water"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.download
def test_combined_workflow(download_dir):
    """Test complete workflow with multiple downloads."""
    # Create InputsManager with country code only
    inputs = InputsManager(
        country_iso3="MWI",
        admin_level=2,
        auto_download=True,
        download_dir=download_dir,
    )
    
    # Check admin boundaries were auto-downloaded
    assert inputs.has_admin(), "Admin boundaries should be auto-downloaded"
    admin_gdf = inputs.get_admin_gdf()
    assert len(admin_gdf) > 0, "Should have admin features"
    
    # Download population raster
    raster_path = inputs.download_population_raster(year=2025)
    assert inputs.has_population(), "Population should be available after download"
    assert Path(raster_path).exists(), "Population raster file should exist"
    
    # Download water bodies (may fail due to API timeouts/rate limits)
    water_gdf = inputs.download_water_bodies()
    # Water may or may not be available depending on OSM data and API availability
    
    # Verify UTM EPSG can be inferred
    utm_epsg = inputs.get_utm_epsg()
    assert utm_epsg is not None, "Should be able to infer UTM EPSG"
    assert 32600 <= utm_epsg <= 32760, "UTM EPSG should be in valid range"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.download
def test_download_with_user_provided_admin(download_dir):
    """Test that download doesn't happen when admin_gdf is provided."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    # Create a dummy admin boundary
    poly = Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    admin_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    
    # Create InputsManager with both country_iso3 and admin_gdf
    # Should use admin_gdf and not download
    inputs = InputsManager(
        country_iso3="MWI",
        admin_gdf=admin_gdf,
        admin_level=2,
        auto_download=True,
        download_dir=download_dir,
    )
    
    # Should use the provided admin_gdf
    result_gdf = inputs.get_admin_gdf()
    assert len(result_gdf) == 1, "Should use provided admin_gdf with 1 feature"
    assert result_gdf.geometry[0].equals(poly), "Should be the same geometry we provided"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.download
def test_download_error_handling():
    """Test error handling for invalid country codes."""
    with pytest.raises(Exception):
        # Invalid country code should raise an error
        inputs = InputsManager(
            country_iso3="INVALID",
            admin_level=2,
            auto_download=True,
        )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.download
def test_auto_download_false_without_admin_raises():
    """Test that auto_download=False without admin_gdf raises error."""
    with pytest.raises(ValueError, match="auto_download is False"):
        inputs = InputsManager(
            country_iso3="MWI",
            admin_level=2,
            auto_download=False,
        )
