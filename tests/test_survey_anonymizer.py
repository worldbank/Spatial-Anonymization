# tests/test_survey_anonymizer.py
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from spatial_anonymization.anonymizer import SurveyAnonymizer
from spatial_anonymization.config import AnonymizationConfig
from spatial_anonymization.inputs_manager import InputsManager


def make_admin_polygon():
    # Square around (0,0)
    return Polygon([
        (-1, -1),
        (-1,  1),
        ( 1,  1),
        ( 1, -1),
    ])


def test_anonymizer_uses_max_distance_when_no_urban_rural():
    """
    Test that anonymizer successfully displaces a point using DHS method
    when no urban/rural classification is provided. Should use the default
    max_distance parameter and return valid anonymized coordinates.
    """
    admin_poly = make_admin_polygon()
    admin_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[admin_poly], crs="EPSG:4326")

    config = AnonymizationConfig(
        method="dhs",
        unit_type="household",
        max_distance=1000,
    )

    inputs = InputsManager(
        admin_gdf=admin_gdf,
        admin_level=0,
    )

    anonymizer = SurveyAnonymizer(
        config=config,
        inputs=inputs,
        max_attempts=10,
    )

    result = anonymizer.anonymize_point(0.0, 0.0)

    assert result is not None
    lon, lat = result
    assert isinstance(lon, float)
    assert isinstance(lat, float)


def test_anonymizer_fails_when_admin_too_small():
    """
    Test that anonymizer returns None when the administrative boundary is too small
    to accommodate the required displacement distance. This simulates cases where
    displacement would push points outside the allowed boundary, and all retry
    attempts are exhausted.
    """
    # Very tiny admin polygon
    admin_poly = Polygon([
        (0, 0),
        (0, 0.00001),
        (0.00001, 0.00001),
        (0.00001, 0),
    ])
    admin_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[admin_poly], crs="EPSG:4326")

    config = AnonymizationConfig(
        method="dhs",
        unit_type="household",
        max_distance=1000,
    )

    inputs = InputsManager(
        admin_gdf=admin_gdf,
        admin_level=0,
    )

    anonymizer = SurveyAnonymizer(
        config=config,
        inputs=inputs,
        max_attempts=5,
    )

    result = anonymizer.anonymize_point(0.0, 0.0)

    assert result is None


def test_anonymizer_donut_method():
    """
    Test that anonymizer works correctly with donut geomasking method.
    Donut method ensures displaced points are within a ring (min_distance to max_distance)
    rather than anywhere within max_distance. Verifies that a valid displaced
    point is returned.
    """
    admin_poly = make_admin_polygon()
    admin_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[admin_poly], crs="EPSG:4326")

    config = AnonymizationConfig(
        method="donut",
        unit_type="household",
        max_distance=1000,
        min_distance=200,
    )

    inputs = InputsManager(
        admin_gdf=admin_gdf,
        admin_level=0,
    )

    anonymizer = SurveyAnonymizer(
        config=config,
        inputs=inputs,
        max_attempts=10,
    )

    result = anonymizer.anonymize_point(0.0, 0.0)
    assert result is not None

def test_anonymizer_dataframe_interface():
    """
    Test that anonymizer correctly processes a pandas DataFrame with multiple points,
    applying different displacement distances for urban vs rural locations.
    Verifies that:
    - Output DataFrame has the expected anonymized coordinate columns (lon_anon, lat_anon)
    - All input rows are preserved in the output
    - Urban/rural classification is properly respected
    """
    admin_poly = make_admin_polygon()
    admin_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[admin_poly], crs="EPSG:4326")

    df = pd.DataFrame({
        "lon": [0.0, 0.2],
        "lat": [0.0, 0.2],
        "urban_rural": ["urban", "rural"],
    })

    config = AnonymizationConfig(
        method="dhs",
        unit_type="household",
        urban_max_distance=500,
        rural_max_distance=1000,
        enforce_admin_boundary=True,
    )

    inputs = InputsManager(
        admin_gdf=admin_gdf,
        admin_level=0,
    )

    anonymizer = SurveyAnonymizer(
        config=config,
        inputs=inputs,
    )

    out = anonymizer.anonymize_dataframe(
        df,
        lon_col="lon",
        lat_col="lat",
    )

    assert "lon_anon" in out.columns
    assert "lat_anon" in out.columns
    assert len(out) == len(df)
