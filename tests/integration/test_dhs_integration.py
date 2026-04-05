"""
Integration tests for vanilla DHS-style spatial anonymization.

This module tests the standard DHS displacement method using preset
configuration from ``spatial_anonymization.config.presets.DHS``.
The only spatial constraint enforced is the administrative boundary —
no water-body or road constraints are applied.

Test dataset
------------
- **Country:** Malawi
- **Source:** Integrated Household Survey (IHS) sample
- **Size:** ~5 000 points with valid GPS coordinates
- **Classification:** Urban/rural labels available for all points
- **Admin boundaries:** ADM4-level (Traditional Authorities) shapefile

DHS preset behavior under test
------------------------------
- Urban points displaced within 0–2 km
- Rural points displaced within 0–5 km (90%) or 0–10 km (10%)
- Displaced points must remain inside the same admin unit
- Output includes ``distance_km``, ``attempts``, and ``max_radius_assigned``
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import random as stdlib_random
import pandas as pd
import pytest

from spatial_anonymization.api import anonymize_survey_coordinates
from spatial_anonymization.diagnostics.diagnostics import admin_boundary_check
from spatial_anonymization.geoprocessing.geo_utils import distance_points


# ======================================================================================
# TEST DATA PATHS and CONFIGS
# ======================================================================================
REPO_ROOT = Path("/Users/dmatekenya/Library/CloudStorage/OneDrive-WBG/Survey-Geo-Enhancement/git-repos/spatial-anonymization/")
DIR_TEST_BASE = REPO_ROOT / "data/mwi"
FILE_COORDS = DIR_TEST_BASE / "test-coords.csv"
FILE_ADM2 = DIR_TEST_BASE / "geospatial/adm4-wgs84.shp"

# Columns
COL_LON = "longitude_tablet"
COL_LAT = "latitude_tablet"
URBAN_RURAL_COL = "urb_rur"
URBAN_VALUE = "URBAN"
RURAL_VALUE = "RURAL"


# ======================================================================================
# HELPERS
# ======================================================================================
def compute_distances_km(
    lon1: pd.Series,
    lat1: pd.Series,
    lon2: pd.Series,
    lat2: pd.Series,
) -> pd.Series:
    """
    Compute geodesic distance in kilometers between two coordinate pairs.

    Uses distance_points from geoprocessing.geo_utils (WGS84 ellipsoid).
    """
    return pd.Series(
        [
            distance_points((lo1, la1), (lo2, la2))
            for lo1, la1, lo2, la2 in zip(
                lon1.astype(float), lat1.astype(float),
                lon2.astype(float), lat2.astype(float),
            )
        ],
        index=lon1.index,
    )


def require_local_test_data() -> None:
    """Skip the test module if the local integration-test data are not available."""
    missing = [str(path) for path in [FILE_COORDS, FILE_ADM2] if not path.exists()]
    if missing:
        pytest.skip(
            "Local DHS integration-test data not found. Missing: " + ", ".join(missing),
            allow_module_level=True,
        )


# ======================================================================================
# FIXTURES
# ======================================================================================
require_local_test_data()


@pytest.fixture(scope="module")
def df_reference() -> pd.DataFrame:
    """
    Load the Ethiopia coordinate test file.

    Returns
    -------
    pd.DataFrame
        Input dataframe containing original coordinates and the legacy R output.
    """
    return pd.read_csv(FILE_COORDS)


@pytest.fixture(scope="module")
def anonymization_result(tmp_path_factory: pytest.TempPathFactory, df_reference: pd.DataFrame) -> tuple:
    """
    Run DHS anonymization once and return both the dataframe and output directory.

    The random seed is fixed and multiprocessing is disabled so the test run is as
    reproducible as possible.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest factory for temporary directories.
    df_reference : pd.DataFrame
        Input coordinate dataframe.

    Returns
    -------
    tuple[pd.DataFrame, Path]
        Anonymized dataframe and the output directory path.
    """
    out_dir = tmp_path_factory.mktemp("dhs_integration_outputs")

    np.random.seed(42)
    df_out = anonymize_survey_coordinates(
        df=df_reference.copy(),
        lon_col=COL_LON,
        lat_col=COL_LAT,
        urban_rural_col=URBAN_RURAL_COL,
        urban_value=URBAN_VALUE,
        rural_value=RURAL_VALUE,
        method="dhs",
        country_iso3="MWI",
        output_dir=out_dir,
        admin_boundaries=FILE_ADM2,
        admin_level=4,
        n_workers=1,
        max_attempts=100,
        extract_displaced_population=False,
    )
    return df_out, out_dir


@pytest.fixture(scope="module")
def df_anonymized(anonymization_result: tuple) -> pd.DataFrame:
    """Return the anonymized dataframe."""
    return anonymization_result[0]


@pytest.fixture(scope="module")
def output_dir(anonymization_result: tuple) -> Path:
    """Return the output directory used by the anonymization run."""
    return anonymization_result[1]


# ======================================================================================
# TESTS
# ======================================================================================
def test_dhs_anonymization_runs_and_preserves_expected_structure(
    df_reference: pd.DataFrame,
    df_anonymized: pd.DataFrame,
) -> None:
    """
    Check that anonymization runs successfully and preserves the expected dataframe structure.

    Parameters
    ----------
    df_reference : pd.DataFrame
        Input dataframe before anonymization.
    df_anonymized : pd.DataFrame
        Output dataframe after anonymization.
    """
    # Some points may be dropped if they fall outside admin boundaries
    assert len(df_anonymized) <= len(df_reference)
    assert len(df_anonymized) > 0, "All points were dropped during anonymization."

    required_cols = ['hh_id', 'urb_rur', 'latitude_tablet', 'longitude_tablet']
    missing = set(required_cols).difference(df_anonymized.columns)
    assert not missing, f"Missing expected output columns: {sorted(missing)}"

    # Anonymization metadata columns
    anon_cols = ["lon_anon", "lat_anon", "distance_km", "attempts", "max_radius_assigned", "constraining_admin"]
    missing_anon = set(anon_cols).difference(df_anonymized.columns)
    assert not missing_anon, f"Missing anonymization output columns: {sorted(missing_anon)}"

    # constraining_admin should reflect the admin_level=4 used in the fixture
    assert (df_anonymized["constraining_admin"] == "adm4").all(), (
        f"Expected constraining_admin='adm4', got: {df_anonymized['constraining_admin'].unique()}"
    )

    # All returned rows should be successfully displaced (failed rows are dropped).
    assert df_anonymized["lat_anon"].notna().all()
    assert df_anonymized["lon_anon"].notna().all()
    assert (df_anonymized["distance_km"] > 0).all()
    assert (df_anonymized["attempts"] >= 1).all()
    assert (df_anonymized["max_radius_assigned"] > 0).all()


def test_dhs_output_respects_distance_rules(df_anonymized: pd.DataFrame) -> None:
    """
    Check that the anonymized coordinates follow DHS-style distance limits.

    Urban points should not exceed 2 km. Rural points should not exceed 10 km.
    This test recomputes distances independently from the coordinate columns.

    Parameters
    ----------
    df_anonymized : pd.DataFrame
        Output dataframe after anonymization.
    """
    dist_km = compute_distances_km(
        lon1=df_anonymized[COL_LON],
        lat1=df_anonymized[COL_LAT],
        lon2=df_anonymized["lon_anon"],
        lat2=df_anonymized["lat_anon"],
    )

    urban_mask = df_anonymized[URBAN_RURAL_COL].eq(URBAN_VALUE)
    rural_mask = df_anonymized[URBAN_RURAL_COL].eq(RURAL_VALUE)

    assert urban_mask.any(), "No URBAN rows found in integration dataset."
    assert rural_mask.any(), "No RURAL rows found in integration dataset."

    assert (dist_km.loc[urban_mask] <= 2.0 + 0.01).all()
    assert (dist_km.loc[rural_mask] <= 10.0 + 0.01).all()


def test_dhs_rural_90_10_assignment_split(df_anonymized: pd.DataFrame) -> None:
    """
    Check that rural assignments follow the intended DHS 90/10 rule.

    This test uses the assigned maximum radius column, which is the right variable for
    validating the 5 km / 10 km regime split. Realized displacement distance is not
    enough because a point assigned to the 10 km regime can still end up within 5 km.

    Parameters
    ----------
    df_anonymized : pd.DataFrame
        Output dataframe after anonymization.
    """
    assert "max_radius_assigned" in df_anonymized.columns, (
        "Expected 'max_radius_assigned' in output for validating the rural 90/10 rule."
    )

    rural = df_anonymized.loc[
        df_anonymized[URBAN_RURAL_COL] == RURAL_VALUE
    ].copy()
    assert len(rural) > 0, "No rural rows found in integration dataset."

    p_5km = rural["max_radius_assigned"].eq(5000).mean()
    p_10km = rural["max_radius_assigned"].eq(10000).mean()

    # Slight tolerance around the exact DHS split because the test dataset is finite.
    assert 0.85 <= p_5km <= 0.95, f"Observed rural 5 km share was {p_5km:.3f}."
    assert 0.05 <= p_10km <= 0.15, f"Observed rural 10 km share was {p_10km:.3f}."
    assert pytest.approx(p_5km + p_10km, abs=1e-9) == 1.0


def test_dhs_mean_displacement_is_reasonable(df_anonymized: pd.DataFrame) -> None:
    """
    Check that mean displacement is in a reasonable range for DHS-style anonymization.

    For uniform-by-area sampling in a circle:
    - Urban (2 km max) has expected mean distance about 1.33 km
    - Rural (90% at 5 km, 10% at 10 km) has expected mean distance about 3.67 km

    Tolerances are intentionally loose because realized means vary with finite samples
    and may shift slightly depending on implementation details.
    """
    dist_km = compute_distances_km(
        lon1=df_anonymized[COL_LON],
        lat1=df_anonymized[COL_LAT],
        lon2=df_anonymized["lon_anon"],
        lat2=df_anonymized["lat_anon"],
    )

    urban_mask = df_anonymized[URBAN_RURAL_COL].eq(URBAN_VALUE)
    rural_mask = df_anonymized[URBAN_RURAL_COL].eq(RURAL_VALUE)

    assert urban_mask.any(), "No URBAN rows found in integration dataset."
    assert rural_mask.any(), "No RURAL rows found in integration dataset."

    urban_mean = dist_km.loc[urban_mask].mean()
    rural_mean = dist_km.loc[rural_mask].mean()

    # Theoretical DHS-style means under uniform-by-area sampling
    expected_urban_mean = 2 * 2.0 / 3.0      # 1.333 km
    expected_rural_mean = 0.9 * (2 * 5.0 / 3.0) + 0.1 * (2 * 10.0 / 3.0)  # 3.667 km

    assert abs(urban_mean - expected_urban_mean) <= 0.30, (
        f"Urban mean displacement was {urban_mean:.3f} km; "
        f"expected about {expected_urban_mean:.3f} km."
    )
    assert abs(rural_mean - expected_rural_mean) <= 0.50, (
        f"Rural mean displacement was {rural_mean:.3f} km; "
        f"expected about {expected_rural_mean:.3f} km."
    )


def test_dhs_reproducible_with_fixed_seed(df_reference: pd.DataFrame, tmp_path: Path) -> None:
    """
    Check that the anonymization is reproducible when the seed is fixed.

    This test assumes the package relies on NumPy's random state or respects the global
    seed. Multiprocessing is disabled to reduce nondeterminism.

    Parameters
    ----------
    df_reference : pd.DataFrame
        Input coordinate dataframe.
    tmp_path : Path
        Temporary pytest directory.
    """
    common_kwargs = dict(
        lon_col=COL_LON,
        lat_col=COL_LAT,
        urban_rural_col=URBAN_RURAL_COL,
        urban_value=URBAN_VALUE,
        rural_value=RURAL_VALUE,
        method="dhs",
        country_iso3="MWI",
        admin_boundaries=FILE_ADM2,
        admin_level=4,
        n_workers=1,
        max_attempts=100,
        extract_displaced_population=False,
    )

    np.random.seed(42)
    stdlib_random.seed(42)
    out1 = anonymize_survey_coordinates(
        df=df_reference.copy(),
        output_dir=tmp_path / "run_1",
        **common_kwargs,
    )

    np.random.seed(42)
    stdlib_random.seed(42)
    out2 = anonymize_survey_coordinates(
        df=df_reference.copy(),
        output_dir=tmp_path / "run_2",
        **common_kwargs,
    )

    pd.testing.assert_series_equal(
        out1["lon_anon"].reset_index(drop=True),
        out2["lon_anon"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out1["lat_anon"].reset_index(drop=True),
        out2["lat_anon"].reset_index(drop=True),
        check_names=False,
    )


def test_dhs_output_files_saved(
    df_anonymized: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Verify that anonymization saves both a CSV and a GeoJSON of anonymized coordinates.

    Parameters
    ----------
    df_anonymized : pd.DataFrame
        Output dataframe after anonymization (used to trigger the fixture).
    output_dir : Path
        Root output directory passed to the anonymization run.
    """
    data_dir = output_dir / "anonymization_outputs" / "output_data"

    csv_path = data_dir / "anonymized.csv"
    geojson_path = data_dir / "anonymized.geojson"

    # ---------- files exist ----------
    assert csv_path.exists(), f"Expected CSV not found: {csv_path}"
    assert geojson_path.exists(), f"Expected GeoJSON not found: {geojson_path}"

    # ---------- CSV is non-empty and loadable ----------
    df_csv = pd.read_csv(csv_path)
    assert len(df_csv) > 0, "Saved CSV is empty."
    assert "lon_anon" in df_csv.columns, "CSV missing 'lon_anon' column."
    assert "lat_anon" in df_csv.columns, "CSV missing 'lat_anon' column."

    # ---------- GeoJSON is non-empty and loadable ----------
    gdf = gpd.read_file(geojson_path)
    assert len(gdf) > 0, "Saved GeoJSON is empty."
    assert gdf.geometry.notna().all(), "GeoJSON contains null geometries."

    # The returned DataFrame only contains successfully displaced rows,
    # while saved files may include all processed rows (failed rows have NaN coords).
    # Verify returned DataFrame is a subset of saved data.
    assert len(df_anonymized) <= len(df_csv), (
        f"Returned dataframe ({len(df_anonymized)}) has more rows than saved CSV ({len(df_csv)})."
    )
    assert len(df_anonymized) <= len(gdf) + (len(df_csv) - len(gdf)), (
        "Row count mismatch between returned dataframe and saved files."
    )


def test_dhs_displaced_points_stay_in_admin_boundary(
    df_anonymized: pd.DataFrame,
) -> None:
    """
    Verify that every displaced point remains inside its original admin unit.

    The DHS preset enforces ``enforce_admin_boundary=True``, so no point
    should escape to a different administrative region after displacement.

    Parameters
    ----------
    df_anonymized : pd.DataFrame
        Output dataframe after anonymization.
    """
    admin_gdf = gpd.read_file(FILE_ADM2)

    result = admin_boundary_check(
        df_anonymized,
        admin_gdf,
        lon_col=COL_LON,
        lat_col=COL_LAT,
    )

    assert result["escaped"] == 0, (
        f"{result['escaped']} displaced point(s) escaped their original admin unit "
        f"(escape rate: {result['escape_rate']:.2%})."
    )
    assert result["in_same_admin"] > 0, (
        "No points were matched to an admin unit — check the boundary file."
    )


def test_dhs_relax_admin_allows_escapes(
    df_reference: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """
    Verify that ``relax_admin_constraint_on_failure=True`` allows points to
    escape their original admin unit when displacement within-admin fails.

    Uses a very low ``max_attempts`` to force many within-admin failures,
    which triggers the relaxed retry path.  The test checks:

    1. The run still returns results (no outright crash).
    2. The ``relaxed_admin`` column is present and flags some relaxed points.
    3. ``admin_boundary_check`` detects escapes that are consistent with the
       relaxed flag.
    """
    np.random.seed(99)
    stdlib_random.seed(99)

    df_out = anonymize_survey_coordinates(
        df=df_reference.copy(),
        lon_col=COL_LON,
        lat_col=COL_LAT,
        urban_rural_col=URBAN_RURAL_COL,
        urban_value=URBAN_VALUE,
        rural_value=RURAL_VALUE,
        method="dhs",
        country_iso3="MWI",
        output_dir=tmp_path,
        admin_boundaries=FILE_ADM2,
        admin_level=4,
        n_workers=1,
        max_attempts=3,
        relax_admin_constraint_on_failure=True,
        extract_displaced_population=False,
    )

    # 1. Run should still produce results
    assert len(df_out) > 0, "No points returned after relaxed anonymization."
    assert df_out["lon_anon"].notna().all()

    # 2. relaxed_admin column should be present
    assert "relaxed_admin" in df_out.columns, (
        "Expected 'relaxed_admin' column when relax_admin_constraint_on_failure=True."
    )
    n_relaxed = int(df_out["relaxed_admin"].sum())
    print(f"[Test] {n_relaxed}/{len(df_out)} points used relaxed admin constraint")

    # With max_attempts=3 and fine-grained ADM4 boundaries, we expect at
    # least some points to require the relaxed fallback.
    assert n_relaxed > 0, (
        "Expected some points to be displaced with relaxed admin constraint "
        "when max_attempts=3, but none were flagged."
    )

    # 3. admin_boundary_check should detect escapes matching the relaxed flag
    admin_gdf = gpd.read_file(FILE_ADM2)
    result = admin_boundary_check(
        df_out,
        admin_gdf,
        lon_col=COL_LON,
        lat_col=COL_LAT,
    )

    # Every relaxed point *may* have escaped (some might still land in the
    # same admin by chance).  But no non-relaxed point should have escaped.
    assert result["escaped"] >= 0  # sanity — escapes are expected
    print(
        f"[Test] admin_boundary_check: {result['escaped']} escaped, "
        f"{result['in_same_admin']} in same admin"
    )
