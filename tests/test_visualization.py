"""Tests for visualization helpers."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest


def create_synthetic_displacement_data(
    n_points: int = 1000,
    *,
    seed: int = 42,
    base_lat: float = -13.9833,
    base_lon: float = 33.7833,
    jitter_degrees: float = 0.5,
    displacement_min_degrees: float = 0.01,
    displacement_max_degrees: float = 0.05,
) -> pd.DataFrame:
    """Create a synthetic dataset with original + anonymized coordinates.

    Notes:
        - Coordinates are generated around (base_lat, base_lon).
        - Displacement is applied in a random direction with magnitude sampled
          uniformly in degrees.

    Returns:
        DataFrame with columns: lat, lon, lat_anon, lon_anon
    """
    if n_points < 1000:
        raise ValueError("n_points must be at least 1000")

    rng = np.random.default_rng(seed)

    lats_orig = base_lat + rng.uniform(-jitter_degrees, jitter_degrees, n_points)
    lons_orig = base_lon + rng.uniform(-jitter_degrees, jitter_degrees, n_points)

    displacement_dist = rng.uniform(displacement_min_degrees, displacement_max_degrees, n_points)
    displacement_angle = rng.uniform(0.0, 2.0 * np.pi, n_points)

    lats_anon = lats_orig + displacement_dist * np.sin(displacement_angle)
    lons_anon = lons_orig + displacement_dist * np.cos(displacement_angle)

    return pd.DataFrame(
        {
            "lat": lats_orig,
            "lon": lons_orig,
            "lat_anon": lats_anon,
            "lon_anon": lons_anon,
        }
    )


def test_plot_displacement_comparison_saves_html():
    """Generates 1000 synthetic points and saves the comparison map to /output."""
    if importlib.util.find_spec("folium") is None:
        pytest.skip("folium is not installed")

    from spatial_anonymization.diagnostics.visualization import plot_displacement_comparison

    df = create_synthetic_displacement_data(n_points=1000)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "displacement_comparison_test.html"

    plot_displacement_comparison(
        df,
        lon_col="lon",
        lat_col="lat",
        lon_anon_col="lon_anon",
        lat_anon_col="lat_anon",
        n_sample=None,
        map_center=None,
        zoom_start=8,
        output_path=output_path,
    )

    assert output_path.exists(), f"Expected HTML output at: {output_path}"
    assert output_path.stat().st_size > 0, "Output HTML file is empty"

    content = output_path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "leaflet" in content, "Output doesn't appear to be a Leaflet/Folium map"
