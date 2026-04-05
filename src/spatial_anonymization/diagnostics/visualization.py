"""
Visualization tools for spatial anonymization diagnostics.

This module provides functions for visualizing anonymization results,
comparing original and displaced coordinates, and assessing displacement
patterns.
"""

import pandas as pd
import geopandas as gpd
from typing import Optional, Union, Tuple
from pathlib import Path

# Default cap to avoid crashing notebook renderers with huge HTML.
_DEFAULT_MAX_POINTS = 500


def plot_displacement_comparison(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    lon_anon_col: str = "lon_anon",
    lat_anon_col: str = "lat_anon",
    *,
    admin_gdf: Optional[gpd.GeoDataFrame] = None,
    n_sample: Optional[int] = _DEFAULT_MAX_POINTS,
    n_buffers: int = 5,
    map_center: Optional[Tuple[float, float]] = None,
    zoom_start: int = 8,
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Create an interactive map comparing original and anonymized coordinates.

    Uses Folium to create an interactive web map showing:
    - Original coordinates (gray markers)
    - Anonymized coordinates (orange markers)
    - Displacement lines connecting original to anonymized points
    - Admin boundaries as thin gray outlines (if provided)
    - Displacement radius buffers for a small subset of points

    For large datasets the function automatically samples down to
    ``n_sample`` records (default 500) to keep the HTML size
    manageable.  Pass ``n_sample=None`` to override this and plot
    everything (not recommended for >1 000 points in a notebook).

    The map is always saved to *output_path* (default:
    ``displacement_map.html`` in the working directory) so it can
    be opened in a full browser for interactive exploration.

    Args:
        df (DataFrame):
            DataFrame containing original and anonymized coordinates.
        lon_col (str):
            Column name for original longitude. Default: "lon".
        lat_col (str):
            Column name for original latitude. Default: "lat".
        lon_anon_col (str):
            Column name for anonymized longitude. Default: "lon_anon".
        lat_anon_col (str):
            Column name for anonymized latitude. Default: "lat_anon".
        admin_gdf (GeoDataFrame, optional):
            Admin boundaries to overlay as thin gray outlines.
        n_sample (int, optional):
            Maximum number of records to plot.  Default: 500.
            Pass ``None`` to plot all records.
        n_buffers (int):
            Number of points for which to draw displacement radius
            buffers (illustrative).  Default: 5.  Set to 0 to disable.
            Requires ``max_radius_assigned`` column in the DataFrame.
        map_center (tuple, optional):
            (latitude, longitude) tuple for map center. If None, uses
            data centroid.
        zoom_start (int):
            Initial zoom level for the map. Default: 8.
        output_path (str or Path, optional):
            Save the map as an HTML file.  Default:
            ``displacement_map.html`` in the current directory.

    Returns:
        folium.Map: Interactive Folium map object.

    Raises:
        ImportError: If folium is not installed.
        ValueError: If required columns are missing from the dataframe.
    """
    try:
        import folium
    except ImportError:
        raise ImportError(
            "folium is required for visualization. "
            "Install it with: pip install folium"
        )

    from spatial_anonymization.geoprocessing.geo_utils import distance_points

    # Validate columns exist
    required_cols = [lon_col, lat_col, lon_anon_col, lat_anon_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Filter to valid records (non-null coordinates)
    df_plot = df.dropna(subset=[lon_col, lat_col, lon_anon_col, lat_anon_col]).copy()

    if len(df_plot) == 0:
        raise ValueError("No valid coordinate pairs found in the dataframe")

    n_total = len(df_plot)

    # Sample if needed
    if n_sample is not None and n_total > n_sample:
        df_plot = df_plot.sample(n=n_sample, random_state=42)
        print(
            f"[Visualization] Sampled {n_sample} of {n_total} records "
            f"(pass n_sample=None to plot all)"
        )

    # Determine map center
    if map_center is None:
        center_lat = df_plot[lat_col].mean()
        center_lon = df_plot[lon_col].mean()
        map_center = (center_lat, center_lon)

    # Create base map
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # ---- Admin boundaries ----
    if admin_gdf is not None:
        fg_admin = folium.FeatureGroup(name="Admin boundaries", show=True)
        admin_wgs = (
            admin_gdf.to_crs("EPSG:4326")
            if admin_gdf.crs is not None and admin_gdf.crs.to_epsg() != 4326
            else admin_gdf
        )
        folium.GeoJson(
            admin_wgs.__geo_interface__,
            style_function=lambda _: {
                "color": "#888888",
                "weight": 1,
                "fillOpacity": 0,
            },
        ).add_to(fg_admin)
        fg_admin.add_to(m)

    # ---- Feature groups for points and lines ----
    fg_orig = folium.FeatureGroup(name="Original points", show=True)
    fg_anon = folium.FeatureGroup(name="Displaced points", show=True)
    fg_lines = folium.FeatureGroup(name="Displacement lines", show=True)

    for _, row in df_plot.iterrows():
        lat_orig = row[lat_col]
        lon_orig = row[lon_col]
        lat_anon = row[lat_anon_col]
        lon_anon = row[lon_anon_col]

        dist_km = distance_points(
            (lon_orig, lat_orig), (lon_anon, lat_anon)
        )

        popup = (
            f"<b>{dist_km:.3f} km</b><br>"
            f"Orig: ({lat_orig:.5f}, {lon_orig:.5f})<br>"
            f"Anon: ({lat_anon:.5f}, {lon_anon:.5f})"
        )

        folium.CircleMarker(
            location=[lat_orig, lon_orig],
            radius=4,
            color="#888888",
            fill=True,
            fillOpacity=0.5,
            weight=1,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(fg_orig)

        folium.CircleMarker(
            location=[lat_anon, lon_anon],
            radius=4,
            color="#e67300",
            fill=True,
            fillOpacity=0.6,
            weight=1,
            popup=folium.Popup(popup, max_width=250),
        ).add_to(fg_anon)

        folium.PolyLine(
            locations=[[lat_orig, lon_orig], [lat_anon, lon_anon]],
            color="#aaaaaa",
            weight=1,
            opacity=0.4,
        ).add_to(fg_lines)

    fg_lines.add_to(m)
    fg_orig.add_to(m)
    fg_anon.add_to(m)

    # ---- Displacement radius buffers (illustrative subset) ----
    if n_buffers > 0 and "max_radius_assigned" in df_plot.columns:
        buf_df = df_plot.dropna(subset=["max_radius_assigned"])
        if len(buf_df) > n_buffers:
            buf_df = buf_df.sample(n=n_buffers, random_state=7)
        if len(buf_df) > 0:
            fg_buf = folium.FeatureGroup(name="Radius buffers", show=True)
            for _, row in buf_df.iterrows():
                radius_m = row["max_radius_assigned"]
                folium.Circle(
                    location=[row[lat_col], row[lon_col]],
                    radius=radius_m,
                    color="#888888",
                    weight=1,
                    fill=True,
                    fillColor="#888888",
                    fillOpacity=0.08,
                    popup=folium.Popup(
                        f"Max radius: {int(radius_m)} m", max_width=200
                    ),
                ).add_to(fg_buf)
            fg_buf.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = (
        '<div style="position:fixed; bottom:30px; right:30px; '
        "background:white; border:1px solid #888; padding:8px; "
        'font-size:13px; z-index:9999;">'
        "<b>Legend</b><br>"
        '<span style="color:#888888;">●</span> Original<br>'
        '<span style="color:#e67300;">●</span> Displaced<br>'
        '<span style="color:#aaaaaa;">—</span> Displacement line<br>'
        '<span style="color:#888888;">○</span> Max radius buffer<br>'
        f"<span style='color:gray;font-size:11px;'>n = {len(df_plot)}</span>"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    # Always save to file so user can open in a full browser
    if output_path is None:
        output_path = Path("displacement_map.html")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"[Visualization] Map saved to {output_path}")

    return m
