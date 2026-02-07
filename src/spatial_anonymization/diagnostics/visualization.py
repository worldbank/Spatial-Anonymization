"""
Visualization tools for spatial anonymization diagnostics.

This module provides functions for visualizing anonymization results,
comparing original and displaced coordinates, and assessing displacement
patterns.
"""

import pandas as pd
from typing import Optional, Union, Tuple
from pathlib import Path


def plot_displacement_comparison(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    lon_anon_col: str = "lon_anon",
    lat_anon_col: str = "lat_anon",
    *,
    n_sample: Optional[int] = None,
    map_center: Optional[Tuple[float, float]] = None,
    zoom_start: int = 8,
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Create an interactive map comparing original and anonymized coordinates.
    
    Uses Folium to create an interactive web map showing:
    - Original coordinates (blue markers)
    - Anonymized coordinates (red markers)
    - Displacement lines connecting original to anonymized points
    - Popups with distance information
    
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
        n_sample (int, optional):
            Number of random records to plot. If None, plots all records.
            Use this for large datasets to avoid performance issues.
        map_center (tuple, optional):
            (latitude, longitude) tuple for map center. If None, uses data centroid.
        zoom_start (int):
            Initial zoom level for the map. Default: 8.
        output_path (str or Path, optional):
            If provided, saves the map as an HTML file at this path.
    
    Returns:
        folium.Map: Interactive Folium map object. Can be displayed in Jupyter
                    notebooks or saved to HTML.
    
    Raises:
        ImportError: If folium is not installed.
        ValueError: If required columns are missing from the dataframe.
    
    Examples:
        Display in Jupyter notebook:
        >>> df_anon = anonymize_survey_coordinates(df, "MWI", method="dhs")
        >>> map_obj = plot_displacement_comparison(
        ...     df_anon,
        ...     lon_col="longitude_tablet",
        ...     lat_col="latitude_tablet",
        ...     n_sample=100
        ... )
        >>> map_obj  # Display in notebook
        
        Save to HTML file:
        >>> plot_displacement_comparison(
        ...     df_anon,
        ...     output_path="output/displacement_map.html",
        ...     n_sample=500
        ... )
    """
    try:
        import folium
        from folium import plugins
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
    
    # Sample if requested
    if n_sample is not None and len(df_plot) > n_sample:
        df_plot = df_plot.sample(n=n_sample, random_state=42)
        print(f"Plotting {n_sample} randomly sampled records out of {len(df)} total")
    
    # Determine map center
    if map_center is None:
        center_lat = df_plot[lat_col].mean()
        center_lon = df_plot[lon_col].mean()
        map_center = (center_lat, center_lon)
    
    # Create base map
    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Add alternative tile layers
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    # Plot each record
    for idx, row in df_plot.iterrows():
        lat_orig = row[lat_col]
        lon_orig = row[lon_col]
        lat_anon = row[lat_anon_col]
        lon_anon = row[lon_anon_col]
        
        # Calculate displacement distance
        distance_km = distance_points(
            (lon_orig, lat_orig),
            (lon_anon, lat_anon)
        )
        
        # Build popup text
        popup_text = f"<b>Displacement: {distance_km:.3f} km</b><br>"
        popup_text += f"Original: ({lat_orig:.6f}, {lon_orig:.6f})<br>"
        popup_text += f"Anonymized: ({lat_anon:.6f}, {lon_anon:.6f})"
        
        # Add original point marker (blue)
        folium.CircleMarker(
            location=[lat_orig, lon_orig],
            radius=5,
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.6,
            popup=folium.Popup(f"<b>Original</b><br>{popup_text}", max_width=300),
            tooltip=f"Original ({distance_km:.2f}km displaced)"
        ).add_to(m)
        
        # Add anonymized point marker (red)
        folium.CircleMarker(
            location=[lat_anon, lon_anon],
            radius=5,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.7,
            popup=folium.Popup(f"<b>Anonymized</b><br>{popup_text}", max_width=300),
            tooltip=f"Anonymized ({distance_km:.2f}km from original)"
        ).add_to(m)
        
        # Add displacement line
        folium.PolyLine(
            locations=[[lat_orig, lon_orig], [lat_anon, lon_anon]],
            color='gray',
            weight=1,
            opacity=0.5,
            popup=folium.Popup(f"Displacement: {distance_km:.3f} km", max_width=200)
        ).add_to(m)
    
    # Add minimap
    plugins.MiniMap(toggle_display=True).add_to(m)
    
    # Add fullscreen option
    plugins.Fullscreen(
        position='topright',
        title='Fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Add measure control
    plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        primary_area_unit='hectares'
    ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
        <p style="margin:0; font-weight:bold;">Legend</p>
        <p style="margin:5px 0;"><span style="color:blue;">●</span> Original Points</p>
        <p style="margin:5px 0;"><span style="color:red;">●</span> Anonymized Points</p>
        <p style="margin:5px 0;"><span style="color:gray;">—</span> Displacement</p>
        <p style="margin:5px 0; font-size:12px; color:gray;">
            n = {n_records} records
        </p>
    </div>
    '''.format(n_records=len(df_plot))
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        print(f"✓ Map saved to: {output_path}")
    
    return m
