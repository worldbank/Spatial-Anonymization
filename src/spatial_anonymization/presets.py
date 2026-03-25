"""
Predefined anonymization policy presets.

Presets provide named configurations for common survey anonymization
workflows, such as DHS-style displacement and World Bank LSMS standards.

Available presets:
    - DHS: Standard DHS displacement (2km urban, 5km rural, 1% at 10km)
    - WB_LSMS: World Bank LSMS survey standards with population thresholds

Usage:
    ```python
    from spatial_anonymization.presets import DHS, WB_LSMS, get_preset, list_presets
    
    # Use preset directly
    df_anon = anonymize_survey_data_from_config(df, config=DHS, country_iso3="MWI")
    
    # Customize a preset
    from dataclasses import replace
    custom_config = replace(DHS, max_distance=3000, lat_col="latitude")
    
    # Discover available presets
    list_presets()
    
    # Get preset by name
    config = get_preset("dhs")
    ```
"""

from typing import Dict, Optional
from spatial_anonymization.config import AnonymizationConfig


# =========================================================================
# PRESET DEFINITIONS
# =========================================================================

DHS = AnonymizationConfig(
    unit_type="cluster",
    method="dhs",
    
    # Distance policy - Standard DHS approach
    max_distance=5000,                      # Fallback
    urban_max_distance=2000,                # 2km for urban clusters
    rural_max_distance=5000,                # 5km for rural clusters
    rural_extra_max_distance=10000,         # 10km for subset of rural
    rural_extra_proportion=0.01,            # 1% of rural clusters
    
    # Donut (not used in standard DHS)
    min_distance=0.0,
    
    # Population policy (optional for DHS)
    min_population=None,
    population_search_max_radius=None,
    population_step_m=500,
    
    # Constraints
    enforce_admin_boundary=True,
    enforce_water_constraint=True,
    water_constraint_strict=True,
    enforce_road_constraint=False,
    
    # Input schema (user can override)
    lat_col="lat",
    lon_col="lon",
    urban_rural_col="urban_rural",
    urban_value="urban",
    rural_value="rural",
)
"""
DHS Standard Displacement Preset.

Implements the standard DHS (Demographic and Health Surveys) displacement
methodology used for protecting respondent privacy while maintaining
geographic utility for spatial analysis.

Key Features:
    - Urban areas: Random displacement within 2km radius
    - Rural areas: Random displacement within 5km radius  
    - Rural subset (1%): Displaced up to 10km for enhanced privacy
    - Enforces administrative boundary constraints
    - Avoids water bodies when displacing points
    
When to Use:
    - DHS-style household or health surveys
    - When compliance with DHS data dissemination standards is required
    - Cluster-level or household-level GPS anonymization
    - Multi-country surveys requiring standardized approach
    
Customization Examples:
    ```python
    from dataclasses import replace
    
    # Increase urban displacement to 3km
    custom = replace(DHS, urban_max_distance=3000)
    
    # Change column names to match your data
    custom = replace(DHS, lat_col="latitude", lon_col="longitude")
    
    # Use for household-level data
    custom = replace(DHS, unit_type="household")
    ```
"""


WB_LSMS = AnonymizationConfig(
    unit_type="cluster",
    method="dhs",
    
    # Distance policy - World Bank LSMS approach
    max_distance=5000,                      # Fallback
    urban_max_distance=2000,                # 2km for urban
    rural_max_distance=5000,                # 5km for rural
    rural_extra_max_distance=10000,         # 10km for subset of rural
    rural_extra_proportion=0.1,             # 10% of rural clusters
    
    # Donut (optional minimum distance)
    min_distance=0.0,
    
    # Population policy - ensure minimum population in anonymization envelope
    min_population=500,                     # Minimum 500 people
    population_search_max_radius=15000,     # Search up to 15km if needed
    population_step_m=500,                  # Expand in 500m increments
    
    # Constraints
    enforce_admin_boundary=True,
    enforce_water_constraint=True,
    water_constraint_strict=True,
    enforce_road_constraint=False,
    
    # Input schema (user can override)
    lat_col="lat",
    lon_col="lon",
    urban_rural_col="urban_rural",
    urban_value="urban",
    rural_value="rural",
)
"""
World Bank LSMS Preset.

Implements World Bank Living Standards Measurement Study (LSMS) displacement
standards with population-based privacy protections.

Key Features:
    - Urban areas: Random displacement within 2km radius
    - Rural areas: Random displacement within 5km radius
    - Rural subset (10%): Displaced up to 10km for enhanced privacy
    - Population threshold: Ensures ≥500 people within displacement radius
    - Enforces administrative boundary constraints
    - Avoids water bodies when displacing points

When to Use:
    - World Bank LSMS surveys (IHS, LSMS-ISA, etc.)
    - When population-based privacy guarantees are required
    - Multi-sector household surveys
    - Surveys integrating with geospatial poverty mapping

Customization Examples:
    ```python
    from dataclasses import replace
    
    # Increase minimum population threshold
    custom = replace(WB_LSMS, min_population=1000)
    
    # Reduce rural extra proportion to 5%
    custom = replace(WB_LSMS, rural_extra_proportion=0.05)
    
    # Use for household-level data
    custom = replace(WB_LSMS, unit_type="household")
    ```
"""


# Maintain backward compatibility
LSMS_CLUSTER = WB_LSMS
"""Alias for WB_LSMS preset (backward compatibility)."""


# =========================================================================
# PRESET REGISTRY & HELPER FUNCTIONS
# =========================================================================

PRESETS: Dict[str, AnonymizationConfig] = {
    "dhs": DHS,
    "wb_lsms": WB_LSMS,
    "lsms": WB_LSMS,  # Alias
    "lsms_cluster": WB_LSMS,  # Backward compatibility
}
"""
Registry of all available presets.

Keys are lowercase preset names for case-insensitive lookup.
"""


def get_preset(name: str) -> AnonymizationConfig:
    """
    Retrieve a preset configuration by name.
    
    Args:
        name: Preset name (case-insensitive). Available: 'dhs', 'wb_lsms', 'lsms'
        
    Returns:
        AnonymizationConfig object for the requested preset
        
    Raises:
        ValueError: If preset name is not recognized
        
    Examples:
        >>> config = get_preset("dhs")
        >>> config = get_preset("WB_LSMS")  # Case-insensitive
        
        >>> # Customize the preset
        >>> from dataclasses import replace
        >>> custom = replace(get_preset("dhs"), max_distance=3000)
    """
    name_lower = name.lower()
    
    if name_lower not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(
            f"Unknown preset: '{name}'. "
            f"Available presets: {available}"
        )
    
    return PRESETS[name_lower]


def list_presets() -> None:
    """
    Display all available presets with descriptions.
    
    Prints a formatted table showing preset names, methods, and key parameters
    to help users choose the appropriate preset for their use case.
    
    Examples:
        >>> from spatial_anonymization.presets import list_presets
        >>> list_presets()
        Available Anonymization Presets:
        ================================
        
        DHS (dhs)
          Method: dhs
          Urban max: 2000m | Rural max: 5000m
          Rural extra: 10000m (1.0% of rural)
          Population threshold: None
          Use case: Standard DHS displacement
        
        WB_LSMS (wb_lsms, lsms)
          Method: dhs  
          Urban max: 2000m | Rural max: 5000m
          Rural extra: 10000m (10.0% of rural)
          Population threshold: 500 people
          Use case: World Bank LSMS surveys
    """
    print("\nAvailable Anonymization Presets:")
    print("=" * 70)
    
    # Define main presets (exclude aliases)
    main_presets = {
        "dhs": ("DHS", "Standard DHS displacement"),
        "wb_lsms": ("WB_LSMS", "World Bank LSMS surveys"),
    }
    
    for key, (display_name, use_case) in main_presets.items():
        config = PRESETS[key]
        
        # Get aliases
        aliases = [k for k, v in PRESETS.items() if v is config and k != key]
        alias_str = f", {', '.join(aliases)}" if aliases else ""
        
        print(f"\n{display_name} ({key}{alias_str})")
        print(f"  Method: {config.method}")
        print(f"  Urban max: {config.urban_max_distance}m | Rural max: {config.rural_max_distance}m")
        
        if config.rural_extra_max_distance:
            pct = config.rural_extra_proportion * 100
            print(f"  Rural extra: {config.rural_extra_max_distance}m ({pct}% of rural)")
        
        if config.min_population:
            print(f"  Population threshold: {config.min_population} people")
        else:
            print(f"  Population threshold: None")
            
        print(f"  Use case: {use_case}")
    
    print("\n" + "=" * 70)
    print("\nUsage:")
    print("  from spatial_anonymization.presets import DHS, WB_LSMS, get_preset")
    print("  config = DHS  # or get_preset('dhs')")
    print("\nCustomize:")
    print("  from dataclasses import replace")
    print("  custom = replace(DHS, max_distance=3000)")
    print()
