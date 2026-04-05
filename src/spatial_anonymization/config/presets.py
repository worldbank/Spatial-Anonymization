"""
Predefined anonymization policy presets.

This module provides pre-configured AnonymizationParams objects for common
survey anonymization workflows. These presets follow established survey
standards and best practices.

Relationship between Params and Presets:
    - AnonymizationParams: A dataclass for building custom anonymization policies
    - Preset: A pre-configured AnonymizationParams following established standards
    
    Think of presets as completed configuration templates - you can use them
    directly or customize them by overriding specific parameters.

Available presets:
    - DHS: Standard DHS displacement (2km urban, 5km rural, 10% at 10km)

Usage:
    ```python
    from spatial_anonymization.config.presets import DHS, get_preset, list_presets
    from spatial_anonymization import anonymize_survey_coordinates
    
    # Use preset directly
    df_anon = anonymize_survey_coordinates(df, country_iso3="MWI", config=DHS)
    
    # Customize a preset
    from dataclasses import replace
    custom_config = replace(DHS, max_distance=3000)
    
    # Discover available presets
    list_presets()
    
    # Get preset by name
    config = get_preset("dhs")
    ```
"""

from dataclasses import fields, MISSING
from typing import Dict, Optional
from spatial_anonymization.config.params import AnonymizationParams


# =========================================================================
# PRESET DEFINITIONS
# =========================================================================

# ---------------------------------------------------------------------------
# DHS Standard Displacement Preset
# ---------------------------------------------------------------------------
# Implements the standard DHS (Demographic and Health Surveys) displacement
# methodology used for protecting respondent privacy while maintaining
# geographic utility for spatial analysis.
#
# Key Features:
#   - Urban areas: Random displacement within 2 km radius
#   - Rural areas: Random displacement within 5 km radius
#   - Rural subset (10%): Displaced up to 10 km for enhanced privacy
#   - Enforces administrative boundary constraints
#
# When to Use:
#   - DHS-style surveys requiring standardized displacement
#   - When compliance with DHS data dissemination standards is required
#   - Multi-country surveys requiring a standardized approach
#
# Customization:
#   from dataclasses import replace
#   custom = replace(DHS, urban_max_distance=3000)
# ---------------------------------------------------------------------------
DHS = AnonymizationParams(
    method="dhs",
    
    # Distance policy - Standard DHS approach
    max_distance=5000,                      # Fallback
    urban_max_distance=2000,                # 2km for urban clusters
    rural_max_distance=5000,                # 5km for rural clusters
    rural_extra_max_distance=10000,         # 10km for subset of rural
    rural_extra_proportion=0.1,            # 10% of rural clusters
    
    # Donut (not used in standard DHS)
    min_distance=0.0,
    
    # Population policy 
    displaced_pop_radius_m=1000,            # 1km buffer for post-displacement pop count
    
    # Constraints — DHS only enforces admin boundaries
    enforce_admin_boundary=True,
    enforce_water_constraint=False,
)


# =========================================================================
# PRESET REGISTRY & HELPER FUNCTIONS
# =========================================================================

PRESETS: Dict[str, AnonymizationParams] = {
    "dhs": DHS,
}
"""
Registry of all available presets.

Keys are lowercase preset names for case-insensitive lookup.
"""

# Fields explicitly configured in each preset (includes those matching defaults).
_PRESET_EXPLICIT_FIELDS: Dict[str, set[str]] = {
    "dhs": {
        "method",
        "max_distance",
        "urban_max_distance",
        "rural_max_distance",
        "rural_extra_max_distance",
        "rural_extra_proportion",
        "min_distance",
        "displaced_pop_radius_m",
        "enforce_admin_boundary",
    },
}


def get_preset(name: str) -> AnonymizationParams:
    """
    Retrieve a preset configuration by name.
    
    Args:
        name: Preset name (case-insensitive). Available: 'dhs'
        
    Returns:
        AnonymizationParams object for the requested preset
        
    Raises:
        ValueError: If preset name is not recognized
        
    Examples:
        >>> from spatial_anonymization.config.presets import get_preset
        >>> params = get_preset("dhs")
        
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
        >>> from spatial_anonymization.config.presets import list_presets
        >>> list_presets()
        Available Anonymization Presets:
        ================================
        
        DHS (dhs)
          Method: dhs
          Urban max: 2000m | Rural max: 5000m
          Rural extra: 10000m (10.0% of rural)
          Population threshold: None
          Use case: Standard DHS displacement
    """
    print("\nAvailable Anonymization Presets:")
    print("=" * 70)
    
    # Define main presets (exclude aliases)
    main_presets = {
        "dhs": ("DHS", "Standard DHS displacement"),
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
    print("  from spatial_anonymization.config.presets import DHS, get_preset")
    print("  params = DHS  # or get_preset('dhs')")
    print("\nCustomize:")
    print("  from dataclasses import replace")
    print("  custom = replace(DHS, max_distance=3000)")
    print()


def print_preset(preset: AnonymizationParams, name: Optional[str] = None) -> None:
    """
    Pretty-print all parameters of an anonymization preset.

    Each parameter is shown with its current value and the description from
    the field metadata defined in ``AnonymizationParams``.

    Args:
        preset: An ``AnonymizationParams`` instance (e.g. ``DHS``).
        name: Optional display name. If *None*, resolved from the preset registry.

    Examples:
        >>> from spatial_anonymization.config.presets import DHS, print_preset
        >>> print_preset(DHS)
    """
    # Resolve display name from registry if not provided
    if name is None:
        for key, val in PRESETS.items():
            if val is preset:
                name = key.upper()
                break
        if name is None:
            name = "Custom"

    # Resolve the set of fields explicitly configured in this preset.
    preset_key = None
    for key, val in PRESETS.items():
        if val is preset:
            preset_key = key
            break
    forced = _PRESET_EXPLICIT_FIELDS.get(preset_key or "", set())

    # Only print fields whose value differs from the dataclass default
    # OR that are listed in the preset's explicit-fields registry.
    explicit_fields: set[str] = set()
    for f in fields(preset):
        if f.name in forced:
            explicit_fields.add(f.name)
            continue
        value = getattr(preset, f.name)
        if f.default is not MISSING:
            default_val = f.default
        elif f.default_factory is not MISSING:
            default_val = f.default_factory()
        else:
            # Required field (no default) — always show
            explicit_fields.add(f.name)
            continue
        if value != default_val:
            explicit_fields.add(f.name)

    header = f" {name} Preset Parameters "
    width = max(70, len(header) + 4)
    print("\n" + "=" * width)
    print(header.center(width))
    print("=" * width)

    for f in fields(preset):
        if f.name not in explicit_fields:
            continue

        value = getattr(preset, f.name)
        desc = f.metadata.get("description", "")

        # Format the value for readability
        if isinstance(value, bool):
            val_str = "Yes" if value else "No"
        elif isinstance(value, float) and value == int(value):
            val_str = f"{int(value):,}"
        elif isinstance(value, (int, float)):
            val_str = f"{value:,}"
        elif value is None:
            val_str = "—"
        else:
            val_str = str(value)

        print(f"\n  {f.name}: {val_str}")
        if desc:
            # Wrap long descriptions
            for line in _wrap_text(desc, width=width - 6):
                print(f"      {line}")

    print("\n" + "=" * width + "\n")


def _wrap_text(text: str, width: int = 64) -> list[str]:
    """Wrap *text* to *width* characters, splitting on word boundaries."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if current and length + len(word) + 1 > width:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return lines
