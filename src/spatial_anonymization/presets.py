"""
Predefined anonymization policy presets.

Presets provide named configurations for common survey anonymization
workflows, such as DHS-style displacement.
"""

from spatial_anonymization.config import AnonymizationConfig


LSMS_CLUSTER = AnonymizationConfig(
    unit_type="cluster",
    method="dhs",

    # Distance policy
    max_distance=5000,                 # fallback
    urban_max_distance=2000,
    rural_max_distance=5000,
    rural_extra_max_distance=10000,
    rural_extra_proportion=0.1,

    # Donut (disabled by default)
    min_distance=0.0,

    # Population policy
    min_population=500,
    population_search_max_radius=15000,
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
