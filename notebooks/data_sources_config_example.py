"""
Example demonstrating the new DataSourcesConfig for configuring input data sources.
"""

from spatial_anonymization import AnonymizationConfig, DataSourcesConfig
from pathlib import Path

# Example 1: Default configuration (uses all automatic downloads)
config_default = AnonymizationConfig(
    method="dhs",
    unit_type="household",
    urban_max_distance=2000,
    rural_max_distance=5000,
)

print("Default Data Sources Configuration:")
print(f"  Admin boundary source: {config_default.data_sources.admin_boundary_source}")
print(f"  Population source: {config_default.data_sources.population_source}")
print(f"  Population year: {config_default.data_sources.population_year}")
print(f"  Water source: {config_default.data_sources.water_source}")
print(f"  Auto download: {config_default.data_sources.auto_download}")
print()

# Example 2: Custom data sources configuration
custom_data_sources = DataSourcesConfig(
    admin_boundary_source="gadm",  # Prefer GADM over geoBoundaries
    admin_boundary_fallback="geoboundaries",
    population_source="worldpop",
    population_year=2020,  # Use 2020 data instead of 2025
    population_dataset_version="R2025A_v1",
    water_source="none",  # Disable water constraints
    working_dir="/path/to/cache",
    cache_downloads=True,
)

config_custom = AnonymizationConfig(
    method="population_adaptive",
    unit_type="cluster",
    min_population=1000,
    data_sources=custom_data_sources,
)

print("Custom Data Sources Configuration:")
print(f"  Admin boundary source: {config_custom.data_sources.admin_boundary_source}")
print(f"  Population source: {config_custom.data_sources.population_source}")
print(f"  Population year: {config_custom.data_sources.population_year}")
print(f"  Water source: {config_custom.data_sources.water_source}")
print(f"  Working dir: {config_custom.data_sources.working_dir}")
print()

# Example 3: Save configuration to YAML
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

yaml_path = output_dir / "anonymization_config.yaml"
config_custom.to_yaml(yaml_path)
print(f"Configuration saved to: {yaml_path}")
print()

# Example 4: Show what the YAML looks like
print("YAML Preview:")
print("=" * 60)
with open(yaml_path, 'r') as f:
    print(f.read())
