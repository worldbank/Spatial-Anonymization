"""
Anonymization parameter definitions.

This module defines the AnonymizationParams class, which
captures all configurable options controlling spatial
anonymization behavior.
"""
import json
import yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass(slots=True)
class DataSourcesConfig:
    """
    Configuration for spatial input data sources and providers.

    This configuration specifies which data sources to use for admin boundaries,
    population, water bodies, etc. It enables reproducibility by documenting
    exactly which datasets were used for anonymization.
    """

    extract_displaced_population: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether to extract population for displaced points after anonymization. "
                "If False, skips post-displacement population extraction."
            )
        }
    )

    # ==================================================================
    # ADMINISTRATIVE BOUNDARIES
    # ==================================================================

    admin_boundary_source: Literal["geoboundaries", "gadm", "user_provided"] = field(
        default="geoboundaries",
        metadata={
            "description": (
                "Primary source for administrative boundaries. "
                "'geoboundaries' uses geoBoundaries API, "
                "'gadm' uses GADM database, "
                "'user_provided' indicates custom shapefile."
            )
        }
    )

    admin_boundary_fallback: Optional[Literal["geoboundaries", "gadm"]] = field(
        default="gadm",
        metadata={
            "description": (
                "Fallback source if primary admin boundary source fails. "
                "Set to None to disable fallback."
            )
        }
    )

    # ==================================================================
    # POPULATION DATA
    # ==================================================================

    population_source: Literal["worldpop", "landscan", "ghsl", "user_provided", "none"] = field(
        default="worldpop",
        metadata={
            "description": (
                "Source for population raster data. "
                "'worldpop' uses WorldPop Global dataset, "
                "'landscan' uses LandScan (future), "
                "'ghsl' uses Global Human Settlement Layer (future), "
                "'user_provided' indicates custom raster, "
                "'none' disables population-based methods."
            )
        }
    )

    population_dataset_version: str = field(
        default="R2025A_v1",
        metadata={
            "description": (
                "Dataset version identifier for population data. "
                "For WorldPop: 'R2025A_v1' refers to the 2025 release version 1."
            )
        }
    )

    population_year: int = field(
        default=2025,
        metadata={
            "description": (
                "Year for population data (e.g., 2020, 2025). "
                "Available range depends on source (WorldPop: 2015-2030)."
            )
        }
    )

    population_resolution: Literal["100m", "1km"] = field(
        default="100m",
        metadata={
            "description": (
                "Spatial resolution of population raster. "
                "WorldPop supports '100m' and '1km'."
            )
        }
    )

    population_type: Literal["constrained", "unconstrained"] = field(
        default="constrained",
        metadata={
            "description": (
                "Type of population model. "
                "'constrained' distributes population within settled areas only, "
                "'unconstrained' distributes uniformly across grid cells."
            )
        }
    )

    # ==================================================================
    # WATER BODIES
    # ==================================================================

    water_source: Literal["osm", "hydrosheds", "user_provided", "none"] = field(
        default="osm",
        metadata={
            "description": (
                "Source for water body geometries. "
                "'osm' uses OpenStreetMap via Overpass API, "
                "'hydrosheds' uses HydroSHEDS dataset (future), "
                "'user_provided' indicates custom shapefile, "
                "'none' disables water constraints."
            )
        }
    )

    # ==================================================================
    # DOWNLOAD BEHAVIOR
    # ==================================================================

    working_dir: Optional[str] = field(
        default=None,
        metadata={
            "description": (
                "Working directory for storing downloaded datasets and outputs. "
                "If None, uses system temporary directory."
            )
        }
    )

    cache_downloads: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether to save downloaded datasets to disk for reuse. "
                "Recommended to avoid repeated downloads."
            )
        }
    )

    auto_download: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether to automatically download missing datasets. "
                "If False, all datasets must be user-provided."
            )
        }
    )



@dataclass(slots=True)
class AnonymizationParams:
    """
    Policy configuration for spatial anonymization.

    This configuration specifies *what* anonymization should be applied,
    not *how data are prepared*. All preprocessing (e.g., clustering,
    deduplication, CRS handling) is expected to occur upstream.
    """
    # ...existing fields...

    method: Literal["dhs", "donut", "population_adaptive"] = field(
        metadata={
            "description": (
                "Anonymization method to use. "
                "'dhs' applies DHS-style random displacement, "
                "'donut' enforces a minimum displacement distance, "
                "and 'population_adaptive' selects displacement distance "
                "based on a minimum population threshold."
            )
        }
    )

    relax_admin_constraint_on_failure: bool = field(
        default=False,
        metadata={
            "description": (
                "If True, will retry anonymization without the admin region constraint if all attempts fail with it. "
                "This allows more flexibility for hard-to-anonymize points."
            )
        },
    )

    extract_displaced_population: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether to extract population for displaced points after anonymization. "
                "If False, skips post-displacement population extraction."
            )
        }
    )

    # ==================================================================
    # DISTANCE POLICY (METERS)
    # ==================================================================

    max_distance: float = field(
        default=2000.0,
        metadata={
            "description": (
                "Fallback maximum displacement distance (in meters) used "
                "when no urban/rural classification is available."
            )
        }
    )

    urban_max_distance: Optional[float] = field(
        default=None,
        metadata={
            "description": (
                "Maximum displacement distance (in meters) applied to "
                "urban locations when urban/rural classification is provided."
            )
        },
    )

    rural_max_distance: Optional[float] = field(
        default=None,
        metadata={
            "description": (
                "Maximum displacement distance (in meters) applied to "
                "rural locations when urban/rural classification is provided."
            )
        },
    )

    rural_extra_max_distance: Optional[float] = field(
        default=None,
        metadata={
            "description": (
                "Larger maximum displacement distance (in meters) applied "
                "to a subset of rural locations to reduce re-identification risk "
                "(DHS-style rural oversampling)."
            )
        },
    )

    rural_extra_proportion: float = field(
        default=0.0,
        metadata={
            "description": (
                "Proportion of rural locations that should receive the "
                "larger displacement distance defined by rural_extra_max_distance. "
                "Set to 0 to disable rural oversampling."
            )
        },
    )

    min_distance: float = field(
        default=0.0,
        metadata={
            "description": (
                "Minimum displacement distance (in meters) used as a fallback "
                "when urban- or rural-specific minimum distances are not provided. "
                "Primarily relevant for donut geomasking."
            )
        },
    )

    urban_min_distance: Optional[float] = field(
        default=None,
        metadata={
            "description": (
                "Minimum displacement distance (in meters) applied to "
                "urban locations when using donut-style geomasking."
            )
        },
    )

    rural_min_distance: Optional[float] = field(
        default=None,
        metadata={
            "description": (
                "Minimum displacement distance (in meters) applied to "
                "rural locations when using donut-style geomasking."
            )
        },
    )

    # ==================================================================
    # POPULATION-BASED ANONYMIZATION POLICY
    # ==================================================================

    min_population: Optional[int] = field(
        default=None,
        metadata={
            "description": (
                "Minimum population required within the anonymization radius. "
                "If specified, the displacement distance may be increased "
                "until this population threshold is met."
            )
        },
    )

    population_search_max_radius: Optional[float] = field(
        default=None,
        metadata={
            "description": (
                "Maximum radius (in meters) to search when attempting to "
                "satisfy the minimum population requirement. If None, "
                "search continues until failure."
            )
        },
    )

    population_step_m: float = field(
        default=500.0,
        metadata={
            "description": (
                "Increment (in meters) by which the search radius is "
                "expanded when using population-adaptive anonymization."
            )
        },
    )

    displaced_pop_radius_m: float = field(
        default=1000.0,
        metadata={
            "description": (
                "Buffer radius (in meters) around each displaced point "
                "used to extract population count from the raster. "
                "Set to 0 to skip post-displacement population extraction."
            )
        },
    )

    # ==================================================================
    # SPATIAL CONSTRAINTS
    # ==================================================================

    admin_level: int = field(
        default=2,
        metadata={
            "description": (
                "Administrative level used for enforcing boundary constraints "
                "(e.g. 0=country, 1=region, 2=district)."
            )
        },
    )

    enforce_admin_boundary: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether anonymized coordinates must remain within the "
                "specified administrative boundary."
            )
        },
    )

    enforce_water_constraint: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether anonymized coordinates should avoid falling "
                "within major water bodies when such data are available."
            )
        },
    )

    water_constraint_strict: bool = field(
        default=True,
        metadata={
            "description": (
                "If True, candidate points falling in water are rejected "
                "and resampled. If False, such points may be accepted "
                "with a warning."
            )
        },
    )

    # ==================================================================
    # DATA SOURCES
    # ==================================================================

    data_sources: DataSourcesConfig = field(
        default_factory=DataSourcesConfig,
        metadata={
            "description": (
                "Configuration for spatial input data sources. "
                "Specifies which datasets and providers to use for "
                "admin boundaries, population, and water bodies."
            )
        }
    )

    def max_distance_for(self, is_urban: Optional[bool]) -> float:
        """
        Return the appropriate maximum displacement distance based on urban/rural status.

        Args:
            is_urban: True for urban, False for rural, None for unknown

        Returns:
            Maximum distance in meters
        """
        if is_urban is None:
            return self.max_distance

        if is_urban:
            return (
                self.urban_max_distance
                if self.urban_max_distance is not None
                else self.max_distance
            )
        return (
            self.rural_max_distance
            if self.rural_max_distance is not None
            else self.max_distance
        )

    def min_distance_for(self, is_urban: Optional[bool]) -> float:
        """
        Return the appropriate minimum displacement distance based on urban/rural status.

        Args:
            is_urban: True for urban, False for rural, None for unknown

        Returns:
            Minimum distance in meters
        """
        if is_urban is None:
            return self.min_distance

        if is_urban:
            return (
                self.urban_min_distance
                if self.urban_min_distance is not None
                else self.min_distance
            )
        return (
            self.rural_min_distance
            if self.rural_min_distance is not None
            else self.min_distance
        )

    def to_yaml(self, path: str | Path):
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    def to_json(self, path: str | Path):
        """Save configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)