"""
Policy-level spatial anonymization orchestrator.

NOTE: This module contains legacy code and is not currently used by the main API.
      The current implementation uses SurveyAnonymizer in anonymizer.py instead.

This module defines the SurveyDisplacer class, which coordinates:
- radius selection (DHS, donut, or population-adaptive)
- candidate point generation
- constraint evaluation
- retry and failure handling

SurveyDisplacer is the main user-facing entry point for anonymizing
survey GPS coordinates, with behavior controlled by:
- unit type (household vs cluster)
- anonymization method
- active constraints
"""

# displacement.py
import random
from typing import Optional

from spatial_anonymization.geoprocessing.geo_utils import terminal_coordinates_faster
from spatial_anonymization.geoprocessing.radius import (
    sample_donut_radius,
    find_population_adaptive_radius,
)
from spatial_anonymization.geoprocessing.constraints import (
    AdminBoundaryConstraint,
    PopulationConstraint,
)
# Legacy import - commenting out as this module is not actively used
from spatial_anonymization.config import params


class SurveyDisplacer:
    """Survey coordinate anonymizer with explicit policies and constraints."""

    def __init__(
        self,
        *,
        unit_type: str,
        method: str,
        admin_polygon,
        population_raster: Optional[str] = None,
        utm_epsg: Optional[int] = None,
        overrides: Optional[dict] = None,
        max_attempts: int = 50,
    ):
        """
        Args:
            unit_type (str): "household" or "cluster".
            method (str): "dhs", "donut", or "population_adaptive".
            admin_polygon: Shapely polygon for admin boundary.
            population_raster (Optional[str]): Path to population raster.
            utm_epsg (Optional[int]): EPSG code for projected CRS.
            overrides (Optional[dict]): Override default preset values.
            max_attempts (int): Maximum sampling attempts.
        """
        self.unit_type = unit_type
        self.method = method
        self.max_attempts = max_attempts

        # Note: DEFAULT_PRESETS usage commented out - legacy code
        # params = DEFAULT_PRESETS[unit_type].copy()
        # if overrides:
        #     params.update(overrides)
        # self.params = params
        
        # Temporary fallback for legacy code
        self.params = overrides or {}

        self.admin_constraint = AdminBoundaryConstraint(admin_polygon)

        self.population_constraint = None
        if params.get("population_min") is not None:
            if not population_raster or not utm_epsg:
                raise ValueError(
                    "population_raster and utm_epsg required for population constraint"
                )
            self.population_constraint = PopulationConstraint(
                population_raster=population_raster,
                min_population=params["population_min"],
                utm_epsg=utm_epsg,
            )

    def _generate_candidate(self, lon: float, lat: float, radius_m: float):
        bearing = random.randint(0, 360)
        return terminal_coordinates_faster(
            lat1=lat, lon1=lon, d=radius_m / 1000, bearing=bearing
        )

    def displace(self, lon: float, lat: float):
        """Generate anonymized coordinate.

        Returns:
            Optional[tuple]: (lon, lat) or None if failed.
        """
        # --- radius selection ---
        if self.method == "population_adaptive":
            checker = self.population_constraint.make_radius_checker(lon, lat)
            radius = find_population_adaptive_radius(
                population_check_fn=checker,
                max_radius_m=self.params.get("max_distance"),
            )
            if radius is None:
                return None
        else:
            for _ in range(self.max_attempts):
                if self.method == "dhs":
                    radius = sample_donut_radius(0, self.params["max_distance"])
                elif self.method == "donut":
                    radius = sample_donut_radius(
                        self.params["min_distance"],
                        self.params["max_distance"],
                    )
                else:
                    raise NotImplementedError(self.method)

                new_lon, new_lat = self._generate_candidate(lon, lat, radius)
                if self.admin_constraint.check(new_lon, new_lat):
                    return new_lon, new_lat

            return None

        # --- population-adaptive candidate generation ---
        for _ in range(self.max_attempts):
            new_lon, new_lat = self._generate_candidate(lon, lat, radius)
            if self.admin_constraint.check(new_lon, new_lat):
                return new_lon, new_lat

        return None
