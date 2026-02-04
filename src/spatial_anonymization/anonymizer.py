"""
Survey-level spatial anonymization engine.

This module defines SurveyAnonymizer, the main entry point for
anonymizing survey GPS coordinates using DHS-style displacement,
donut geomasking, and population-adaptive methods.

SurveyAnonymizer orchestrates:
- radius selection
- candidate generation
- constraint evaluation
- retry and failure handling

All coordinates are assumed to be WGS84 decimal degrees.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import pandas as pd
from shapely.geometry import Point

from spatial_anonymization.config import AnonymizationConfig
from spatial_anonymization.inputs_manager import InputsManager
from spatial_anonymization.geo_utils import displace_point_geodesic
from spatial_anonymization.radius import (
    sample_dhs_radius,
    sample_donut_radius,
)
from spatial_anonymization.population import find_radius_for_population
from spatial_anonymization.constraints import passes_constraints


class SurveyAnonymizer:
    """
    Orchestrates survey GPS anonymization according to a defined policy
    and resolved spatial inputs.
    """

    def __init__(
        self,
        *,
        config: AnonymizationConfig,
        inputs: InputsManager,
        max_attempts: int = 50,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            config: AnonymizationConfig defining anonymization policy.
            inputs: InputsManager providing resolved spatial layers.
            max_attempts: Maximum candidate attempts per point.
            random_seed: Optional seed for reproducibility.
        """
        self.config = config
        self.inputs = inputs
        self.max_attempts = max_attempts

        if random_seed is not None:
            random.seed(random_seed)

    # ======================================================
    # CORE SINGLE-POINT LOGIC
    # ======================================================

    def anonymize_point(
        self,
        lon: float,
        lat: float,
        *,
        is_urban: Optional[bool] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Anonymize a single coordinate.

        Args:
            lon: Longitude in decimal degrees (WGS84).
            lat: Latitude in decimal degrees (WGS84).
            is_urban: Optional urban/rural indicator.

        Returns:
            (lon, lat) tuple if successful, otherwise None.
        """

        # --------------------------------------------------
        # RADIUS SELECTION
        # --------------------------------------------------

        if self.config.method == "population_adaptive":
            radius_m = find_radius_for_population(
                lon=lon,
                lat=lat,
                population_raster=self.inputs.get_population_dataset(),
                min_population=self.config.population_min,
                max_radius_m=self.config.max_distance_for(is_urban),
                utm_epsg=self.inputs.get_utm_epsg(),
            )
            if radius_m is None:
                return None

        else:
            for _ in range(self.max_attempts):
                radius_m = self._sample_radius(is_urban)

                candidate = displace_point_geodesic(
                    lon=lon,
                    lat=lat,
                    distance_m=radius_m,
                )

                if self._passes_constraints(candidate):
                    return candidate

            return None

        # --------------------------------------------------
        # POPULATION-ADAPTIVE CANDIDATE GENERATION
        # --------------------------------------------------

        for _ in range(self.max_attempts):
            candidate = displace_point_geodesic(
                lon=lon,
                lat=lat,
                distance_m=radius_m,
            )
            if self._passes_constraints(candidate):
                return candidate

        return None

    # ======================================================
    # DATAFRAME-LEVEL LOGIC
    # ======================================================

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        *,
        lon_col: str,
        lat_col: str,
        group_id_col: Optional[str] = None,
        urban_col: Optional[str] = None,
        out_lon_col: str = "lon_anon",
        out_lat_col: str = "lat_anon",
    ) -> pd.DataFrame:
        """
        Anonymize coordinates in a DataFrame.

        Supports both household-level and cluster-level anonymization.

        Args:
            df: Input DataFrame.
            lon_col: Longitude column name.
            lat_col: Latitude column name.
            group_id_col: Optional cluster/group ID column.
            urban_col: Optional urban/rural indicator column.
            out_lon_col: Output anonymized longitude column.
            out_lat_col: Output anonymized latitude column.

        Returns:
            DataFrame with anonymized coordinates added.
        """

        df = df.copy()

        if group_id_col:
            anonymized = {}

            for gid, group in df.groupby(group_id_col):
                lon = group[lon_col].iloc[0]
                lat = group[lat_col].iloc[0]
                is_urban = group[urban_col].iloc[0] if urban_col else None

                result = self.anonymize_point(lon, lat, is_urban=is_urban)
                anonymized[gid] = result

            df[out_lon_col] = df[group_id_col].map(
                lambda g: anonymized[g][0] if anonymized[g] else None
            )
            df[out_lat_col] = df[group_id_col].map(
                lambda g: anonymized[g][1] if anonymized[g] else None
            )

        else:
            results = []
            for _, row in df.iterrows():
                is_urban = row[urban_col] if urban_col else None
                result = self.anonymize_point(
                    row[lon_col], row[lat_col], is_urban=is_urban
                )
                results.append(result)

            df[out_lon_col] = [r[0] if r else None for r in results]
            df[out_lat_col] = [r[1] if r else None for r in results]

        return df

    # ======================================================
    # INTERNAL HELPERS
    # ======================================================

    def _sample_radius(self, is_urban: Optional[bool]) -> float:
        """Sample a displacement radius based on policy."""
        if self.config.method == "dhs":
            return sample_dhs_radius(self.config.max_distance_for(is_urban))

        if self.config.method == "donut":
            return sample_donut_radius(
                self.config.min_distance_for(is_urban),
                self.config.max_distance_for(is_urban),
            )

        raise ValueError(f"Unsupported method: {self.config.method}")

    def _passes_constraints(self, candidate: Tuple[float, float]) -> bool:
        """Check all active constraints."""
        lon, lat = candidate
        point = Point(lon, lat)

        return passes_constraints(
            point=point,
            admin_geom=self.inputs.get_admin_polygon() if self.config.enforce_admin_boundary else None,
            water_gdf=self.inputs.get_water_gdf() if self.config.enforce_water_constraint else None,
            roads_gdf=self.inputs.get_roads_gdf() if self.config.enforce_road_constraint else None,
            road_buffer_m=self.config.road_buffer_m or 0.0,
            allow_roads=not self.config.enforce_road_constraint,
        )
