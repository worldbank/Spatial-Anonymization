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

import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd

from spatial_anonymization.config.params import AnonymizationParams
from spatial_anonymization.data.inputs_manager import InputsManager
from spatial_anonymization.geoprocessing.geo_utils import (
    displace_point_geodesic,
    distance_points,
)
from spatial_anonymization.geoprocessing.radius import (
    sample_donut_radius,
)
from spatial_anonymization.geoprocessing.population import (
    find_radius_for_population,
    RasterPopulationProvider,
)
from spatial_anonymization.geoprocessing.constraints import passes_constraints

# Min points before spawning worker processes (avoids pool overhead on tiny data)
_PARALLEL_THRESHOLD = 50


# ---- Shared data for worker processes (set via pool initializer) ----
_worker_config: Optional[AnonymizationParams] = None
_worker_water_gdf: Optional[gpd.GeoDataFrame] = None
_worker_roads_gdf: Optional[gpd.GeoDataFrame] = None
_worker_max_attempts: int = 50


def _init_worker(config, max_attempts, water_gdf, roads_gdf):
    """Initialize large shared data once per worker process."""
    global _worker_config, _worker_water_gdf, _worker_roads_gdf, _worker_max_attempts
    _worker_config = config
    _worker_max_attempts = max_attempts
    _worker_water_gdf = water_gdf
    _worker_roads_gdf = roads_gdf
    # Pre-build spatial index in each worker
    if _worker_water_gdf is not None and not _worker_water_gdf.empty:
        _ = _worker_water_gdf.sindex


def _worker_anonymize_point(
    lon: float,
    lat: float,
    is_urban: Optional[bool],
    admin_geom: Optional[Polygon],
) -> dict:
    """Standalone worker — uses module-level shared data for large objects.

    Returns a dict with keys: lon, lat, distance_km, attempts, success.
    """
    from shapely.prepared import prep as shapely_prep

    config = _worker_config
    max_attempts = _worker_max_attempts
    water_gdf = _worker_water_gdf
    roads_gdf = _worker_roads_gdf

    road_buffer_m = config.road_buffer_m or 0.0
    allow_roads = not config.enforce_road_constraint
    prepared_admin = shapely_prep(admin_geom) if admin_geom is not None else None

    def _passes(candidate: Tuple[float, float]) -> bool:
        point = Point(candidate[0], candidate[1])
        return passes_constraints(
            point=point,
            prepared_admin=prepared_admin,
            water_gdf=water_gdf,
            roads_gdf=roads_gdf,
            road_buffer_m=road_buffer_m,
            allow_roads=allow_roads,
        )

    fail = {"lon": None, "lat": None, "distance_km": None,
            "attempts": max_attempts, "success": False}

    # --- DHS / Donut methods ---
    if config.method in ("dhs", "donut"):
        for attempt in range(1, max_attempts + 1):
            if config.method == "dhs":
                radius_m = sample_donut_radius(0, config.max_distance_for(is_urban))
            else:
                radius_m = sample_donut_radius(
                    config.min_distance_for(is_urban),
                    config.max_distance_for(is_urban),
                )
            candidate = displace_point_geodesic(lon=lon, lat=lat, distance_m=radius_m)
            if _passes(candidate):
                dist_km = distance_points((lon, lat), candidate)
                return {"lon": candidate[0], "lat": candidate[1],
                        "distance_km": dist_km, "attempts": attempt,
                        "success": True, "relaxed_admin": False}
        # If all attempts fail and relax_admin_constraint_on_failure is True, retry without admin constraint
        if getattr(config, "relax_admin_constraint_on_failure", False) and prepared_admin is not None:
            for attempt in range(1, max_attempts + 1):
                if config.method == "dhs":
                    radius_m = sample_donut_radius(0, config.max_distance_for(is_urban))
                else:
                    radius_m = sample_donut_radius(
                        config.min_distance_for(is_urban),
                        config.max_distance_for(is_urban),
                    )
                candidate = displace_point_geodesic(lon=lon, lat=lat, distance_m=radius_m)
                # Pass None for prepared_admin to relax the constraint
                if passes_constraints(
                    point=Point(candidate[0], candidate[1]),
                    prepared_admin=None,
                    water_gdf=water_gdf,
                    roads_gdf=roads_gdf,
                    road_buffer_m=road_buffer_m,
                    allow_roads=allow_roads,
                ):
                    dist_km = distance_points((lon, lat), candidate)
                    return {"lon": candidate[0], "lat": candidate[1],
                            "distance_km": dist_km, "attempts": attempt + max_attempts,
                            "success": True, "relaxed_admin": True}
        return fail

    # --- population_adaptive is NOT parallelised (rasterio not picklable) ---
    return fail

class SurveyAnonymizer:
    """
    Orchestrates survey GPS anonymization according to a defined policy
    and resolved spatial inputs.
    """

    def __init__(
        self,
        *,
        config: AnonymizationParams,
        inputs: InputsManager,
        max_attempts: int = 50,
        random_seed: Optional[int] = None,
        n_workers: Optional[int] = None,
    ):
        """
        Args:
            config: AnonymizationParams defining anonymization policy.
            inputs: InputsManager providing resolved spatial layers.
            max_attempts: Maximum candidate attempts per point.
            random_seed: Optional seed for reproducibility.
            n_workers: Number of parallel worker processes.
                       None = use all available CPU cores.
                       1 = sequential (no multiprocessing).
        """
        self.config = config
        self.inputs = inputs
        self.max_attempts = max_attempts
        self.n_workers = n_workers
        self._pop_provider = None

        if random_seed is not None:
            random.seed(random_seed)

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
            pop_provider = RasterPopulationProvider(
                raster_path=str(self.inputs.population_raster_path),
            )
            min_r = self.config.min_distance_for(is_urban) or self.config.population_step_m
            try:
                radius_m = find_radius_for_population(
                    population_fn=pop_provider.population_within_radius,
                    lon=lon,
                    lat=lat,
                    population_threshold=self.config.min_population or 0,
                    min_radius_m=min_r,
                    max_radius_m=self.config.max_distance_for(is_urban),
                    step_m=self.config.population_step_m,
                )
            except ValueError:
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

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        *,
        lon_col: str,
        lat_col: str,
        urban_col: Optional[str] = None,
        urban_value: Optional[str] = "urban",
        rural_value: Optional[str] = "rural",
        out_lon_col: str = "lon_anon",
        out_lat_col: str = "lat_anon",
    ) -> pd.DataFrame:
        """
        Anonymize coordinates in a DataFrame.

        Each row is treated as an independent point and displaced individually.
        If you need cluster-level anonymization, pre-aggregate to cluster
        centroids before calling this method.

        Args:
            df: Input DataFrame.
            lon_col: Longitude column name.
            lat_col: Latitude column name.
            urban_col: Optional urban/rural indicator column.
            urban_value: Value in urban_col representing urban areas.
            rural_value: Value in urban_col representing rural areas.
            out_lon_col: Output anonymized longitude column.
            out_lat_col: Output anonymized latitude column.

        Returns:
            DataFrame with anonymized coordinates added.
        """

        df = df.copy()
        n_total = len(df)

        # Decide whether to use parallel execution
        use_parallel = (
            self.config.method in ("dhs", "donut")
            and (self.n_workers is None or self.n_workers > 1)
        )
        effective_workers = self.n_workers or os.cpu_count() or 1

        # Pre-resolve constraint data once (shared across workers)
        water_gdf = (
            self.inputs.get_water_gdf()
            if self.config.enforce_water_constraint else None
        )
        roads_gdf = (
            self.inputs.get_roads_gdf()
            if self.config.enforce_road_constraint else None
        )

        # ---------------------------------------------------------
        # Per-point admin boundary: spatial-join each original point
        # to its admin feature so displaced candidates are checked
        # against the SAME admin unit, not a merged country outline.
        # ---------------------------------------------------------
        admin_geom_lookup: dict[int, Polygon] = {}  # df row index -> polygon
        admin_fallback: Optional[Polygon] = None

        if self.config.enforce_admin_boundary and self.inputs.has_admin():
            admin_gdf = self.inputs.get_admin_gdf()
            admin_fallback = admin_gdf.unary_union

            # Build a GeoDataFrame of original points for spatial join
            points_gdf = gpd.GeoDataFrame(
                df[[lon_col, lat_col]],
                geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                crs="EPSG:4326",
            )
            joined = gpd.sjoin(
                points_gdf,
                admin_gdf[["geometry"]].reset_index().rename(columns={"index": "_adm_idx"}),
                how="left",
                predicate="within",
            )
            # Map each df row to its admin polygon (use first match if duplicates)
            for df_idx, row in joined.iterrows():
                if df_idx in admin_geom_lookup:
                    continue  # keep first match
                adm_idx = row.get("_adm_idx")
                if pd.notna(adm_idx):
                    admin_geom_lookup[df_idx] = admin_gdf.geometry.iloc[int(adm_idx)]

            n_matched = len(admin_geom_lookup)
            n_fallback = n_total - n_matched
            print(
                f"[Anonymizer] Admin join: {n_matched}/{n_total} points matched to admin unit"
                + (f" ({n_fallback} will use merged boundary fallback)" if n_fallback else "")
            )

        print(f"[Anonymizer] Displacing {n_total} points")

        points = []
        for i, (_, row) in enumerate(df.iterrows()):
            raw = row[urban_col] if urban_col else None
            is_urban = (raw == urban_value) if raw is not None else None
            pt_admin = admin_geom_lookup.get(df.index[i], admin_fallback)
            points.append((row[lon_col], row[lat_col], is_urban, pt_admin))

        # Report urban/rural breakdown and effective distances
        if urban_col:
            n_urban = sum(1 for *_, iu, _ in points if iu is True)
            n_rural = sum(1 for *_, iu, _ in points if iu is False)
            n_unknown = sum(1 for *_, iu, _ in points if iu is None)
            print(
                f"[Anonymizer] Urban/rural split: {n_urban} urban, {n_rural} rural"
                + (f", {n_unknown} unknown" if n_unknown else "")
            )
            print(
                f"[Anonymizer] Effective max distances: "
                f"urban={self.config.max_distance_for(True)}m, "
                f"rural={self.config.max_distance_for(False)}m"
            )
        else:
            print(
                f"[Anonymizer] No urban/rural column — "
                f"using fallback max_distance={self.config.max_distance}m for all points"
            )

        if use_parallel and n_total >= _PARALLEL_THRESHOLD:
            print(f"[Anonymizer] Using {effective_workers} worker processes")
            results = [None] * n_total
            with ProcessPoolExecutor(
                max_workers=effective_workers,
                initializer=_init_worker,
                initargs=(self.config, self.max_attempts, water_gdf, roads_gdf),
            ) as pool:
                future_to_idx = {
                    pool.submit(
                        _worker_anonymize_point,
                        lon, lat, is_urban, pt_admin,
                    ): i
                    for i, (lon, lat, is_urban, pt_admin) in enumerate(points)
                }
                processed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    processed += 1
                    if processed % 100 == 0 or processed == n_total:
                        print(f"[Anonymizer] Processed {processed}/{n_total} points")
        else:
            if n_total >= _PARALLEL_THRESHOLD:
                print("[Anonymizer] Sequential mode (population_adaptive not parallelised)")
            results = []
            # Cache RasterPopulationProvider for all points if needed
            if self.config.method == "population_adaptive" or (
                self.config.displaced_pop_radius_m > 0 and hasattr(self.inputs, "population_raster_path") and self.inputs.population_raster_path is not None
            ):
                if self._pop_provider is None:
                    self._pop_provider = RasterPopulationProvider(
                        raster_path=str(self.inputs.population_raster_path),
                    )
            for idx, (lon, lat, is_urban, pt_admin) in enumerate(points, 1):
                result = self._anonymize_point_with_admin(
                    lon, lat, is_urban=is_urban, admin_geom=pt_admin,
                )
                results.append(result)
                if idx % 100 == 0 or idx == n_total:
                    print(f"[Anonymizer] Processed {idx}/{n_total} points")

        df[out_lon_col] = [r["lon"] for r in results]
        df[out_lat_col] = [r["lat"] for r in results]
        df["distance_km"] = [r["distance_km"] for r in results]
        df["attempts"] = [r["attempts"] for r in results]

        succeeded = df[out_lon_col].notna().sum()
        failed = n_total - succeeded
        print(f"[Anonymizer] Done. {succeeded} succeeded, {failed} failed.")

        # Post-displacement population extraction (batched)
        pop_radius = self.config.displaced_pop_radius_m
        pop_col = f"pop_within_{int(pop_radius)}m"
        extract_pop = True
        if hasattr(self.config, "extract_displaced_population"):
            extract_pop = self.config.extract_displaced_population
        if (
            extract_pop
            and pop_radius > 0
            and hasattr(self.inputs, "population_raster_path")
            and self.inputs.population_raster_path is not None
        ):
            print(f"[Anonymizer] Extracting population within {int(pop_radius)}m of displaced points...")
            pop_provider = self._pop_provider or RasterPopulationProvider(
                raster_path=str(self.inputs.population_raster_path),
            )

            success_indices = [i for i, r in enumerate(results) if r["success"]]
            if success_indices:
                pts = [(results[i]["lon"], results[i]["lat"]) for i in success_indices]
                pop_arr = pop_provider.batch_population_within_radius(pts, pop_radius)

                pop_lookup = {si: int(pop_arr[j]) for j, si in enumerate(success_indices)}
                pop_values = [pop_lookup.get(i) if results[i]["success"] else None
                              for i in range(len(results))]
            else:
                pop_values = [None] * len(results)

            df[pop_col] = pop_values
            valid_pops = [p for p in pop_values if p is not None]
            if valid_pops:
                print(
                    f"[Anonymizer] Population ({pop_col}): "
                    f"mean={sum(valid_pops)/len(valid_pops):.0f}, "
                    f"min={min(valid_pops)}, max={max(valid_pops)}"
                )
        # If extract_pop is False, do not add the population column at all

        # Save failed displacement points to intermediate_data
        if failed > 0:
            failed_mask = df[out_lon_col].isna()
            failed_df = df.loc[failed_mask, [lon_col, lat_col]].copy()
            if hasattr(self.inputs, "intermediate_dir") and self.inputs.intermediate_dir is not None:
                from pathlib import Path as _Path
                csv_path = _Path(self.inputs.intermediate_dir) / "failed_displacement.csv"
                failed_df.to_csv(csv_path, index=True)

                failed_gdf = gpd.GeoDataFrame(
                    failed_df,
                    geometry=gpd.points_from_xy(failed_df[lon_col], failed_df[lat_col]),
                    crs="EPSG:4326",
                )
                geojson_path = _Path(self.inputs.intermediate_dir) / "failed_displacement.geojson"
                failed_gdf.to_file(geojson_path, driver="GeoJSON")

                self.inputs.log(
                    f"DISPLACEMENT | {failed} of {n_total} points failed displacement "
                    f"| Saved to {csv_path} and {geojson_path}"
                )
            else:
                self.inputs.log(
                    f"DISPLACEMENT | {failed} of {n_total} points failed displacement "
                    f"| No output_dir set, files not saved"
                )
        else:
            self.inputs.log(f"DISPLACEMENT | All {n_total} points displaced successfully")

        # Save anonymized output to output_data
        if hasattr(self.inputs, "output_data_dir") and self.inputs.output_data_dir is not None:
            from pathlib import Path as _Path
            csv_out = _Path(self.inputs.output_data_dir) / "anonymized.csv"
            df.to_csv(csv_out, index=False)

            # Build GeoDataFrame from successfully displaced points
            success_mask = df[out_lon_col].notna()
            out_gdf = gpd.GeoDataFrame(
                df.loc[success_mask].copy(),
                geometry=gpd.points_from_xy(
                    df.loc[success_mask, out_lon_col],
                    df.loc[success_mask, out_lat_col],
                ),
                crs="EPSG:4326",
            )
            geojson_out = _Path(self.inputs.output_data_dir) / "anonymized.geojson"
            out_gdf.to_file(geojson_out, driver="GeoJSON")

            self.inputs.log(
                f"OUTPUT | Saved anonymized data to {csv_out} and {geojson_out}"
            )

        return df

    def _anonymize_point_with_admin(
        self,
        lon: float,
        lat: float,
        *,
        is_urban: Optional[bool] = None,
        admin_geom: Optional[Polygon] = None,
    ) -> dict:
        """Anonymize a single point using a specific admin polygon for constraint.

        Returns a dict with keys: lon, lat, distance_km, attempts, success.
        """
        from shapely.prepared import prep as shapely_prep

        water_gdf = (
            self.inputs.get_water_gdf()
            if self.config.enforce_water_constraint else None
        )
        roads_gdf = (
            self.inputs.get_roads_gdf()
            if self.config.enforce_road_constraint else None
        )
        road_buffer_m = self.config.road_buffer_m or 0.0
        allow_roads = not self.config.enforce_road_constraint
        prepared_admin = shapely_prep(admin_geom) if admin_geom is not None else None

        def _check(candidate: Tuple[float, float]) -> bool:
            point = Point(candidate[0], candidate[1])
            return passes_constraints(
                point=point,
                prepared_admin=prepared_admin,
                water_gdf=water_gdf,
                roads_gdf=roads_gdf,
                road_buffer_m=road_buffer_m,
                allow_roads=allow_roads,
            )

        fail = {"lon": None, "lat": None, "distance_km": None,
                "attempts": self.max_attempts, "success": False}

        if self.config.method == "population_adaptive":
            pop_provider = RasterPopulationProvider(
                raster_path=str(self.inputs.population_raster_path),
            )
            min_r = self.config.min_distance_for(is_urban) or self.config.population_step_m
            try:
                radius_m = find_radius_for_population(
                    population_fn=pop_provider.population_within_radius,
                    lon=lon,
                    lat=lat,
                    population_threshold=self.config.min_population or 0,
                    min_radius_m=min_r,
                    max_radius_m=self.config.max_distance_for(is_urban),
                    step_m=self.config.population_step_m,
                )
            except ValueError:
                return fail
            for attempt in range(1, self.max_attempts + 1):
                candidate = displace_point_geodesic(lon=lon, lat=lat, distance_m=radius_m)
                if _check(candidate):
                    dist_km = distance_points((lon, lat), candidate)
                    return {"lon": candidate[0], "lat": candidate[1],
                            "distance_km": dist_km, "attempts": attempt,
                            "success": True}
            return fail

        for attempt in range(1, self.max_attempts + 1):
            radius_m = self._sample_radius(is_urban)
            candidate = displace_point_geodesic(lon=lon, lat=lat, distance_m=radius_m)
            if _check(candidate):
                dist_km = distance_points((lon, lat), candidate)
                return {"lon": candidate[0], "lat": candidate[1],
                        "distance_km": dist_km, "attempts": attempt,
                        "success": True}
        return fail

    def _sample_radius(self, is_urban: Optional[bool]) -> float:
        """Sample a displacement radius based on policy."""
        if self.config.method == "dhs":
            return sample_donut_radius(0, self.config.max_distance_for(is_urban))

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
