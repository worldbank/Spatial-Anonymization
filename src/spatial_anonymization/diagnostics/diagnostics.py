# diagnostics.py
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd

from spatial_anonymization.geoprocessing.geo_utils import distance_points


def displacement_statistics(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    anon_lon_col: str = "lon_anon",
    anon_lat_col: str = "lat_anon",
    urban_col: Optional[str] = None,
    stats: Iterable[str] = (
        "count",
        "mean",
        "median",
        "min",
        "max",
        "p90",
        "p95",
        "p99",
    ),
    include_dhs_checks: bool = True,
    rural_long_tail_share: float = 0.01,
) -> pd.DataFrame:
    """
    Compute displacement-distance summary statistics between original and
    anonymized coordinates, with optional DHS-style validation checks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing original and anonymized coordinates.
    lon_col : str
        Column containing original longitude.
    lat_col : str
        Column containing original latitude.
    anon_lon_col : str, default="lon_anon"
        Column containing anonymized longitude.
    anon_lat_col : str, default="lat_anon"
        Column containing anonymized latitude.
    urban_col : str | None, default=None
        Optional column identifying urban/rural class. If provided, statistics
        are computed by this grouping column and for the full dataset.
    stats : Iterable[str], default=("count", "mean", "median", "min", "max", "p90", "p95", "p99")
        Statistics to compute. Supported values are:
        "count", "mean", "median", "min", "max", "std", "p50", "p90", "p95", "p99".
    include_dhs_checks : bool, default=True
        Whether to append simple DHS-style checks:
        - urban max should be <= 2 km
        - rural max should be <= 10 km
        - rural share above 5 km should be close to the expected long-tail share
    rural_long_tail_share : float, default=0.01
        Expected share of rural points displaced beyond 5 km under the DHS rule.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        - Area
        - Stat
        - Distance (Km)

        If `include_dhs_checks=True`, additional rows are included for DHS checks.
    """

    required_cols = [lon_col, lat_col, anon_lon_col, anon_lat_col]
    if urban_col:
        required_cols.append(urban_col)

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    work = df.copy()

    # Keep rows with complete original and anonymized coordinates
    mask = work[[lon_col, lat_col, anon_lon_col, anon_lat_col]].notna().all(axis=1)
    work = work.loc[mask].copy()

    if work.empty:
        result = pd.DataFrame(columns=["Area", "Stat", "Distance (Km)"])
        print("\nNo valid rows with both original and anonymized coordinates were found.")
        return result

    # Compute distance row-wise
    work["distance_km"] = work.apply(
        lambda row: distance_points(
            (row[lon_col], row[lat_col]),
            (row[anon_lon_col], row[anon_lat_col]),
        ),
        axis=1,
    )

    if urban_col:
        work["Area"] = work[urban_col].astype(str)
    else:
        work["Area"] = "all"

    supported_stats = {
        "count": lambda s: s.count(),
        "mean": lambda s: s.mean(),
        "median": lambda s: s.median(),
        "min": lambda s: s.min(),
        "max": lambda s: s.max(),
        "std": lambda s: s.std(),
        "p50": lambda s: s.quantile(0.50),
        "p90": lambda s: s.quantile(0.90),
        "p95": lambda s: s.quantile(0.95),
        "p99": lambda s: s.quantile(0.99),
    }

    invalid_stats = [stat for stat in stats if stat not in supported_stats]
    if invalid_stats:
        raise ValueError(
            f"Unsupported stats requested: {invalid_stats}. "
            f"Supported stats are: {list(supported_stats.keys())}"
        )

    out = []

    # Overall stats
    overall = work["distance_km"]
    for stat in stats:
        out.append(
            {
                "Area": "all",
                "Stat": stat,
                "Distance (Km)": supported_stats[stat](overall),
            }
        )

    # Group stats
    if urban_col:
        for area, g in work.groupby("Area", dropna=False):
            s = g["distance_km"]
            for stat in stats:
                out.append(
                    {
                        "Area": area,
                        "Stat": stat,
                        "Distance (Km)": supported_stats[stat](s),
                    }
                )

    result = pd.DataFrame(out)

    if include_dhs_checks and urban_col:
        checks = []

        area_map = {str(a).strip().lower(): a for a in work["Area"].dropna().unique()}

        urban_key = next((area_map[k] for k in area_map if k in {"urban", "u"}), None)
        rural_key = next((area_map[k] for k in area_map if k in {"rural", "r"}), None)

        if urban_key is not None:
            urban_dist = work.loc[work["Area"] == urban_key, "distance_km"]
            if not urban_dist.empty:
                checks.append(
                    {
                        "Area": urban_key,
                        "Stat": "check_max<=2km",
                        "Distance (Km)": float((urban_dist <= 2).all()),
                    }
                )
                checks.append(
                    {
                        "Area": urban_key,
                        "Stat": "share_>2km",
                        "Distance (Km)": (urban_dist > 2).mean(),
                    }
                )

        if rural_key is not None:
            rural_dist = work.loc[work["Area"] == rural_key, "distance_km"]
            if not rural_dist.empty:
                checks.append(
                    {
                        "Area": rural_key,
                        "Stat": "check_max<=10km",
                        "Distance (Km)": float((rural_dist <= 10).all()),
                    }
                )
                checks.append(
                    {
                        "Area": rural_key,
                        "Stat": "share_>5km",
                        "Distance (Km)": (rural_dist > 5).mean(),
                    }
                )
                checks.append(
                    {
                        "Area": rural_key,
                        "Stat": "share_>10km",
                        "Distance (Km)": (rural_dist > 10).mean(),
                    }
                )
                checks.append(
                    {
                        "Area": rural_key,
                        "Stat": "expected_share_>5km",
                        "Distance (Km)": rural_long_tail_share,
                    }
                )

        if checks:
            result = pd.concat([result, pd.DataFrame(checks)], ignore_index=True)

    # Nice display formatting — print each area as a separate block
    print(
        "\n=====================================================\n"
        " Distance Stats: Original vs. Displaced Coordinates\n"
        "====================================================="
    )

    for area in result["Area"].unique():
        block = result[result["Area"] == area].copy()
        block["Distance (Km)"] = block["Distance (Km)"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else str(x)
        )
        print(f"\n--- {area} ---")
        stat_w = max(len("Stat"), block["Stat"].astype(str).str.len().max())
        dist_w = max(len("Distance (Km)"), block["Distance (Km)"].astype(str).str.len().max())
        print(f"  {'Stat'.ljust(stat_w)}  {'Distance (Km)'.rjust(dist_w)}")
        print(f"  {'-' * stat_w}  {'-' * dist_w}")
        for _, row in block.iterrows():
            print(f"  {str(row['Stat']).ljust(stat_w)}  {str(row['Distance (Km)']).rjust(dist_w)}")

    return result


def admin_boundary_check(
    df: pd.DataFrame,
    admin_gdf: gpd.GeoDataFrame,
    *,
    lon_col: str,
    lat_col: str,
    anon_lon_col: str = "lon_anon",
    anon_lat_col: str = "lat_anon",
) -> dict:
    """
    Verify that each displaced point remains within the same admin
    boundary as its original point.

    Args:
        df: DataFrame with original and anonymized coordinates.
        admin_gdf: GeoDataFrame of admin boundaries (each feature = one unit).
        lon_col: Column with original longitude.
        lat_col: Column with original latitude.
        anon_lon_col: Column with anonymized longitude.
        anon_lat_col: Column with anonymized latitude.

    Returns:
        dict with keys:
            total: total records
            failed_anonymization: records with no anonymized coords
            no_admin_match: original point not in any admin feature
            in_same_admin: displaced point stayed in same admin unit
            escaped: displaced point landed in a different admin unit
            escape_rate: proportion escaped out of those checked
            details: DataFrame with per-row check results (row index,
                     original_admin_idx, displaced_in_same_admin)
    """
    valid_mask = df[anon_lon_col].notna() & df[anon_lat_col].notna()
    n_failed = int((~valid_mask).sum())
    work = df[valid_mask].copy()

    if len(work) == 0:
        return {
            "total": len(df),
            "failed_anonymization": n_failed,
            "no_admin_match": 0,
            "in_same_admin": 0,
            "escaped": 0,
            "escape_rate": 0.0,
            "details": pd.DataFrame(),
        }

    # Spatial-join original points to admin features
    orig_gdf = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work[lon_col], work[lat_col]),
        crs="EPSG:4326",
    )
    admin_indexed = (
        admin_gdf[["geometry"]]
        .reset_index()
        .rename(columns={"index": "_adm_idx"})
    )
    orig_joined = gpd.sjoin(
        orig_gdf, admin_indexed, how="left", predicate="within",
    )
    # Keep first match per original row
    orig_joined = orig_joined[~orig_joined.index.duplicated(keep="first")]

    # Spatial-join displaced points to admin features
    disp_gdf = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work[anon_lon_col], work[anon_lat_col]),
        crs="EPSG:4326",
    )
    disp_joined = gpd.sjoin(
        disp_gdf, admin_indexed, how="left", predicate="within",
    )
    disp_joined = disp_joined[~disp_joined.index.duplicated(keep="first")]

    # Compare admin indices
    results = pd.DataFrame(index=work.index)
    results["orig_admin_idx"] = orig_joined["_adm_idx"]
    results["disp_admin_idx"] = disp_joined["_adm_idx"]
    results["no_admin_match"] = results["orig_admin_idx"].isna()
    results["in_same_admin"] = (
        results["orig_admin_idx"] == results["disp_admin_idx"]
    ) & ~results["no_admin_match"]
    results["escaped"] = (
        results["orig_admin_idx"] != results["disp_admin_idx"]
    ) & ~results["no_admin_match"]

    n_no_match = int(results["no_admin_match"].sum())
    n_in_same = int(results["in_same_admin"].sum())
    n_escaped = int(results["escaped"].sum())
    n_checked = n_in_same + n_escaped

    summary = {
        "total": len(df),
        "failed_anonymization": n_failed,
        "no_admin_match": n_no_match,
        "in_same_admin": n_in_same,
        "escaped": n_escaped,
        "escape_rate": n_escaped / n_checked if n_checked > 0 else 0.0,
        "details": results,
    }

    print(
        f"[Admin Check] {n_in_same}/{n_checked} displaced points stayed in same admin unit, "
        f"{n_escaped} escaped ({summary['escape_rate']:.1%})"
    )

    return summary
