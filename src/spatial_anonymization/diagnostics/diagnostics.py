# diagnostics.py
import pandas as pd
import geopandas as gpd
from typing import Iterable, Optional

from spatial_anonymization.geoprocessing.geo_utils import distance_points


def displacement_statistics(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    anon_lon_col: str = "lon_anon",
    anon_lat_col: str = "lat_anon",
    urban_col: Optional[str] = None,
    stats: Iterable[str] = ("mean", "median", "min", "max"),
) -> pd.DataFrame:
    """
    Compute summary statistics of displacement distance between
    original and anonymized coordinates.

    Args:
        df: DataFrame containing original and anonymized coordinates.
        lon_col: Column with original longitude.
        lat_col: Column with original latitude.
        anon_lon_col: Column with anonymized longitude.
        anon_lat_col: Column with anonymized latitude.
        urban_col: Optional urban/rural indicator column.
        stats: Iterable of statistics to compute.

    Returns:
        DataFrame with columns: stat, area, value
    """

    records = []

    for _, row in df.iterrows():
        if pd.isna(row[anon_lon_col]) or pd.isna(row[anon_lat_col]):
            continue

        d_km = distance_points(
            (row[lon_col], row[lat_col]),
            (row[anon_lon_col], row[anon_lat_col]),
        )

        area = row[urban_col] if urban_col else "all"
        records.append({"area": area, "distance_km": d_km})

    dist_df = pd.DataFrame(records)

    out = []
    for area, g in dist_df.groupby("area"):
        for stat in stats:
            out.append(
                {
                    "Stat": stat,
                    "Area": area,
                    "Distance (Km)": getattr(g["distance_km"], stat)(),
                }
            )

    result = pd.DataFrame(out)

    print(
        "\n=====================================================\n"
        " Distance Stats:Original vs.Displaced Coordinates\n"
        "====================================================="
    )
    # Build formatted table with underlined headers
    col_widths = {
        col: max(len(col), result[col].astype(str).str.len().max())
        for col in result.columns
    }
    header = "  ".join(col.ljust(col_widths[col]) for col in result.columns)
    separator = "  ".join("-" * col_widths[col] for col in result.columns)
    rows = "\n".join(
        "  ".join(str(row[col]).ljust(col_widths[col]) for col in result.columns)
        for _, row in result.iterrows()
    )
    print(f"{header}\n{separator}\n{rows}")

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
