# diagnostics.py
import pandas as pd
from typing import Iterable, Optional

from spatial_anonymization.geo_utils import distance_points


def displacement_statistics(
    df: pd.DataFrame,
    *,
    lon_col: str,
    lat_col: str,
    anon_lon_col: str,
    anon_lat_col: str,
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
                    "stat": stat,
                    "area": area,
                    "value": getattr(g["distance_km"], stat)(),
                }
            )

    return pd.DataFrame(out)


def anonymization_outcome_summary(
    df: pd.DataFrame,
    *,
    anon_lon_col: str,
    anon_lat_col: str,
) -> dict:
    """
    Summarize per-record anonymization outcomes.

    Reports how many observations were successfully anonymized
    versus those that failed to produce anonymized coordinates.
    """
    total = len(df)
    failed_mask = df[anon_lon_col].isna() | df[anon_lat_col].isna()

    return {
        "total": total,
        "success": int((~failed_mask).sum()),
        "failed": int(failed_mask.sum()),
        "failure_rate": float(failed_mask.mean()),
    }
