"""
Traffic Safety Score — Neighbourhood Safety Index for Toronto (2020–2025)

Queries the traffic_collisions_data table in PostgreSQL, computes a composite
safety index per neighbourhood, and exports two CSVs:
  1. neighbourhood_safety_scores.csv   — one row per neighbourhood (for Power BI map)
  2. neighbourhood_collisions_detail.csv — per-neighbourhood / year / type (for filters)

Scoring weights:
    Fatalities          5×
    Injury collisions   3×
    Fail-to-remain      2×
    Vulnerable users    2×  (pedestrian + bicycle + motorcycle)
    Property damage     1×

Recency weights:
    2024-2025  → 1.0
    2022-2023  → 0.7
    2020-2021  → 0.5

Higher Safety Index = Safer neighbourhood (0–100 scale).
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(os.path.join('project', '.env'))

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user':   os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host':   os.getenv('DB_HOST'),
    'port':   os.getenv('DB_PORT'),
}

OUTPUT_DIR = os.path.join('project', 'DB_csv')

# Severity weights
W_FATAL      = 5
W_INJURY     = 3
W_FTR        = 2
W_VULNERABLE = 2
W_PD         = 1

# Recency weights by year
RECENCY = {
    2024: 1.0, 2025: 1.0,
    2022: 0.7, 2023: 0.7,
    2020: 0.5, 2021: 0.5,
}

YEAR_MIN = 2020
YEAR_MAX = 2025


# ── SQL ───────────────────────────────────────────────────────────────────────
DETAIL_QUERY = """
SELECT
    neighbourhood_158,
    EXTRACT(YEAR FROM occ_date)::INT          AS occ_year,
    COUNT(*)                                   AS total_collisions,
    SUM(fatalities)                            AS fatalities,
    COUNT(*) FILTER (WHERE injury_collisions)  AS injury_collisions,
    COUNT(*) FILTER (WHERE pd_collisions)      AS pd_collisions,
    COUNT(*) FILTER (WHERE ftr_collisions)     AS ftr_collisions,
    COUNT(*) FILTER (WHERE pedestrian)         AS pedestrian_collisions,
    COUNT(*) FILTER (WHERE bicycle)            AS bicycle_collisions,
    COUNT(*) FILTER (WHERE motorcycle)         AS motorcycle_collisions,
    COUNT(*) FILTER (WHERE automobile)         AS automobile_collisions
FROM traffic_collisions_data
WHERE neighbourhood_158 <> 'NSA'
  AND occ_date >= '2020-01-01'
  AND occ_date <  '2026-01-01'
GROUP BY neighbourhood_158, EXTRACT(YEAR FROM occ_date)
ORDER BY neighbourhood_158, occ_year;
"""


def fetch_detail(conn):
    """Return per-neighbourhood / per-year detail as a DataFrame."""
    print("Querying traffic data (2020–2025, excl. NSA) …")
    df = pd.read_sql(DETAIL_QUERY, conn)
    print(f"  Retrieved {len(df)} rows across "
          f"{df['neighbourhood_158'].nunique()} neighbourhoods.")
    return df


def compute_scores(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate detail into one safety-score row per neighbourhood."""

    # ── 1. Apply recency weight to each year's counts ─────────────────────
    detail = detail.copy()
    detail['recency_w'] = detail['occ_year'].map(RECENCY).fillna(0.3)

    weighted_cols = [
        'total_collisions', 'fatalities', 'injury_collisions',
        'pd_collisions', 'ftr_collisions', 'pedestrian_collisions',
        'bicycle_collisions', 'motorcycle_collisions',
    ]
    for col in weighted_cols:
        detail[f'{col}_rw'] = detail[col] * detail['recency_w']

    # ── 2. Aggregate per neighbourhood ────────────────────────────────────
    agg = detail.groupby('neighbourhood_158').agg(
        total_collisions        = ('total_collisions', 'sum'),
        total_collisions_rw     = ('total_collisions_rw', 'sum'),
        fatalities              = ('fatalities', 'sum'),
        fatalities_rw           = ('fatalities_rw', 'sum'),
        injury_collisions       = ('injury_collisions', 'sum'),
        injury_collisions_rw    = ('injury_collisions_rw', 'sum'),
        pd_collisions           = ('pd_collisions', 'sum'),
        pd_collisions_rw        = ('pd_collisions_rw', 'sum'),
        ftr_collisions          = ('ftr_collisions', 'sum'),
        ftr_collisions_rw       = ('ftr_collisions_rw', 'sum'),
        pedestrian_collisions   = ('pedestrian_collisions', 'sum'),
        pedestrian_collisions_rw= ('pedestrian_collisions_rw', 'sum'),
        bicycle_collisions      = ('bicycle_collisions', 'sum'),
        bicycle_collisions_rw   = ('bicycle_collisions_rw', 'sum'),
        motorcycle_collisions   = ('motorcycle_collisions', 'sum'),
        motorcycle_collisions_rw= ('motorcycle_collisions_rw', 'sum'),
        years_active            = ('occ_year', 'nunique'),
    ).reset_index()

    # ── 3. Compute rates (per 1 000 recency-weighted collisions) ──────────
    rw_total = agg['total_collisions_rw'].replace(0, np.nan)

    agg['fatal_rate']      = (agg['fatalities_rw']           / rw_total) * 1000
    agg['injury_rate']     = (agg['injury_collisions_rw']    / rw_total) * 1000
    agg['pd_rate']         = (agg['pd_collisions_rw']        / rw_total) * 1000
    agg['ftr_rate']        = (agg['ftr_collisions_rw']       / rw_total) * 1000
    agg['vulnerable_rate'] = ((agg['pedestrian_collisions_rw']
                              + agg['bicycle_collisions_rw']
                              + agg['motorcycle_collisions_rw']) / rw_total) * 1000

    # ── 4. Composite raw danger score (higher = more dangerous) ───────────
    agg['raw_danger'] = (
        W_FATAL      * agg['fatal_rate']
      + W_INJURY     * agg['injury_rate']
      + W_FTR        * agg['ftr_rate']
      + W_VULNERABLE * agg['vulnerable_rate']
      + W_PD         * agg['pd_rate']
    )

    # ── 5. Also factor in absolute volume (recency-weighted) ──────────────
    #    Normalise volume to 0–1 and blend 70 % severity, 30 % volume
    vol_norm = (agg['total_collisions_rw'] - agg['total_collisions_rw'].min()) / \
               (agg['total_collisions_rw'].max() - agg['total_collisions_rw'].min())

    danger_norm = (agg['raw_danger'] - agg['raw_danger'].min()) / \
                  (agg['raw_danger'].max() - agg['raw_danger'].min())

    agg['blended_danger'] = 0.7 * danger_norm + 0.3 * vol_norm

    # ── 6. Invert to Safety Index 0–100 (higher = safer) ─────────────────
    agg['safety_index'] = ((1 - agg['blended_danger']) * 100).round(1)

    # ── 7. Quartile-based safety category ─────────────────────────────────
    agg['safety_category'] = pd.qcut(
        agg['safety_index'], q=4,
        labels=['Unsafe', 'Caution', 'Moderate', 'Safe'],
    )

    # ── 8. Rank (1 = safest) ──────────────────────────────────────────────
    agg['safety_rank'] = agg['safety_index'].rank(ascending=False, method='min').astype(int)

    return agg.sort_values('safety_rank')


def main():
    print("=" * 60)
    print("TRAFFIC SAFETY SCORE — Toronto Neighbourhoods (2020–2025)")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        # ── Fetch & compute ───────────────────────────────────────────────
        detail = fetch_detail(conn)
        scores = compute_scores(detail)

        # ── Save detail CSV ───────────────────────────────────────────────
        detail_path = os.path.join(OUTPUT_DIR, 'neighbourhood_collisions_detail.csv')
        detail.to_csv(detail_path, index=False)
        print(f"\n✓ Detail CSV saved:  {detail_path}  ({len(detail)} rows)")

        # ── Save scores CSV ──────────────────────────────────────────────
        score_cols = [
            'neighbourhood_158', 'safety_index', 'safety_category', 'safety_rank',
            'total_collisions', 'fatalities', 'injury_collisions',
            'pd_collisions', 'ftr_collisions',
            'pedestrian_collisions', 'bicycle_collisions', 'motorcycle_collisions',
            'fatal_rate', 'injury_rate', 'pd_rate', 'ftr_rate', 'vulnerable_rate',
            'years_active',
        ]
        scores_path = os.path.join(OUTPUT_DIR, 'neighbourhood_safety_scores.csv')
        scores[score_cols].to_csv(scores_path, index=False)
        print(f"✓ Scores CSV saved:  {scores_path}  ({len(scores)} rows)")

        # ── Print top / bottom 10 ────────────────────────────────────────
        print(f"\n{'─' * 60}")
        print("TOP 10 SAFEST NEIGHBOURHOODS")
        print(f"{'─' * 60}")
        top = scores.head(10)[['safety_rank', 'neighbourhood_158',
                                'safety_index', 'safety_category',
                                'total_collisions', 'fatalities']]
        print(top.to_string(index=False))

        print(f"\n{'─' * 60}")
        print("BOTTOM 10 LEAST SAFE NEIGHBOURHOODS")
        print(f"{'─' * 60}")
        bot = scores.tail(10)[['safety_rank', 'neighbourhood_158',
                                'safety_index', 'safety_category',
                                'total_collisions', 'fatalities']]
        print(bot.to_string(index=False))

        print(f"\n{'─' * 60}")
        print("SUMMARY STATISTICS")
        print(f"{'─' * 60}")
        print(f"  Neighbourhoods scored : {len(scores)}")
        print(f"  Safety Index range    : {scores['safety_index'].min()} – {scores['safety_index'].max()}")
        print(f"  Mean Safety Index     : {scores['safety_index'].mean():.1f}")
        print(f"  Median Safety Index   : {scores['safety_index'].median():.1f}")
        cats = scores['safety_category'].value_counts().sort_index()
        for cat, n in cats.items():
            print(f"  {cat:10s} : {n} neighbourhoods")

    finally:
        conn.close()

    print(f"\n{'=' * 60}")
    print("DONE — Import CSVs into Power BI for mapping.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
