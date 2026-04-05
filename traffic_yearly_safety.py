"""
Traffic Yearly Safety Score — One row per neighbourhood per year.

Loads Traffic_Collisions_Neighbourhood_details.csv, computes a weighted
risk score *within each year*, and normalises to a 0-100 safety index
so that Power BI year slicers work correctly.

Output:  Traffic_Collisions_Neighbourhood_Yearly_Safety.csv
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(_SCRIPT_DIR, "project", "DB_csv")

INPUT_CSV = os.path.join(CSV_DIR, "Traffic_Collisions_Neighbourhood_details.csv")
OUTPUT_CSV = os.path.join(CSV_DIR, "Traffic_Collisions_Neighbourhood_Yearly_Safety.csv")

# Old aggregated CSV that is no longer useful (one row per neighbourhood,
# does not respond to a year slicer in Power BI).
OLD_SCORES_CSV = os.path.join(CSV_DIR, "Traffic_Collisions_Neighbourhood_Safety_Scores.csv")

# ── Severity weights ─────────────────────────────────────────────────────────
# Fatalities carry the highest weight because they represent the worst outcome.
# Injury, vulnerable-road-user (pedestrian + bicycle + motorcycle), and
# fail-to-remain collisions are weighted more heavily than property-damage-only.
W_FATAL      = 10   # fatalities per collision (most severe)
W_INJURY     = 4    # injury collisions
W_PED        = 3    # pedestrian-involved collisions
W_BICYCLE    = 3    # bicycle-involved collisions
W_MOTORCYCLE = 3    # motorcycle-involved collisions
W_FTR        = 2    # fail-to-remain collisions
W_PD         = 1    # property-damage-only collisions

# ── Required and optional columns ────────────────────────────────────────────
REQUIRED_COLS = [
    "neighbourhood_158",
    "occ_year",
    "total_collisions",
    "injury_collisions",
    "fatalities",
    "pd_collisions",
    "ftr_collisions",
    "pedestrian_collisions",
    "bicycle_collisions",
    "motorcycle_collisions",
]

OPTIONAL_COLS = [
    "automobile_collisions",  # kept if present, not used in scoring
    "years_active",           # not meaningful at the yearly grain — see note below
]

# NOTE: years_active
# ------------------
# This column counts how many distinct years a neighbourhood appears in the
# *full* dataset.  At the yearly grain each row already represents exactly one
# year, so years_active is irrelevant for scoring.  If it exists in the input
# CSV it will be loaded but *not* included in the output table.


def load_and_validate(path: str) -> pd.DataFrame:
    """Load the CSV and verify that every required column exists."""
    print(f"Loading CSV: {path}")
    df = pd.read_csv(path)
    print(f"  Rows loaded: {len(df):,}")
    print(f"  Columns:     {list(df.columns)}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR — missing required columns: {missing}")

    # Warn about optional columns
    for col in OPTIONAL_COLS:
        if col in df.columns:
            print(f"  Optional column '{col}' found — loaded but not used in scoring.")
        else:
            print(f"  Optional column '{col}' not found — continuing without it.")

    return df


def compute_yearly_safety(df: pd.DataFrame) -> pd.DataFrame:
    """Compute safety_index, safety_rank, safety_category per year."""

    # ── 1. Group by neighbourhood + year (should already be 1-to-1, but
    #       summing makes the code resilient to duplicates). ───────────────
    agg_cols = {
        "total_collisions":      "sum",
        "fatalities":            "sum",
        "injury_collisions":     "sum",
        "pd_collisions":         "sum",
        "ftr_collisions":        "sum",
        "pedestrian_collisions": "sum",
        "bicycle_collisions":    "sum",
        "motorcycle_collisions": "sum",
    }
    if "automobile_collisions" in df.columns:
        agg_cols["automobile_collisions"] = "sum"

    grouped = (
        df.groupby(["neighbourhood_158", "occ_year"], as_index=False)
          .agg(agg_cols)
    )

    # ── 2. Weighted risk score (higher = more dangerous) ─────────────────
    #    The formula combines absolute counts weighted by severity.
    #    Property-damage collisions are already captured by total_collisions
    #    minus the more severe types, so we weight pd_collisions separately.
    grouped["risk_score"] = (
        W_FATAL      * grouped["fatalities"]
      + W_INJURY     * grouped["injury_collisions"]
      + W_PED        * grouped["pedestrian_collisions"]
      + W_BICYCLE    * grouped["bicycle_collisions"]
      + W_MOTORCYCLE * grouped["motorcycle_collisions"]
      + W_FTR        * grouped["ftr_collisions"]
      + W_PD         * grouped["pd_collisions"]
    )

    # ── 3. Normalise to 0-100 safety_index *within each year* ────────────
    #    Min-max normalisation per year so that neighbourhoods are only
    #    compared against peers in the same year.
    #    safety_index = 100 means safest, 0 means least safe.
    # Per-year min-max normalisation using transform
    yr_min = grouped.groupby("occ_year")["risk_score"].transform("min")
    yr_max = grouped.groupby("occ_year")["risk_score"].transform("max")
    denom = (yr_max - yr_min).replace(0, np.nan)
    grouped["safety_index"] = ((1 - (grouped["risk_score"] - yr_min) / denom) * 100).round(1)
    # If all neighbourhoods identical in a year, assign 50
    grouped["safety_index"] = grouped["safety_index"].fillna(50.0)

    # ── 4. Rank within each year (1 = safest) ────────────────────────────
    grouped["safety_rank"] = (
        grouped.groupby("occ_year")["safety_index"]
               .rank(ascending=False, method="min")
               .astype(int)
    )

    # ── 5. Safety category based on index thresholds ─────────────────────
    #    Thresholds (applied uniformly across years):
    #        80-100  →  Very Safe
    #        60-79.9 →  Safe
    #        40-59.9 →  Moderate
    #        20-39.9 →  Risky
    #         0-19.9 →  Very Risky
    bins   = [-0.1, 20, 40, 60, 80, 100.1]
    labels = ["Very Risky", "Risky", "Moderate", "Safe", "Very Safe"]
    grouped["safety_category"] = pd.cut(
        grouped["safety_index"], bins=bins, labels=labels
    )

    # Drop the intermediate risk_score column
    grouped.drop(columns=["risk_score"], inplace=True)

    # Sort for readability
    grouped.sort_values(["occ_year", "safety_rank"], inplace=True, ignore_index=True)

    return grouped


def remove_old_csv(path: str) -> None:
    """Delete the old aggregated safety-scores CSV if it exists."""
    if os.path.isfile(path):
        os.remove(path)
        print(f"  Removed old aggregated CSV: {path}")
    else:
        print(f"  Old CSV not found (already removed): {path}")


def main():
    print("=" * 65)
    print("TRAFFIC YEARLY SAFETY — neighbourhood × year")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────────────
    df = load_and_validate(INPUT_CSV)

    # ── Compute ───────────────────────────────────────────────────────────
    result = compute_yearly_safety(df)

    # ── Preview ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("PREVIEW (first 20 rows)")
    print(f"{'─' * 65}")
    print(result.head(20).to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved: {OUTPUT_CSV}  ({len(result):,} rows)")

    # ── Remove old aggregated CSV ─────────────────────────────────────────
    remove_old_csv(OLD_SCORES_CSV)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("SUMMARY")
    print(f"{'─' * 65}")
    print(f"  Rows in output          : {len(result):,}")
    print(f"  Unique neighbourhoods   : {result['neighbourhood_158'].nunique()}")
    print(f"  Min occ_year            : {result['occ_year'].min()}")
    print(f"  Max occ_year            : {result['occ_year'].max()}")
    cats = result["safety_category"].value_counts().sort_index()
    print(f"  Category distribution   :")
    for cat, n in cats.items():
        print(f"    {str(cat):12s} : {n:,} rows")
    print(f"\n{'=' * 65}")
    print("DONE — Import the CSV into Power BI; the occ_year column")
    print("will work with year slicers and filters.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
