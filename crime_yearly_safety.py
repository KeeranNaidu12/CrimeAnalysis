"""
Crime Yearly Safety Score — One row per neighbourhood per year.

Loads Open_Consolidated_Data_updated_deduplicated.csv, computes a
severity- and recency-weighted risk score per neighbourhood per year,
and normalises to a 0-100 safety index so Power BI year slicers work.

Output CSV : Open_Consolidated_Neighbourhood_Yearly_Safety.csv
DB table   : crime_neighbourhood_yearly_safety
"""

import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(_SCRIPT_DIR, "project", "DB_csv")
load_dotenv(os.path.join(_SCRIPT_DIR, "project", ".env"))

INPUT_CSV = os.path.join(CSV_DIR, "Open_Consolidated_Data_updated_deduplicated.csv")
OUTPUT_CSV = os.path.join(CSV_DIR, "Open_Consolidated_Neighbourhood_Yearly_Safety.csv")

# Old aggregated CSV (no year column — useless for Power BI year slicers).
OLD_BREAKDOWN_CSV = os.path.join(CSV_DIR, "Open_Consolidated_Data_Crime_Category_Breakdown.csv")

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}

# ── CSI category severity weights ────────────────────────────────────────────
# Higher weight = more dangerous crime type.
SEVERITY_WEIGHTS = {
    "Assault":         6,
    "Robbery":         5,
    "Break and Enter": 4,
    "Auto Theft":      3,
    "Theft Over":      3,
    "NonMCI":          2,
}

# The 6 categories we keep (everything else is dropped).
VALID_CATEGORIES = list(SEVERITY_WEIGHTS.keys())

# ── Recency weighting ────────────────────────────────────────────────────────
# APPROACH CHOSEN — global date-range recency:
#
#   Each incident gets a recency weight based on how recent its OCC_DATE is
#   relative to the FULL dataset date range [min_date … max_date].
#
#       recency_weight = 0.3 + 0.7 × (occ_date − min_date) / (max_date − min_date)
#
#   This means:
#     • The oldest incident in the dataset gets a weight of 0.3
#     • The most recent incident gets a weight of 1.0
#     • Everything in-between scales linearly
#
#   The weighted incidents are then aggregated into neighbourhood-year totals.
#
#   Because each row still lands in exactly one (neighbourhood, year) bucket,
#   the final table is one row per neighbourhood per year — compatible with
#   a Power BI year slicer.
RECENCY_FLOOR = 0.3   # minimum weight for the oldest incident
RECENCY_CEIL  = 1.0   # weight for the most recent incident

# ── Required source columns ──────────────────────────────────────────────────
REQUIRED_COLS = ["OCC_DATE", "NEIGHBOURHOOD_158", "CSI_CATEGORY"]

# ── Safety category thresholds ───────────────────────────────────────────────
#   80-100 → Very Safe
#   60-79  → Safe
#   40-59  → Moderate
#   20-39  → Risky
#    0-19  → Very Risky
SAFETY_BINS   = [-0.1, 20, 40, 60, 80, 100.1]
SAFETY_LABELS = ["Very Risky", "Risky", "Moderate", "Safe", "Very Safe"]


# ── SQL ───────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS crime_neighbourhood_yearly_safety;
CREATE TABLE crime_neighbourhood_yearly_safety (
    neighbourhood_158       TEXT,
    occ_year                INTEGER,
    assault_incidents       INTEGER,
    auto_theft_incidents    INTEGER,
    break_and_enter_incidents INTEGER,
    nonmci_incidents        INTEGER,
    robbery_incidents       INTEGER,
    theft_over_incidents    INTEGER,
    total_incidents         INTEGER,
    weighted_risk_score     REAL,
    safety_score            REAL,
    safety_rank             INTEGER,
    safety_category         TEXT,
    dominant_crime_category TEXT,
    PRIMARY KEY (neighbourhood_158, occ_year)
);
"""

INSERT_SQL = """
INSERT INTO crime_neighbourhood_yearly_safety
    (neighbourhood_158, occ_year,
     assault_incidents, auto_theft_incidents, break_and_enter_incidents,
     nonmci_incidents, robbery_incidents, theft_over_incidents,
     total_incidents, weighted_risk_score,
     safety_score, safety_rank, safety_category, dominant_crime_category)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Load & validate
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate(path: str) -> pd.DataFrame:
    print(f"Loading CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows loaded : {len(df):,}")
    print(f"  Columns     : {list(df.columns)}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR — missing required columns: {missing}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Parse dates, derive occ_year, filter categories
# ─────────────────────────────────────────────────────────────────────────────
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse OCC_DATE → datetime
    df["occ_date"] = pd.to_datetime(df["OCC_DATE"], errors="coerce")
    before = len(df)
    df.dropna(subset=["occ_date"], inplace=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} rows with unparseable OCC_DATE")

    # Derive year
    df["occ_year"] = df["occ_date"].dt.year

    # Keep only 2014–2025 to match the traffic dataset range
    df = df[df["occ_year"].between(2014, 2025)]
    print(f"  Rows after year filter (2014-2025) : {len(df):,}")

    # Drop rows without a valid neighbourhood
    df = df[df["NEIGHBOURHOOD_158"].notna() & (df["NEIGHBOURHOOD_158"] != "NSA")]

    # Standardise CSI_CATEGORY and keep only the 6 valid categories
    df["csi_category"] = df["CSI_CATEGORY"].str.strip()
    before = len(df)
    df = df[df["csi_category"].isin(VALID_CATEGORIES)]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with unrecognised CSI_CATEGORY")

    print(f"  Rows after filtering : {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Compute recency weight per incident
# ─────────────────────────────────────────────────────────────────────────────
def add_recency_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each incident a recency weight in [RECENCY_FLOOR, RECENCY_CEIL]
    based on its position in the full dataset date range."""
    df = df.copy()
    min_date = df["occ_date"].min()
    max_date = df["occ_date"].max()
    date_span = (max_date - min_date).total_seconds() or 1.0

    df["recency_weight"] = (
        RECENCY_FLOOR
        + (RECENCY_CEIL - RECENCY_FLOOR)
        * (df["occ_date"] - min_date).dt.total_seconds() / date_span
    )
    print(f"  Recency range : {min_date.date()} → {max_date.date()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Aggregate into neighbourhood × year
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per neighbourhood per year with category counts, weighted risk,
    and dominant crime category."""

    # Map category → severity weight
    df["severity"] = df["csi_category"].map(SEVERITY_WEIGHTS)

    # Per-incident weighted risk = severity × recency
    df["incident_risk"] = df["severity"] * df["recency_weight"]

    # Pivot counts per category
    cat_counts = (
        df.groupby(["NEIGHBOURHOOD_158", "occ_year", "csi_category"])
          .size()
          .unstack(fill_value=0)
    )

    # Ensure all 6 category columns exist (fill missing with 0)
    for cat in VALID_CATEGORIES:
        if cat not in cat_counts.columns:
            cat_counts[cat] = 0

    # Rename to snake_case column names
    rename_map = {
        "Assault":         "assault_incidents",
        "Auto Theft":      "auto_theft_incidents",
        "Break and Enter": "break_and_enter_incidents",
        "NonMCI":          "nonmci_incidents",
        "Robbery":         "robbery_incidents",
        "Theft Over":      "theft_over_incidents",
    }
    cat_counts = cat_counts.rename(columns=rename_map)[list(rename_map.values())]
    cat_counts = cat_counts.reset_index()
    cat_counts.rename(columns={"NEIGHBOURHOOD_158": "neighbourhood_158"}, inplace=True)

    # Total incidents
    incident_cols = list(rename_map.values())
    cat_counts["total_incidents"] = cat_counts[incident_cols].sum(axis=1)

    # Weighted risk score per neighbourhood-year
    risk = (
        df.groupby(["NEIGHBOURHOOD_158", "occ_year"])["incident_risk"]
          .sum()
          .reset_index()
          .rename(columns={"NEIGHBOURHOOD_158": "neighbourhood_158",
                           "incident_risk": "weighted_risk_score"})
    )

    # Merge
    result = cat_counts.merge(risk, on=["neighbourhood_158", "occ_year"], how="left")
    result["weighted_risk_score"] = result["weighted_risk_score"].fillna(0).round(2)

    # ── Dominant crime category ──────────────────────────────────────────
    # The category with the highest incident count in that neighbourhood-year.
    # TIE-BREAK RULE: when two or more categories share the max count, pick
    # the one with the highest severity weight (i.e. more dangerous wins).
    # If still tied, alphabetical order is used as a final deterministic rule.
    severity_order = sorted(SEVERITY_WEIGHTS.keys(),
                            key=lambda c: (-SEVERITY_WEIGHTS[c], c))
    ordered_cols = [rename_map[c] for c in severity_order]

    # idxmax across ordered columns — because we ordered by severity desc,
    # idxmax will naturally pick the first among ties (highest severity).
    result["dominant_crime_category"] = (
        result[ordered_cols]
        .rename(columns={v: k for k, v in rename_map.items()})
        .idxmax(axis=1)
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Safety score, rank, category
# ─────────────────────────────────────────────────────────────────────────────
def compute_safety(result: pd.DataFrame) -> pd.DataFrame:
    # Min-max normalise weighted_risk_score WITHIN each year → safety_score
    # Higher risk → lower safety.  100 = safest, 0 = least safe.
    yr_min = result.groupby("occ_year")["weighted_risk_score"].transform("min")
    yr_max = result.groupby("occ_year")["weighted_risk_score"].transform("max")
    denom = (yr_max - yr_min).replace(0, np.nan)
    result["safety_score"] = (
        (1 - (result["weighted_risk_score"] - yr_min) / denom) * 100
    ).round(1)
    result["safety_score"] = result["safety_score"].fillna(50.0)

    # Rank within each year (1 = safest)
    result["safety_rank"] = (
        result.groupby("occ_year")["safety_score"]
              .rank(ascending=False, method="min")
              .astype(int)
    )

    # Categorical label
    result["safety_category"] = pd.cut(
        result["safety_score"], bins=SAFETY_BINS, labels=SAFETY_LABELS
    )

    # Sort for readability
    result.sort_values(["occ_year", "safety_rank"], inplace=True, ignore_index=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Write to PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────
def write_to_db(result: pd.DataFrame) -> None:
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)

        db_cols = [
            "neighbourhood_158", "occ_year",
            "assault_incidents", "auto_theft_incidents",
            "break_and_enter_incidents", "nonmci_incidents",
            "robbery_incidents", "theft_over_incidents",
            "total_incidents", "weighted_risk_score",
            "safety_score", "safety_rank", "safety_category",
            "dominant_crime_category",
        ]
        for row in result[db_cols].itertuples(index=False):
            cur.execute(INSERT_SQL, [
                None if pd.isna(v) else (str(v) if isinstance(v, pd.Categorical) else v)
                for v in row
            ])

        conn.commit()
        cur.close()
        print(f"✓ DB table saved:  crime_neighbourhood_yearly_safety  ({len(result):,} rows)")
    except Exception as exc:
        conn.rollback()
        raise exc
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Remove old aggregated CSV
# ─────────────────────────────────────────────────────────────────────────────
def remove_old_csv(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)
        print(f"  Removed old CSV: {path}")
    else:
        print(f"  Old CSV not found (already removed): {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CRIME YEARLY SAFETY — neighbourhood × year")
    print("=" * 70)

    # Load
    df = load_and_validate(INPUT_CSV)

    # Prepare
    df = prepare(df)

    # Recency
    df = add_recency_weight(df)

    # Aggregate
    result = aggregate(df)

    # Safety score / rank / category
    result = compute_safety(result)

    # ── Preview ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("PREVIEW (first 20 rows)")
    print(f"{'─' * 70}")
    print(result.head(20).to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────────────────
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved: {OUTPUT_CSV}  ({len(result):,} rows)")

    # ── Write to PostgreSQL ───────────────────────────────────────────────
    write_to_db(result)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("SUMMARY")
    print(f"{'─' * 70}")
    print(f"  Rows in output          : {len(result):,}")
    print(f"  Unique neighbourhoods   : {result['neighbourhood_158'].nunique()}")
    print(f"  Min occ_year            : {result['occ_year'].min()}")
    print(f"  Max occ_year            : {result['occ_year'].max()}")
    cats = result["safety_category"].value_counts().sort_index()
    print(f"  Category distribution   :")
    for cat, n in cats.items():
        print(f"    {str(cat):12s} : {n:,} rows")

    # ── Example: one neighbourhood across years ──────────────────────────
    print(f"\n{'─' * 70}")
    print("EXAMPLE — Moss Park (73) across years")
    print(f"{'─' * 70}")
    example = result[result["neighbourhood_158"] == "Moss Park (73)"]
    if example.empty:
        # Fall back to first neighbourhood alphabetically
        first = result["neighbourhood_158"].sort_values().iloc[0]
        print(f"  (Moss Park not found, showing {first} instead)")
        example = result[result["neighbourhood_158"] == first]
    cols = [
        "neighbourhood_158", "occ_year", "total_incidents",
        "weighted_risk_score", "safety_score", "safety_rank",
        "safety_category", "dominant_crime_category",
    ]
    print(example[cols].to_string(index=False))

    print(f"\n{'=' * 70}")
    print("DONE — Import the CSV into Power BI; occ_year supports year slicers.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
