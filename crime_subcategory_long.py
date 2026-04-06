"""
Crime Subcategory Long-Format Table
    — one row per neighbourhood × year × crime_subcategory.

Reads Open_Consolidated_Data_updated_deduplicated.csv and builds a
granular long table where crime_subcategory = CSI_CATEGORY + " - " + OFFENCE.
Attaches yearly safety metrics to every row for Power BI cross-filtering.

Output CSV : Open_Consolidated_Neighbourhood_Yearly_Subcategory_Long.csv
DB table   : open_consolidated_neighbourhood_yearly_subcategory_long
"""

import os
import sys
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

# ── Paths & config ────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(_SCRIPT_DIR, "project", "DB_csv")
load_dotenv(os.path.join(_SCRIPT_DIR, "project", ".env"))

INPUT_CSV = os.path.join(CSV_DIR, "Open_Consolidated_Data_updated_deduplicated.csv")
OUTPUT_CSV = os.path.join(
    CSV_DIR, "Open_Consolidated_Neighbourhood_Yearly_Subcategory_Long.csv"
)

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
VALID_CATEGORIES = list(SEVERITY_WEIGHTS.keys())

# ── Recency weighting ────────────────────────────────────────────────────────
# Each incident gets a recency weight based on where its OCC_DATE falls in the
# full dataset date range [min_date … max_date]:
#
#   recency_weight = 0.5 + 1.0 × (occ_date − min_date) / (max_date − min_date)
#
# Range: 0.5 (oldest) → 1.5 (newest).
# Slightly wider than the 0.3-1.0 range used in the yearly safety table so
# that recency has more influence at the subcategory granularity.
RECENCY_FLOOR = 0.5
RECENCY_CEIL  = 1.5

# ── Safety category thresholds ───────────────────────────────────────────────
SAFETY_BINS   = [-0.1, 20, 40, 60, 80, 100.1]
SAFETY_LABELS = ["Very Risky", "Risky", "Moderate", "Safe", "Very Safe"]

# ── Offence cleaning dictionary ──────────────────────────────────────────────
# Maps raw OFFENCE strings to a single canonical label.
# Expand this dictionary as new inconsistencies are discovered.
# Keys must be the EXACT raw string (after strip/title-case).
OFFENCE_ALIAS = {
    # Example: if the data ever contains variant spellings, map them here.
    # "B&E W'intent"  : "B&E W'Intent",
    # "Robbery - Atm" : "Robbery - ATM",
    #
    # Current dataset (52 unique offences) has no duplicates after title-case
    # normalisation, but the hook is here for future use.
}

# ── Required source columns ──────────────────────────────────────────────────
REQUIRED_COLS = [
    "OCC_DATE", "NEIGHBOURHOOD_158", "CSI_CATEGORY", "OFFENCE",
]

# ── SQL ───────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS open_consolidated_neighbourhood_yearly_subcategory_long;
CREATE TABLE open_consolidated_neighbourhood_yearly_subcategory_long (
    neighbourhood_158            TEXT,
    occ_year                     INTEGER,
    csi_category                 TEXT,
    offence                      TEXT,
    crime_subcategory            TEXT,
    incidents                    INTEGER,
    total_incidents              INTEGER,
    subcategory_share            REAL,
    yearly_weighted_risk_score   REAL,
    yearly_safety_score          REAL,
    yearly_safety_rank           INTEGER,
    yearly_safety_category       TEXT,
    dominant_csi_category        TEXT,
    dominant_subcategory         TEXT,
    PRIMARY KEY (neighbourhood_158, occ_year, crime_subcategory)
);
"""

INSERT_SQL = """
INSERT INTO open_consolidated_neighbourhood_yearly_subcategory_long
    (neighbourhood_158, occ_year, csi_category, offence, crime_subcategory,
     incidents, total_incidents, subcategory_share,
     yearly_weighted_risk_score, yearly_safety_score, yearly_safety_rank,
     yearly_safety_category, dominant_csi_category, dominant_subcategory)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Load & validate
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate(path: str) -> pd.DataFrame:
    print(f"Loading CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Rows loaded : {len(df):,}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR — missing required columns: {missing}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Parse dates, filter years, clean text
# ─────────────────────────────────────────────────────────────────────────────
def clean_offence(s: pd.Series) -> pd.Series:
    """Clean OFFENCE strings:
    1. Strip leading/trailing whitespace
    2. Collapse repeated internal whitespace to a single space
    3. Title-case for consistency
    4. Apply the OFFENCE_ALIAS mapping for known duplicates
    """
    cleaned = (
        s.fillna("")
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)  # collapse whitespace
         .str.title()                            # consistent casing
    )
    # Apply alias mapping (expand OFFENCE_ALIAS dict as needed)
    if OFFENCE_ALIAS:
        cleaned = cleaned.replace(OFFENCE_ALIAS)
    return cleaned


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse OCC_DATE → datetime
    df["occ_date"] = pd.to_datetime(df["OCC_DATE"], errors="coerce")
    before = len(df)
    df.dropna(subset=["occ_date"], inplace=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} rows with unparseable OCC_DATE")

    # Derive year and filter to 2014–2025
    df["occ_year"] = df["occ_date"].dt.year
    df = df[df["occ_year"].between(2014, 2025)]
    print(f"  Rows after year filter (2014-2025) : {len(df):,}")

    # Drop rows with no valid neighbourhood
    df = df[df["NEIGHBOURHOOD_158"].notna() & (df["NEIGHBOURHOOD_158"] != "NSA")]

    # Clean CSI_CATEGORY — strip and keep only valid categories
    df["csi_category"] = df["CSI_CATEGORY"].str.strip()
    before = len(df)
    df = df[df["csi_category"].isin(VALID_CATEGORIES)]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with unrecognised CSI_CATEGORY")

    # Clean OFFENCE
    df["offence"] = clean_offence(df["OFFENCE"])

    # Handle missing OFFENCE — use fallback label
    mask_empty = df["offence"] == ""
    if mask_empty.any():
        df.loc[mask_empty, "offence"] = "Unknown Offence"
        print(f"  Filled {mask_empty.sum():,} empty OFFENCE values with 'Unknown Offence'")

    # Build crime_subcategory = "CSI_CATEGORY - OFFENCE"
    df["crime_subcategory"] = df["csi_category"] + " - " + df["offence"]

    print(f"  Rows after all cleaning : {len(df):,}")
    print(f"  Unique offence values   : {df['offence'].nunique()}")
    print(f"  Unique subcategories    : {df['crime_subcategory'].nunique()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Recency weight
# ─────────────────────────────────────────────────────────────────────────────
def add_recency_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Linear recency weight from RECENCY_FLOOR (oldest) to RECENCY_CEIL (newest)."""
    df = df.copy()
    min_date = df["occ_date"].min()
    max_date = df["occ_date"].max()
    span = (max_date - min_date).total_seconds() or 1.0

    df["recency_weight"] = (
        RECENCY_FLOOR
        + (RECENCY_CEIL - RECENCY_FLOOR)
        * (df["occ_date"] - min_date).dt.total_seconds() / span
    )
    print(f"  Recency range : {min_date.date()} → {max_date.date()}  "
          f"(weight {RECENCY_FLOOR}–{RECENCY_CEIL})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Compute yearly neighbourhood-level safety metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_yearly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to neighbourhood-year and compute safety score/rank/category."""

    # Per-incident risk = csi_weight × recency_weight
    df["incident_risk"] = df["csi_category"].map(SEVERITY_WEIGHTS) * df["recency_weight"]

    # ── Neighbourhood-year aggregates ─────────────────────────────────────
    yearly = (
        df.groupby(["NEIGHBOURHOOD_158", "occ_year"])
          .agg(
              total_incidents=("occ_date", "size"),
              yearly_weighted_risk_score=("incident_risk", "sum"),
          )
          .reset_index()
          .rename(columns={"NEIGHBOURHOOD_158": "neighbourhood_158"})
    )
    yearly["yearly_weighted_risk_score"] = yearly["yearly_weighted_risk_score"].round(2)

    # ── Safety score — min-max normalised within each year ────────────────
    yr_min = yearly.groupby("occ_year")["yearly_weighted_risk_score"].transform("min")
    yr_max = yearly.groupby("occ_year")["yearly_weighted_risk_score"].transform("max")
    denom = (yr_max - yr_min).replace(0, np.nan)
    yearly["yearly_safety_score"] = (
        (1 - (yearly["yearly_weighted_risk_score"] - yr_min) / denom) * 100
    ).round(1)
    yearly["yearly_safety_score"] = yearly["yearly_safety_score"].fillna(50.0)

    # ── Rank within year (1 = safest) ─────────────────────────────────────
    yearly["yearly_safety_rank"] = (
        yearly.groupby("occ_year")["yearly_safety_score"]
              .rank(ascending=False, method="min")
              .astype(int)
    )

    # ── Safety category ───────────────────────────────────────────────────
    yearly["yearly_safety_category"] = pd.cut(
        yearly["yearly_safety_score"], bins=SAFETY_BINS, labels=SAFETY_LABELS
    )

    # ── Dominant CSI category per neighbourhood-year ─────────────────────
    # TIE-BREAK: highest severity weight wins; if still tied, alphabetical.
    csi_counts = (
        df.groupby(["NEIGHBOURHOOD_158", "occ_year", "csi_category"])
          .size()
          .reset_index(name="cnt")
          .rename(columns={"NEIGHBOURHOOD_158": "neighbourhood_158"})
    )
    csi_counts["severity"] = csi_counts["csi_category"].map(SEVERITY_WEIGHTS)
    csi_counts.sort_values(
        ["neighbourhood_158", "occ_year", "cnt", "severity", "csi_category"],
        ascending=[True, True, False, False, True],
        inplace=True,
    )
    dominant_csi = (
        csi_counts.groupby(["neighbourhood_158", "occ_year"])
                  .first()
                  .reset_index()[["neighbourhood_158", "occ_year", "csi_category"]]
                  .rename(columns={"csi_category": "dominant_csi_category"})
    )

    # ── Dominant subcategory per neighbourhood-year ───────────────────────
    # TIE-BREAK: highest severity of the parent CSI wins; then alphabetical.
    sub_counts = (
        df.groupby(["NEIGHBOURHOOD_158", "occ_year", "crime_subcategory", "csi_category"])
          .size()
          .reset_index(name="cnt")
          .rename(columns={"NEIGHBOURHOOD_158": "neighbourhood_158"})
    )
    sub_counts["severity"] = sub_counts["csi_category"].map(SEVERITY_WEIGHTS)
    sub_counts.sort_values(
        ["neighbourhood_158", "occ_year", "cnt", "severity", "crime_subcategory"],
        ascending=[True, True, False, False, True],
        inplace=True,
    )
    dominant_sub = (
        sub_counts.groupby(["neighbourhood_158", "occ_year"])
                  .first()
                  .reset_index()[["neighbourhood_158", "occ_year", "crime_subcategory"]]
                  .rename(columns={"crime_subcategory": "dominant_subcategory"})
    )

    # ── Merge dominance fields onto yearly ────────────────────────────────
    yearly = yearly.merge(dominant_csi, on=["neighbourhood_158", "occ_year"], how="left")
    yearly = yearly.merge(dominant_sub, on=["neighbourhood_158", "occ_year"], how="left")

    return yearly


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — Build the subcategory-level long table
# ─────────────────────────────────────────────────────────────────────────────
def build_long_table(df: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to neighbourhood × year × subcategory, then merge yearly metrics."""

    # Count incidents per neighbourhood-year-subcategory
    sub = (
        df.groupby(["NEIGHBOURHOOD_158", "occ_year", "csi_category",
                     "offence", "crime_subcategory"])
          .size()
          .reset_index(name="incidents")
          .rename(columns={"NEIGHBOURHOOD_158": "neighbourhood_158"})
    )

    # Merge yearly metrics
    long = sub.merge(yearly, on=["neighbourhood_158", "occ_year"], how="left")

    # Compute subcategory_share = incidents / total_incidents
    long["subcategory_share"] = np.where(
        long["total_incidents"] > 0,
        (long["incidents"] / long["total_incidents"]).round(4),
        0.0,
    )

    # Reorder columns to match the requested schema
    col_order = [
        "neighbourhood_158", "occ_year", "csi_category", "offence",
        "crime_subcategory", "incidents", "total_incidents", "subcategory_share",
        "yearly_weighted_risk_score", "yearly_safety_score", "yearly_safety_rank",
        "yearly_safety_category", "dominant_csi_category", "dominant_subcategory",
    ]
    long = long[col_order]

    # Sort for deterministic output
    long.sort_values(
        ["occ_year", "neighbourhood_158", "csi_category", "crime_subcategory"],
        inplace=True, ignore_index=True,
    )

    return long


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
            "neighbourhood_158", "occ_year", "csi_category", "offence",
            "crime_subcategory", "incidents", "total_incidents", "subcategory_share",
            "yearly_weighted_risk_score", "yearly_safety_score", "yearly_safety_rank",
            "yearly_safety_category", "dominant_csi_category", "dominant_subcategory",
        ]
        for row in result[db_cols].itertuples(index=False):
            cur.execute(INSERT_SQL, [
                None if pd.isna(v) else (str(v) if isinstance(v, pd.Categorical) else v)
                for v in row
            ])

        conn.commit()
        cur.close()
        print(f"✓ DB table saved:  open_consolidated_neighbourhood_yearly_subcategory_long"
              f"  ({len(result):,} rows)")
    except Exception as exc:
        conn.rollback()
        raise exc
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("CRIME SUBCATEGORY LONG FORMAT — neighbourhood × year × subcategory")
    print("=" * 75)

    # ── Load ──────────────────────────────────────────────────────────────
    df = load_and_validate(INPUT_CSV)

    # ── Clean & prepare ──────────────────────────────────────────────────
    df = prepare(df)

    # ── Recency weights ──────────────────────────────────────────────────
    df = add_recency_weight(df)

    # ── Compute yearly neighbourhood metrics ─────────────────────────────
    yearly = compute_yearly_metrics(df)

    # ── Build long table ─────────────────────────────────────────────────
    long = build_long_table(df, yearly)

    # ── Preview ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 75}")
    print("PREVIEW (first 20 rows)")
    print(f"{'─' * 75}")
    print(long.head(20).to_string(index=False))

    # ── Save CSV ─────────────────────────────────────────────────────────
    long.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved: {OUTPUT_CSV}  ({len(long):,} rows)")

    # ── Write to DB ──────────────────────────────────────────────────────
    write_to_db(long)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 75}")
    print("SUMMARY")
    print(f"{'─' * 75}")
    print(f"  Rows in output              : {len(long):,}")
    print(f"  Unique neighbourhoods       : {long['neighbourhood_158'].nunique()}")
    print(f"  Min occ_year                : {long['occ_year'].min()}")
    print(f"  Max occ_year                : {long['occ_year'].max()}")
    print(f"  Unique csi_category values  : {long['csi_category'].nunique()}")
    print(f"  Unique crime_subcategory    : {long['crime_subcategory'].nunique()}")
    cats = long["yearly_safety_category"].value_counts().sort_index()
    print(f"  Safety category distribution:")
    for cat, n in cats.items():
        print(f"    {str(cat):12s} : {n:,} rows")

    # ── Example: one neighbourhood, one year ─────────────────────────────
    print(f"\n{'─' * 75}")
    print("EXAMPLE — Moss Park (73), 2024")
    print(f"{'─' * 75}")
    ex = long[
        (long["neighbourhood_158"] == "Moss Park (73)")
        & (long["occ_year"] == 2024)
    ]
    if ex.empty:
        first_n = long["neighbourhood_158"].sort_values().iloc[0]
        print(f"  (Moss Park 2024 not found, showing {first_n} 2014)")
        ex = long[
            (long["neighbourhood_158"] == first_n) & (long["occ_year"] == 2014)
        ]
    show_cols = [
        "csi_category", "offence", "crime_subcategory",
        "incidents", "subcategory_share",
        "dominant_csi_category", "dominant_subcategory",
    ]
    print(ex[show_cols].to_string(index=False))

    print(f"\n{'=' * 75}")
    print("DONE — Import into Power BI for subcategory slicers and charts.")
    print(f"{'=' * 75}")

    # ──────────────────────────────────────────────────────────────────────
    # OPTIONAL: write to PostgreSQL using SQLAlchemy (uncomment to use)
    # ──────────────────────────────────────────────────────────────────────
    # from sqlalchemy import create_engine
    # engine = create_engine(
    #     f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    #     f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    # )
    # long.to_sql(
    #     "open_consolidated_neighbourhood_yearly_subcategory_long",
    #     engine,
    #     if_exists="replace",
    #     index=False,
    #     method="multi",
    #     chunksize=1000,
    # )
    # print("✓ Written via SQLAlchemy")


if __name__ == "__main__":
    main()
