"""
Crime Category Long-Format Table — one row per neighbourhood × year × category.

Reads the wide-format Open_Consolidated_Neighbourhood_Yearly_Safety.csv and
melts the six category count columns into a long table for Power BI slicers.

Output CSV : Open_Consolidated_Neighbourhood_Yearly_Category_Long.csv
DB table   : open_consolidated_neighbourhood_yearly_category_long
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

INPUT_CSV = os.path.join(CSV_DIR, "Open_Consolidated_Neighbourhood_Yearly_Safety.csv")
OUTPUT_CSV = os.path.join(CSV_DIR, "Open_Consolidated_Neighbourhood_Yearly_Category_Long.csv")

DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}

# ── Column mapping: wide column name → readable crime_category label ─────────
CATEGORY_MAP = {
    "assault_incidents":         "Assault",
    "auto_theft_incidents":      "Auto Theft",
    "break_and_enter_incidents": "Break and Enter",
    "nonmci_incidents":          "NonMCI",
    "robbery_incidents":         "Robbery",
    "theft_over_incidents":      "Theft Over",
}

# ── All columns we expect in the source CSV ──────────────────────────────────
REQUIRED_COLS = [
    "neighbourhood_158", "occ_year",
    *CATEGORY_MAP.keys(),
    "total_incidents", "weighted_risk_score",
    "safety_score", "safety_rank", "safety_category", "dominant_crime_category",
]

# Columns that should be carried along (not melted)
ID_COLS = [
    "neighbourhood_158", "occ_year",
    "total_incidents", "weighted_risk_score",
    "safety_score", "safety_rank", "safety_category", "dominant_crime_category",
]

# ── SQL ───────────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS open_consolidated_neighbourhood_yearly_category_long;
CREATE TABLE open_consolidated_neighbourhood_yearly_category_long (
    neighbourhood_158       TEXT,
    occ_year                INTEGER,
    crime_category          TEXT,
    incidents               INTEGER,
    total_incidents         INTEGER,
    category_share          REAL,
    weighted_risk_score     REAL,
    safety_score            REAL,
    safety_rank             INTEGER,
    safety_category         TEXT,
    dominant_crime_category TEXT,
    PRIMARY KEY (neighbourhood_158, occ_year, crime_category)
);
"""

INSERT_SQL = """
INSERT INTO open_consolidated_neighbourhood_yearly_category_long
    (neighbourhood_158, occ_year, crime_category, incidents, total_incidents,
     category_share, weighted_risk_score, safety_score, safety_rank,
     safety_category, dominant_crime_category)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Load & validate
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate(path: str) -> pd.DataFrame:
    print(f"Loading CSV: {path}")
    df = pd.read_csv(path)
    print(f"  Rows loaded : {len(df):,}")
    print(f"  Columns     : {list(df.columns)}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR — missing required columns: {missing}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Melt wide → long
# ─────────────────────────────────────────────────────────────────────────────
def melt_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Unpivot the six category count columns into rows."""

    # Step 1: melt the category columns
    long = df.melt(
        id_vars=ID_COLS,
        value_vars=list(CATEGORY_MAP.keys()),
        var_name="crime_category_raw",
        value_name="incidents",
    )

    # Step 2: map raw column names to readable labels
    long["crime_category"] = long["crime_category_raw"].map(CATEGORY_MAP)
    long.drop(columns=["crime_category_raw"], inplace=True)

    # Step 3: compute category_share = incidents / total_incidents
    # If total_incidents is 0 (unlikely but possible), set share to 0 to avoid division by zero.
    long["category_share"] = np.where(
        long["total_incidents"] > 0,
        (long["incidents"] / long["total_incidents"]).round(4),
        0.0,
    )

    # Step 4: keep zero-incident rows
    # WHY: Power BI slicers work best when every neighbourhood-year has all 6
    # categories present.  Dropping zeros would create gaps that break
    # cross-filtering and make bar charts show misleading category totals.

    # Step 5: sort for readability and deterministic output
    long.sort_values(
        ["occ_year", "neighbourhood_158", "crime_category"],
        inplace=True,
        ignore_index=True,
    )

    # Step 6: reorder columns to match the requested output schema
    col_order = [
        "neighbourhood_158", "occ_year", "crime_category", "incidents",
        "total_incidents", "category_share", "weighted_risk_score",
        "safety_score", "safety_rank", "safety_category", "dominant_crime_category",
    ]
    long = long[col_order]

    return long


# ─────────────────────────────────────────────────────────────────────────────
#  Write to PostgreSQL
# ─────────────────────────────────────────────────────────────────────────────
def write_to_db(result: pd.DataFrame) -> None:
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)

        db_cols = [
            "neighbourhood_158", "occ_year", "crime_category", "incidents",
            "total_incidents", "category_share", "weighted_risk_score",
            "safety_score", "safety_rank", "safety_category", "dominant_crime_category",
        ]
        for row in result[db_cols].itertuples(index=False):
            cur.execute(INSERT_SQL, [
                None if pd.isna(v) else (str(v) if isinstance(v, pd.Categorical) else v)
                for v in row
            ])

        conn.commit()
        cur.close()
        print(f"✓ DB table saved:  open_consolidated_neighbourhood_yearly_category_long  ({len(result):,} rows)")
    except Exception as exc:
        conn.rollback()
        raise exc
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CRIME CATEGORY LONG FORMAT — neighbourhood × year × category")
    print("=" * 70)

    # Load
    df = load_and_validate(INPUT_CSV)

    # Melt
    long = melt_to_long(df)

    # ── Preview ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("PREVIEW (first 20 rows)")
    print(f"{'─' * 70}")
    print(long.head(20).to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────────────────
    long.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved: {OUTPUT_CSV}  ({len(long):,} rows)")

    # ── Write to DB ───────────────────────────────────────────────────────
    write_to_db(long)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("SUMMARY")
    print(f"{'─' * 70}")
    print(f"  Rows in output          : {len(long):,}")
    print(f"  Unique neighbourhoods   : {long['neighbourhood_158'].nunique()}")
    print(f"  Min occ_year            : {long['occ_year'].min()}")
    print(f"  Max occ_year            : {long['occ_year'].max()}")
    print(f"  Distinct crime_category : {sorted(long['crime_category'].unique())}")

    # ── Example: one neighbourhood, one year, all 6 categories ────────────
    print(f"\n{'─' * 70}")
    print("EXAMPLE — Moss Park (73), 2024")
    print(f"{'─' * 70}")
    example = long[
        (long["neighbourhood_158"] == "Moss Park (73)")
        & (long["occ_year"] == 2024)
    ]
    if example.empty:
        first_n = long["neighbourhood_158"].sort_values().iloc[0]
        first_y = long["occ_year"].min()
        print(f"  (Moss Park 2024 not found, showing {first_n} {first_y})")
        example = long[
            (long["neighbourhood_158"] == first_n) & (long["occ_year"] == first_y)
        ]
    print(example.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("DONE — Import into Power BI for crime_category slicers.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
