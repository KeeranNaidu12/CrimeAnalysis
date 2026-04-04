import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Higher weights indicates that the crime is more dangerous and a lower score indicates that a crime is less dangerous.
SEVERITY_WEIGHTS = {
    'Assault' : 6,
    'NonMCI' : 5,
    'Break and Enter' : 4,
    'Auto Theft': 3,
    'Robbery' : 2,
    'Theft Over' : 1,
}

# Recency weights by year band
RECENCY_WEIGHTS = {
    2025: 1.0, 2024: 1.0,
    2023: 0.7, 2022: 0.7,
    2021: 0.5, 2020: 0.5,
}
DEFAULT_RECENCY = 0.3

# Opens a connection to the database using credentials from .env.
def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Calculates a weighted danger score per neighbourhood using severity and recency.
def fetch_crime_scores(conn) -> dict[str, float]:
    case_severity = ' '.join(
        f"WHEN '{cat}' THEN {w}" for cat, w in SEVERITY_WEIGHTS.items()
    )
    case_recency = ' '.join(
        f"WHEN {year} THEN {w}" for year, w in RECENCY_WEIGHTS.items()
    )
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT neighbourhood_158,
               SUM(
                   CASE csi_category {case_severity} ELSE 1 END
                   * CASE EXTRACT(YEAR FROM occ_date)::INT {case_recency} ELSE {DEFAULT_RECENCY} END
               ) AS score
        FROM open_consolidated_data
        WHERE neighbourhood_158 IS NOT NULL
          AND neighbourhood_158 <> 'NSA'
          AND csi_category IS NOT NULL
          AND occ_date IS NOT NULL
        GROUP BY neighbourhood_158
    """)
    rows = cursor.fetchall()
    cursor.close()
    return {neighbourhood: float(score) for neighbourhood, score in rows}

# Min-max normalises danger scores into a 0–100 safety index (100 = safest).
def compute_safety_rankings(crime_scores: dict[str, float]) -> list[tuple[str, float]]:
    # 100 = safest, 0 = most dangerous
    lo, hi = min(crime_scores.values()), max(crime_scores.values())
    span = hi - lo or 1.0
    return sorted(
        ((n, round(100.0 * (1.0 - (d - lo) / span), 2)) for n, d in crime_scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )

# Prints the ranked results as a formatted table.
def print_rankings(ranked: list[tuple[str, float]]) -> None:
    print(f"\n{'Rank':<6} {'Neighbourhood':<50} {'Safety Score (0–100)'}")
    print("-" * 78)
    for rank, (neighbourhood, score) in enumerate(ranked, start=1):
        print(f"{rank:<6} {neighbourhood:<50} {score:.2f}")


def main():
    print("Connecting to database…")
    conn = get_db_connection()
    if not conn:
        return

    try:
        print("Fetching crime data…")
        crime_scores = fetch_crime_scores(conn)
        print(f"  {len(crime_scores)} neighbourhoods with crime data.")

        print("Computing safety rankings…")
        ranked = compute_safety_rankings(crime_scores)

        print(f"\nTotal neighbourhoods ranked: {len(ranked)}")
        print_rankings(ranked)
    finally:
        conn.close()


if __name__ == '__main__':
    main()

