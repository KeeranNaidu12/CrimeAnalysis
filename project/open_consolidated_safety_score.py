import csv
import os

import matplotlib.pyplot as plt
import psycopg2
import seaborn as sns
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


# Returns the safety zone label for a given score.
def _zone(score: float) -> str:
    if score < 34:
        return 'Dangerous'
    if score < 67:
        return 'Neutral'
    return 'Safe'

# Exports the ranked results with zone labels to a CSV file.
def export_rankings(ranked: list[tuple[str, float]], path: str = "Open_Consolidated_Data_Safety_score_ranking.csv") -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Neighbourhood', 'Safety Score (0-100)', 'Zone'])
        for rank, (neighbourhood, score) in enumerate(ranked, start=1):
            writer.writerow([rank, neighbourhood, score, _zone(score)])
    print(f"Rankings exported to {path}")

# Fetches incident counts per neighbourhood and crime category.
def fetch_crime_category_breakdown(conn) -> list[tuple[str, str, int]]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT neighbourhood_158, csi_category, COUNT(*) AS incident_count
        FROM open_consolidated_data
        WHERE neighbourhood_158 IS NOT NULL
          AND neighbourhood_158 <> 'NSA'
          AND csi_category IS NOT NULL
        GROUP BY neighbourhood_158, csi_category
        ORDER BY neighbourhood_158, csi_category
    """)
    rows = cursor.fetchall()
    cursor.close()
    return rows

# Exports the crime category breakdown to a CSV file.
def export_category_breakdown(rows: list[tuple[str, str, int]], path: str = "Open_Consolidated_Data_Crime_Category_Breakdown.csv") -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Neighbourhood', 'Crime Category', 'Incident Count'])
        writer.writerows(rows)
    print(f"Category breakdown exported to {path}")


# Plots top 15 safest, top 15 most dangerous, and all neighbourhoods as separate charts.
def plot_rankings(ranked: list[tuple[str, float]]) -> None:
    sns.set_theme(style='whitegrid')
    names  = [n for n, _ in ranked]
    scores = [s for _, s in ranked]

    # Top 15 safest
    plt.figure(figsize=(10, 6))
    plt.barh(names[:15][::-1], scores[:15][::-1], color='steelblue')
    plt.title('Top 15 Safest Neighbourhoods')
    plt.xlabel('Safety Score (0–100)')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('safety_top15_safest.png', dpi=150)
    print('Chart saved to safety_top15_safest.png')
    plt.show()

    # Top 15 most dangerous
    plt.figure(figsize=(10, 6))
    plt.barh(names[-15:][::-1], scores[-15:][::-1], color='firebrick')
    plt.title('Top 15 Most Dangerous Neighbourhoods')
    plt.xlabel('Safety Score (0–100)')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('safety_top15_dangerous.png', dpi=150)
    print('Chart saved to safety_top15_dangerous.png')
    plt.show()

    # Zone breakdown: how many neighbourhoods are in each safety zone
    red    = sum(1 for s in scores if s < 34)
    yellow = sum(1 for s in scores if 34 <= s < 67)
    green  = sum(1 for s in scores if s >= 67)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Dangerous\n(0–33)', 'Neutral\n(34–66)', 'Safe\n(67–100)'],
           [red, yellow, green],
           color=['firebrick', 'gold', 'seagreen'])
    ax.set_title('Neighbourhoods by Safety Zone')
    ax.set_ylabel('Number of Neighbourhoods')
    for i, count in enumerate([red, yellow, green]):
        ax.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
    fig.tight_layout()
    fig.savefig('safety_all_neighbourhoods.png', dpi=150)
    print('Chart saved to safety_all_neighbourhoods.png')
    plt.show()


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
        export_rankings(ranked)

        print("Fetching crime category breakdown…")
        breakdown = fetch_crime_category_breakdown(conn)
        export_category_breakdown(breakdown)

        plot_rankings(ranked)
    finally:
        conn.close()


if __name__ == '__main__':
    main()

