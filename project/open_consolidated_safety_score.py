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
    'Robbery' : 5,
    'Break and Enter' : 4,
    'Auto Theft': 3,
    'NonMCI' : 2,
    'Theft Over' : 1,
}

# Since cirmes will be geenralized if we are going by CSI numbers, we also need to consider the offence types.
# The range for this weighting is 0-2. 0 being less severe to 2 being extremely severe
OFFENCE_WEIGHTS = {
    # Assault (most severe) - firearms and lethal force
    'Discharge Firearm With Intent'    : 2.0,
    'Discharge Firearm - Recklessly'   : 2.0,
    'Use Firearm / Immit Commit Off'   : 2.0,
    'Pointing A Firearm'               : 2.0,
    'Aggravated Assault'               : 2.0,
    'Aggravated Aslt Peace Officer'    : 2.0,

    # Assault - severe - weapons/bodily harm
    'Hoax Terrorism Causing Bodily'    : 1.8,
    'Set/Place Trap/Intend Death/Bh'   : 1.8,
    'Traps Likely Cause Bodily Harm'   : 1.6,
    'Assault With Weapon'              : 1.6,
    'Assault Bodily Harm'              : 1.6,
    'Aggravated Assault Avails Pros'   : 1.6,
    'Air Gun Or Pistol: Bodily Harm'   : 1.6,

    # Assault — concerning
    'Assault Peace Officer Wpn/Cbh'    : 1.5,
    'Crim Negligence Bodily Harm'      : 1.4,
    'Unlawfully Causing Bodily Harm'   : 1.4,
    'Administering Noxious Thing'      : 1.3,

    # Assault — lower severity variants
    'Assault Peace Officer'            : 1.1,
    'Disarming Peace/Public Officer'   : 1.0,
    'Assault - Resist/ Prevent Seiz'   : 0.70,
    'Assault - Force/Thrt/Impede'      : 0.60,

    # Robbery — firearm acquisition
    'Robbery To Steal Firearm'         : 2.0,
    # Robbery — severe (invasion / carjacking / weapon)
    'Robbery - Home Invasion'          : 1.8,
    'Robbery - Vehicle Jacking'        : 1.7,
    'Robbery With Weapon'              : 1.6,
    # Robbery — high (organised / institutional / group)
    'Robbery - Financial Institute'    : 1.4,
    'Robbery - Armoured Car'           : 1.4,
    'Robbery - Swarming'               : 1.4,
    # Robbery — medium (confrontational / targeted)
    'Robbery - Business'               : 1.2,
    'Robbery - Mugging'                : 1.1,
    'Robbery - Delivery Person'        : 1.1,
    'Robbery - Taxi'                   : 1.0,
    'Robbery - Other'                  : 1.0,
    # Robbery — lower (opportunistic)
    'Robbery - Purse Snatch'           : 0.8,
    'Robbery - Atm'                    : 0.7,

    # Break and Enter
    'B&E - To Steal Firearm'           : 2.0,
    'B&E - M/Veh To Steal Firearm'     : 1.7,
    'B&E W\'Intent'                    : 1.1,
    'Unlawfully In Dwelling-House'     : 0.8,
    'B&E Out'                          : 0.6,

    # Theft Over
    'Theft From Motor Vehicle Over'    : 2.0,
    'Theft From Mail / Bag / Key'      : 1.3,
    'Theft Over - Distraction'         : 0.5,
    'Theft Over - Shoplifting'         : 0.6,
    'Theft Over - Bicycle'             : 0.4,
    'Theft - Misapprop Funds Over'     : 0.3,
    'Theft Of Utilities Over'          : 0.3,

    # NonMCI
    'Theft From Motor Vehicle Under'   : 0.7,
}
DEFAULT_OFFENCE = 1.0

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

# Calculates a weighted danger score per neighbourhood using severity, offence sub-weight, recency and year frequency.
def fetch_crime_scores(conn) -> dict[str, float]:
    case_severity = ' '.join(
        f"WHEN '{cat}' THEN {w}" for cat, w in SEVERITY_WEIGHTS.items()
    )
    case_offence = ' '.join(
        "WHEN '{}' THEN {}".format(off.replace("'", "''"), w) for off, w in OFFENCE_WEIGHTS.items()
    )
    case_recency = ' '.join(
        f"WHEN {year} THEN {w}" for year, w in RECENCY_WEIGHTS.items()
    )
    cursor = conn.cursor()
    cursor.execute(f"""
        WITH offence_year_counts AS (
            SELECT offence,
                   EXTRACT(YEAR FROM occ_date)::INT AS yr,
                   COUNT(*) AS cnt
            FROM open_consolidated_data
            WHERE neighbourhood_158 IS NOT NULL
              AND neighbourhood_158 <> 'NSA'
              AND csi_category IS NOT NULL
              AND occ_date IS NOT NULL
            GROUP BY offence, yr
        ),
        offence_year_weighted AS (
            -- Apply recency weight to raw count before normalising
            SELECT offence, yr,
                   cnt::FLOAT * CASE yr {case_recency} ELSE {DEFAULT_RECENCY} END AS weighted_cnt
            FROM offence_year_counts
        ),
        offence_year_freq AS (
            -- Normalise recency-weighted count 0-1 globally across all offence-year pairs
            SELECT offence, yr,
                   (weighted_cnt - MIN(weighted_cnt) OVER ())
                   / NULLIF(MAX(weighted_cnt) OVER () - MIN(weighted_cnt) OVER (), 0)
                   AS norm_freq
            FROM offence_year_weighted
        )
        SELECT o.neighbourhood_158,
               SUM(
                   CASE o.csi_category {case_severity} ELSE 1 END
                   * CASE o.offence {case_offence} ELSE {DEFAULT_OFFENCE} END
                   * COALESCE(oyf.norm_freq, 0)
               ) AS score
        FROM open_consolidated_data o
        LEFT JOIN offence_year_freq oyf
               ON o.offence = oyf.offence
              AND EXTRACT(YEAR FROM o.occ_date)::INT = oyf.yr
        WHERE o.neighbourhood_158 IS NOT NULL
          AND o.neighbourhood_158 <> 'NSA'
          AND o.csi_category IS NOT NULL
          AND o.occ_date IS NOT NULL
        GROUP BY o.neighbourhood_158
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


# Fetches incident counts per neighbourhood and crime category (all years).
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

# Exports a wide-format CSV: one row per neighbourhood with rank, safety score,
# per-category incident counts (all years), and total incidents (all years).
def export_category_breakdown(
    rows: list[tuple[str, str, int]],
    ranked: list[tuple[str, float]],
    path: str = "Open_Consolidated_Data_Crime_Category_Breakdown.csv",
) -> None:
    categories = ['Assault', 'Auto Theft', 'Break and Enter', 'NonMCI', 'Robbery', 'Theft Over']

    # Build lookup: neighbourhood -> {category: count}
    counts: dict[str, dict[str, int]] = {}
    for neighbourhood, category, count in rows:
        counts.setdefault(neighbourhood, {})[category] = count

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Neighbourhood', 'Rank', 'Safety Score'] + categories + ['Total Incidents'])
        for rank, (neighbourhood, score) in enumerate(ranked, start=1):
            cat_counts = counts.get(neighbourhood, {})
            row_counts = [cat_counts.get(cat, 0) for cat in categories]
            total = sum(row_counts)
            writer.writerow([neighbourhood, rank, round(score, 2)] + row_counts + [total])

    print(f"Category breakdown exported to {path}")


# Fetches and prints each offence's recency-weighted frequency total and its normalised 0–1 value
# (this is the freq factor used in the scoring formula).
def print_offence_year_table(conn) -> None:
    cursor = conn.cursor()
    case_recency = ' '.join(
        f"WHEN {year} THEN {w}" for year, w in RECENCY_WEIGHTS.items()
    )
    cursor.execute(f"""
        WITH offence_year_counts AS (
            SELECT offence,
                   EXTRACT(YEAR FROM occ_date)::INT AS yr,
                   COUNT(*) AS cnt
            FROM open_consolidated_data
            WHERE offence IS NOT NULL
            GROUP BY offence, yr
        ),
        offence_weighted_freq AS (
            SELECT offence,
                   SUM(cnt * CASE yr {case_recency} ELSE {DEFAULT_RECENCY} END) AS weighted_total
            FROM offence_year_counts
            GROUP BY offence
        )
        SELECT offence,
               ROUND(weighted_total::NUMERIC, 2) AS weighted_total,
               ROUND(
                   ((weighted_total - MIN(weighted_total) OVER ())
                   / NULLIF(MAX(weighted_total) OVER () - MIN(weighted_total) OVER (), 0))::NUMERIC
               , 4) AS norm_freq
        FROM offence_weighted_freq
        ORDER BY weighted_total DESC
    """)
    rows = cursor.fetchall()
    cursor.close()

    print('\nOffence Recency-Weighted Frequency (used as freq factor in scoring formula)')
    print(f"{'Offence':<45} {'WeightedTotal':>14} {'NormFreq (0–1)':>15}")
    print('-' * 76)
    for offence, weighted_total, norm_freq in rows:
        print(f"{str(offence):<45} {float(weighted_total):>14.2f} {float(norm_freq):>15.4f}")


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

# querying all offences per csi   

def print_offences_by_csi(conn) -> None:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT csi_category, offence, COUNT(*) AS total
        FROM open_consolidated_data
        WHERE csi_category IS NOT NULL
          AND offence IS NOT NULL
        GROUP BY csi_category, offence
        ORDER BY csi_category, COUNT(*) DESC
    """)
    rows = cursor.fetchall()
    cursor.close()

    current_category = None
    print('\nAll Offences by CSI Category')
    print('=' * 70)
    for csi_category, offence, total in rows:
        if csi_category != current_category:
            current_category = csi_category
            print(f'\n[{csi_category}]')
            print('-' * 70)
        print(f"  {str(offence):<50} {total:>8,}")


def main():
    print("Connecting to database…")
    conn = get_db_connection()
    if not conn:
        return

    try:
        print("Fetching all offences by CSI category…")
        print_offences_by_csi(conn)

        print("Fetching crime data…")
        crime_scores = fetch_crime_scores(conn)
        print(f"  {len(crime_scores)} neighbourhoods with crime data.")

        print("Computing safety rankings…")
        ranked = compute_safety_rankings(crime_scores)

        print(f"\nTotal neighbourhoods ranked: {len(ranked)}")
        print_rankings(ranked)

        print("Fetching crime category breakdown…")
        breakdown = fetch_crime_category_breakdown(conn)
        export_category_breakdown(breakdown, ranked)

        print("Fetching offence frequency by year…")
        print_offence_year_table(conn)

        plot_rankings(ranked)
    finally:
        conn.close()


if __name__ == '__main__':
    main()

