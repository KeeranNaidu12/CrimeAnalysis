import os
import csv
from datetime import datetime
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

CSV_PATH = Path('project') / 'DB_csv' / 'Traffic_Collisions_Data_enhanced.csv'

CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS traffic_collisions_data;
CREATE TABLE traffic_collisions_data (
    event_unique_id   TEXT,
    occ_date          TIMESTAMP,
    fatalities        INTEGER,
    injury_collisions BOOLEAN,
    automobile        BOOLEAN,
    motorcycle        BOOLEAN,
    passenger         BOOLEAN,
    bicycle           BOOLEAN,
    pedestrian        BOOLEAN,
    neighbourhood_158 TEXT,
    ftr_collisions    BOOLEAN,
    pd_collisions     BOOLEAN,
    week_day          TEXT,
    season            TEXT,
    holiday           BOOLEAN
);
"""

INSERT_SQL = """
INSERT INTO traffic_collisions_data (
    event_unique_id, occ_date, fatalities, injury_collisions,
    automobile, motorcycle, passenger, bicycle, pedestrian,
    neighbourhood_158, ftr_collisions, pd_collisions,
    week_day, season, holiday
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""


def parse_date(value: str):
    """Parse date string; return None if empty or unparseable."""
    value = value.strip()
    if not value:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    print(f"  Warning: could not parse date '{value}', storing as NULL")
    return None


def parse_int(value: str):
    """Parse float-encoded integers like '1.0'; return None if empty."""
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def parse_bool(value: str):
    """Convert YES/NO strings to booleans; return None if empty."""
    value = value.strip().upper()
    if not value:
        return None
    return value == 'YES'


def load_csv(path: Path) -> list[tuple]:
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for line_num, row in enumerate(reader, start=2):
            try:
                record = (
                    row['EVENT_UNIQUE_ID'].strip() or None,
                    parse_date(row['OCC_DATE']),
                    parse_int(row['FATALITIES']),
                    parse_bool(row['INJURY_COLLISIONS']),
                    parse_bool(row['AUTOMOBILE']),
                    parse_bool(row['MOTORCYCLE']),
                    parse_bool(row['PASSENGER']),
                    parse_bool(row['BICYCLE']),
                    parse_bool(row['PEDESTRIAN']),
                    row['NEIGHBOURHOOD_158'].strip() or None,
                    parse_bool(row['FTR_COLLISIONS']),
                    parse_bool(row['PD_COLLISIONS']),
                    row['week_day'].strip() or None,
                    row['season'].strip() or None,
                    parse_bool(row['holiday']),
                )
                rows.append(record)
            except KeyError as e:
                print(f"  Warning: missing column {e} on line {line_num}, skipping row")
    return rows


def main():
    print(f"Reading CSV: {CSV_PATH}")
    rows = load_csv(CSV_PATH)
    print(f"  Parsed {len(rows)} rows")

    print("Connecting to PostgreSQL …")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            print("Dropping existing table (if any) and recreating …")
            cur.execute(CREATE_TABLE_SQL)

            print("Inserting rows …")
            CHUNK = 500
            with tqdm(total=len(rows), unit='rows', desc='Inserting') as pbar:
                for i in range(0, len(rows), CHUNK):
                    cur.executemany(INSERT_SQL, rows[i:i + CHUNK])
                    pbar.update(min(CHUNK, len(rows) - i))
            conn.commit()
            print(f"Done — {len(rows)} rows inserted.")
    except Exception as exc:
        conn.rollback()
        print(f"Error: {exc}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()