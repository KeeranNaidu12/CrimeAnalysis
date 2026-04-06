import os
import csv
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'sslmode': os.getenv('DB_SSLMODE', 'prefer')
}

CSV_PATH = Path('project') / 'DB_csv' / 'Open_Consolidated_Data_updated_deduplicated_enhanced.csv'

CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS open_consolidated_data;
CREATE TABLE open_consolidated_data (
    event_unique_id   TEXT,
    occ_date          TIMESTAMP,
    neighbourhood_158 TEXT,
    csi_category      TEXT,
    offence           TEXT,
    death             INTEGER,
    injuries          INTEGER,
    event_type        TEXT,
    premise_type      TEXT,
    week_day          TEXT,
    season            TEXT,
    holiday           BOOLEAN
);
"""

INSERT_SQL = """
INSERT INTO open_consolidated_data (
    event_unique_id, occ_date, neighbourhood_158,
    csi_category, offence, death, injuries, event_type, premise_type,
    week_day, season, holiday
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
    """Convert True/False strings to booleans; return None if empty."""
    value = value.strip().upper()
    if not value:
        return None
    return value == 'TRUE'


def load_csv(path: Path) -> list[tuple]:
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for line_num, row in enumerate(reader, start=2):
            try:
                record = (
                    row['EVENT_UNIQUE_ID'].strip() or None,
                    parse_date(row['OCC_DATE']),
                    row['NEIGHBOURHOOD_158'].strip() or None,
                    row['CSI_CATEGORY'].strip() or None,
                    row['OFFENCE'].strip() or None,
                    parse_int(row['DEATH']),
                    parse_int(row['INJURIES']),
                    row['EVENT_TYPE'].strip() or None,
                    row['PREMISE_TYPE'].strip() or None,
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
            CHUNK = 5000
            values_template = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            insert_query = """INSERT INTO open_consolidated_data (
                event_unique_id, occ_date, neighbourhood_158,
                csi_category, offence, death, injuries, event_type, premise_type,
                week_day, season, holiday
            ) VALUES %s"""
            with tqdm(total=len(rows), unit='rows', desc='Inserting') as pbar:
                for i in range(0, len(rows), CHUNK):
                    batch = rows[i:i + CHUNK]
                    execute_values(cur, insert_query, batch, page_size=CHUNK)
                    pbar.update(len(batch))
            inserted = len(rows)
            conn.commit()
            print(f"Done — {inserted} rows inserted.")
    except Exception as exc:
        conn.rollback()
        print(f"Error: {exc}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()