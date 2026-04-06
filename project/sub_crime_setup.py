import os
import re
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

SOURCE_TABLE = 'open_consolidated_data'


def slugify(name: str) -> str:
    """Convert a CSI category name to a safe PostgreSQL table name."""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]+', '_', name)   # replace non-alphanumeric with _
    name = name.strip('_')
    return f"csi_{name}"


def main():
    print("Connecting to PostgreSQL …")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:

            # ----------------------------------------------------------------
            # 1. Discover all distinct CSI categories in the source table
            # ----------------------------------------------------------------
            print(f"Fetching distinct CSI categories from '{SOURCE_TABLE}' …")
            cur.execute(f"""
                SELECT DISTINCT csi_category
                FROM {SOURCE_TABLE}
                WHERE csi_category IS NOT NULL AND csi_category <> ''
                ORDER BY csi_category;
            """)
            categories = [row[0] for row in cur.fetchall()]
            print(f"  Found {len(categories)} categories: {', '.join(categories)}\n")

            # ----------------------------------------------------------------
            # 2. For each category: drop, recreate, and populate its table
            # ----------------------------------------------------------------
            for category in categories:
                table_name = slugify(category)
                print(f"Processing '{category}' → table '{table_name}' …")

                # Drop & recreate with enhanced columns
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
                cur.execute(f"""
                    CREATE TABLE {table_name} (
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
                """)

                # Count rows for progress bar (2014+ only)
                cur.execute(f"""
                    SELECT COUNT(*) FROM {SOURCE_TABLE}
                    WHERE csi_category = %s
                      AND occ_date >= '2014-01-01';
                """, (category,))
                total = cur.fetchone()[0]

                # Fetch in chunks using a server-side cursor to avoid
                # loading all rows into memory at once
                CHUNK = 1000
                server_cur_name = f"fetch_{table_name}"
                cur.execute(f"""
                    DECLARE {server_cur_name} CURSOR FOR
                    SELECT event_unique_id, occ_date, neighbourhood_158,
                           csi_category, offence, death, injuries,
                           event_type, premise_type, week_day, season, holiday
                    FROM {SOURCE_TABLE}
                    WHERE csi_category = %s
                      AND occ_date >= '2014-01-01';
                """, (category,))

                inserted = 0
                with tqdm(total=total, unit='rows', desc=f'  Inserting') as pbar:
                    while True:
                        cur.execute(f"FETCH {CHUNK} FROM {server_cur_name};")
                        batch = cur.fetchall()
                        if not batch:
                            break
                        cur.executemany(f"""
                            INSERT INTO {table_name} (
                                event_unique_id, occ_date, neighbourhood_158,
                                csi_category, offence, death, injuries,
                                event_type, premise_type, week_day, season, holiday
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """, batch)
                        inserted += len(batch)
                        pbar.update(len(batch))

                cur.execute(f"CLOSE {server_cur_name};")
                print(f"  Done — {inserted} rows inserted into '{table_name}'\n")

            conn.commit()
            print("All category tables created and populated successfully.")

    except Exception as exc:
        conn.rollback()
        print(f"Error: {exc}")
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()