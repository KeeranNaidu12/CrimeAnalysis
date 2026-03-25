import os
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Database configuration from .env
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Minimum year to include in data loading
MIN_YEAR = 2014

def create_database():
    """Create the Toronto_Crime database if it doesn't exist"""
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(DB_CONFIG['dbname'])
            ))
            print(f"✓ Database '{DB_CONFIG['dbname']}' created successfully.")
        else:
            print(f"✓ Database '{DB_CONFIG['dbname']}' already exists.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"✗ Error creating database: {e}")

def create_tables():
    """Drop and recreate tables from scratch"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # FORCE DROP tables with CASCADE to rebuild from scratch
        cursor.execute("DROP TABLE IF EXISTS open_consolidated_data CASCADE")
        cursor.execute("DROP TABLE IF EXISTS traffic_collisions_data CASCADE")
        conn.commit()
        print("✓ Dropped existing tables (fresh rebuild mode)")
        
        # Create Open_Consolidated_Data table
        create_consolidated_table = """
        CREATE TABLE open_consolidated_data (
            event_unique_id VARCHAR(50) PRIMARY KEY,
            occ_date TIMESTAMP,
            neighbourhood_158 VARCHAR(100),
            csi_category VARCHAR(100),
            offence VARCHAR(200),
            death BOOLEAN,
            injuries BOOLEAN,
            event_type VARCHAR(100),
            premise_type VARCHAR(200)
        );
        """
        
        # Create Traffic_Collisions_Data table
        create_traffic_table = """
        CREATE TABLE traffic_collisions_data (
            event_unique_id VARCHAR(50) PRIMARY KEY,
            occ_date TIMESTAMP,
            fatalities INTEGER,
            injury_collisions BOOLEAN,
            automobile BOOLEAN,
            motorcycle BOOLEAN,
            passenger BOOLEAN,
            bicycle BOOLEAN,
            pedestrian BOOLEAN,
            neighbourhood_158 VARCHAR(100),
            ftr_collisions BOOLEAN,
            pd_collisions BOOLEAN
        );
        """
        
        cursor.execute(create_consolidated_table)
        print("✓ Table 'open_consolidated_data' created successfully.")
        
        cursor.execute(create_traffic_table)
        print("✓ Table 'traffic_collisions_data' created successfully.")
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ Error creating tables: {e}")
        if conn:
            conn.rollback()

def parse_toronto_date(date_str):
    """Parse Toronto CSV date format: MM/DD/YYYY H:MM:SS AM/PM"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y %I:%M:%S %p')
    except:
        try:
            # Fallback for edge cases
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return None

def convert_yes_no(value):
    """Convert YES/NO/0.0/1.0 style values to boolean"""
    if pd.isna(value) or value == '':
        return None
    val_str = str(value).strip().upper()
    if val_str in ['YES', 'Y', '1', '1.0', 'TRUE']:
        return True
    elif val_str in ['NO', 'N', '0', '0.0', 'FALSE', 'NONE']:
        return False
    return None

def convert_to_int(value, default=0):
    """Convert value to integer safely"""
    if pd.isna(value) or value == '':
        return default
    try:
        return int(float(value))
    except:
        return default

def filter_by_min_year(df, date_column='occ_date', min_year=MIN_YEAR):
    """Filter dataframe to only include rows with dates >= min_year"""
    if date_column not in df.columns:
        print(f"⚠️  Warning: Column '{date_column}' not found, skipping date filter")
        return df
    
    # Keep rows where occ_date is not null AND year >= min_year
    initial_count = len(df)
    df_filtered = df[
        df[date_column].notna() & 
        (df[date_column].dt.year >= min_year)
    ].copy()
    
    filtered_out = initial_count - len(df_filtered)
    if filtered_out > 0:
        print(f"   Filtered out {filtered_out:,} rows with dates before {min_year}")
    
    return df_filtered

def load_csv_to_table(csv_path, table_name, columns, column_converters, batch_size=1000):
    """Load CSV data into specified table with tqdm progress bars"""
    try:
        print(f" Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[columns]
        df.columns = [col.lower() for col in df.columns]
        
        # Apply converters
        for col, converter in column_converters.items():
            if col in df.columns:
                df[col] = df[col].apply(converter)
        
        # 🔹 FILTER: Remove all data before MIN_YEAR (2014)
        df = filter_by_min_year(df, date_column='occ_date', min_year=MIN_YEAR)
        
        # Replace NaN with None for PostgreSQL compatibility
        df = df.replace({np.nan: None})
        
        total_rows = len(df)
        if total_rows == 0:
            print(f"⚠️  No rows remaining after filtering for year >= {MIN_YEAR}")
            return
            
        print(f" Total rows to insert (after filtering): {total_rows:,}")
        
        conn = psycopg2.connect(**DB_CONFIG)
        successful_rows = 0
        failed_rows = []
        
        # tqdm progress bar for batches
        with tqdm(total=total_rows, desc=f" Loading {table_name}", unit="row") as pbar:
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                cursor = conn.cursor()
                
                try:
                    for _, row in batch.iterrows():
                        placeholders = ', '.join(['%s'] * len(row))
                        columns_str = ', '.join([f'"{col}"' for col in df.columns])
                        insert_query = f"""
                            INSERT INTO {table_name} ({columns_str}) 
                            VALUES ({placeholders}) 
                            ON CONFLICT (event_unique_id) DO NOTHING
                        """
                        
                        try:
                            cursor.execute(insert_query, tuple(row))
                            successful_rows += 1
                        except Exception as e:
                            failed_rows.append({
                                'event_id': row.get('event_unique_id', 'unknown'),
                                'error': str(e)
                            })
                        pbar.update(1)
                    
                    conn.commit()
                    
                except Exception as e:
                    print(f"\n✗ Error in batch {i//batch_size + 1}: {e}")
                    conn.rollback()
                finally:
                    cursor.close()
        
        conn.close()
        
        # Summary
        print(f"\n✅ Load Summary for '{table_name}':")
        print(f"   Total rows (after filter): {total_rows:,}")
        print(f"   Inserted: {successful_rows:,}")
        print(f"   Failed: {len(failed_rows):,}")
        
        if failed_rows:
            print(f"\n⚠️  First 5 failed rows:")
            for fail in failed_rows[:5]:
                print(f"   • Event ID: {fail['event_id']} | Error: {fail['error'][:100]}")
        
    except Exception as e:
        print(f"✗ Error loading CSV to {table_name}: {e}")

def verify_data():
    """Verify that data was loaded correctly"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print(f"\n Verification Results:")
        
        cursor.execute("SELECT COUNT(*) FROM open_consolidated_data")
        consolidated_count = cursor.fetchone()[0]
        print(f"   • open_consolidated_data: {consolidated_count:,} records")
        
        cursor.execute("SELECT COUNT(*) FROM traffic_collisions_data")
        traffic_count = cursor.fetchone()[0]
        print(f"   • traffic_collisions_data: {traffic_count:,} records")
        
        # Sample data preview
        if consolidated_count > 0:
            cursor.execute("""
                SELECT event_unique_id, occ_date, offence, death, injuries 
                FROM open_consolidated_data LIMIT 3
            """)
            print(f"\n Sample from open_consolidated_data:")
            for row in cursor.fetchall():
                print(f"   {row}")
        
        if traffic_count > 0:
            cursor.execute("""
                SELECT event_unique_id, occ_date, fatalities, injury_collisions, automobile 
                FROM traffic_collisions_data LIMIT 3
            """)
            print(f"\n Sample from traffic_collisions_data:")
            for row in cursor.fetchall():
                print(f"   {row}")
        
        # Verify no pre-2014 data exists
        cursor.execute("""
            SELECT COUNT(*) FROM open_consolidated_data 
            WHERE occ_date < %s
        """, (pd.Timestamp('2014-01-01'),))
        pre_2014 = cursor.fetchone()[0]
        if pre_2014 > 0:
            print(f"\n⚠️  WARNING: {pre_2014} records with dates before 2014 found in open_consolidated_data!")
        else:
            print(f"\n✓ Confirmed: No records before 2014 in open_consolidated_data")
            
        cursor.execute("""
            SELECT COUNT(*) FROM traffic_collisions_data 
            WHERE occ_date < %s
        """, (pd.Timestamp('2014-01-01'),))
        pre_2014 = cursor.fetchone()[0]
        if pre_2014 > 0:
            print(f"⚠️  WARNING: {pre_2014} records with dates before 2014 found in traffic_collisions_data!")
        else:
            print(f"✓ Confirmed: No records before 2014 in traffic_collisions_data")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ Error verifying data: {e}")

def main():
    """Main orchestration function"""
    print("\n" + "*"*25)
    print("  TORONTO_CRIME DATABASE SETUP - FRESH REBUILD MODE")
    print(f"  Filtering: Data from {MIN_YEAR} onwards only")
    print("*"*25 + "\n")
    
    # Step 1: Create database
    create_database()
    
    # Step 2: Drop & recreate tables (fresh start)
    create_tables()
    
    # Define columns
    consolidated_columns = [
        'EVENT_UNIQUE_ID', 'OCC_DATE', 'NEIGHBOURHOOD_158', 'CSI_CATEGORY',
        'OFFENCE', 'DEATH', 'INJURIES', 'EVENT_TYPE', 'PREMISE_TYPE'
    ]
    
    traffic_columns = [
        'EVENT_UNIQUE_ID', 'OCC_DATE', 'FATALITIES', 'INJURY_COLLISIONS',
        'AUTOMOBILE', 'MOTORCYCLE', 'PASSENGER', 'BICYCLE', 'PEDESTRIAN',
        'NEIGHBOURHOOD_158', 'FTR_COLLISIONS', 'PD_COLLISIONS'
    ]
    
    # Define converters for Open_Consolidated_Data
    consolidated_converters = {
        'occ_date': parse_toronto_date,
        'death': lambda x: convert_yes_no(x) if pd.notna(x) else None,
        'injuries': lambda x: convert_yes_no(x) if pd.notna(x) else None
    }
    
    # Define converters for Traffic_Collisions_Data
    traffic_converters = {
        'occ_date': parse_toronto_date,
        'fatalities': convert_to_int,
        'injury_collisions': convert_yes_no,
        'automobile': convert_yes_no,
        'motorcycle': convert_yes_no,
        'passenger': convert_yes_no,
        'bicycle': convert_yes_no,
        'pedestrian': convert_yes_no,
        'ftr_collisions': convert_yes_no,
        'pd_collisions': convert_yes_no
    }
    
    # Load Open_Consolidated_Data.csv
    print("\n" + "─"*60)
    load_csv_to_table(
        'project/DB_csv/Open_Consolidated_Data.csv',
        'open_consolidated_data',
        consolidated_columns,
        consolidated_converters
    )
    
    # Load Traffic_Collisions_Data.csv
    print("\n" + "─"*60)
    load_csv_to_table(
        'project/DB_csv/Traffic_Collisions_Data.csv',
        'traffic_collisions_data',
        traffic_columns,
        traffic_converters
    )
    
    # Verify
    verify_data()
    
    print("\n" + "*"*25)
    print("  DATABASE SETUP COMPLETED SUCCESSFULLY!")
    print("*"*25 + "\n")

if __name__ == "__main__":
    main()