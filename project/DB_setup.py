import os
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import pandas as pd
import numpy as np

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

def create_database():
    """Create the Toronto_Crime database if it doesn't exist"""
    try:
        # Connect to default postgres database to create new database
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(DB_CONFIG['dbname'])
            ))
            print(f"Database '{DB_CONFIG['dbname']}' created successfully.")
        else:
            print(f"Database '{DB_CONFIG['dbname']}' already exists.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")

def create_tables():
    """Create tables for the CSV files"""
    try:
        # Connect to Toronto_Crime database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Drop existing tables if they exist (optional - remove if you want to preserve data)
        cursor.execute("DROP TABLE IF EXISTS open_consolidated_data CASCADE")
        cursor.execute("DROP TABLE IF EXISTS traffic_collisions_data CASCADE")
        print("Dropped existing tables if they existed.")
        
        # Create Open_Consolidated_Data table
        create_consolidated_table = """
        CREATE TABLE open_consolidated_data (
            event_unique_id VARCHAR(50) PRIMARY KEY,
            occ_date DATE,
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
            occ_date DATE,
            fatalities INTEGER,
            injury_collisions INTEGER,
            automobile INTEGER,
            motorcycle INTEGER,
            passenger INTEGER,
            bicycle INTEGER,
            pedestrian INTEGER,
            neighbourhood_158 VARCHAR(100),
            ftr_collisions INTEGER,
            pd_collisions INTEGER
        );
        """
        
        cursor.execute(create_consolidated_table)
        print("Table 'open_consolidated_data' created successfully.")
        
        cursor.execute(create_traffic_table)
        print("Table 'traffic_collisions_data' created successfully.")
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error creating tables: {e}")

def load_csv_to_table(csv_path, table_name, columns, dtype_mapping=None, batch_size=1000):
    """Load CSV data into specified table with batch processing"""
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Select only the columns we need
        df = df[columns]
        
        # Convert column names to lowercase for PostgreSQL
        df.columns = [col.lower() for col in df.columns]
        
        # Convert data types if mapping provided
        if dtype_mapping:
            for col, dtype in dtype_mapping.items():
                if col in df.columns:
                    try:
                        if dtype == 'date':
                            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                        elif dtype == 'boolean':
                            # Handle various boolean representations
                            df[col] = df[col].astype(str).str.upper().map({
                                'YES': True, 'Y': True, '1': True, 'TRUE': True,
                                'NO': False, 'N': False, '0': False, 'FALSE': False,
                                'NONE': False, '': False, 'NAN': False
                            }).fillna(False)
                        elif dtype == 'integer':
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    except Exception as e:
                        print(f"Warning: Could not convert column {col} to {dtype}: {e}")
                        continue
        
        # Replace NaN with None for PostgreSQL NULL
        df = df.replace({np.nan: None})
        
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Insert data in batches to avoid transaction issues
        total_rows = len(df)
        successful_rows = 0
        failed_rows = []
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size]
            cursor = conn.cursor()
            
            try:
                for _, row in batch.iterrows():
                    placeholders = ', '.join(['%s'] * len(row))
                    columns_str = ', '.join([f'"{col}"' for col in df.columns])
                    insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders}) ON CONFLICT (event_unique_id) DO NOTHING"
                    
                    try:
                        cursor.execute(insert_query, tuple(row))
                        successful_rows += 1
                    except Exception as e:
                        failed_rows.append({
                            'event_id': row.get('event_unique_id', 'unknown'),
                            'error': str(e)
                        })
                        continue
                
                # Commit the batch
                conn.commit()
                print(f"Processed batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                conn.rollback()
            finally:
                cursor.close()
        
        conn.close()
        
        print(f"\nLoad Summary for '{table_name}':")
        print(f"  Total rows in CSV: {total_rows}")
        print(f"  Successfully inserted: {successful_rows}")
        print(f"  Failed inserts: {len(failed_rows)}")
        
        if failed_rows:
            print(f"\nFirst 10 failed rows:")
            for fail in failed_rows[:10]:
                print(f"  Event ID: {fail['event_id']}, Error: {fail['error']}")
        
    except Exception as e:
        print(f"Error loading CSV to {table_name}: {e}")

def verify_data():
    """Verify that data was loaded correctly"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check consolidated data
        cursor.execute("SELECT COUNT(*) FROM open_consolidated_data")
        consolidated_count = cursor.fetchone()[0]
        print(f"\nVerification:")
        print(f"  Records in open_consolidated_data: {consolidated_count}")
        
        # Check traffic data
        cursor.execute("SELECT COUNT(*) FROM traffic_collisions_data")
        traffic_count = cursor.fetchone()[0]
        print(f"  Records in traffic_collisions_data: {traffic_count}")
        
        # Show sample data
        if consolidated_count > 0:
            cursor.execute("SELECT event_unique_id, occ_date, offence FROM open_consolidated_data LIMIT 5")
            print(f"\nSample records from open_consolidated_data:")
            for row in cursor.fetchall():
                print(f"  {row}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error verifying data: {e}")

def main():
    """Main function to orchestrate database setup and data loading"""
    print("Starting Toronto_Crime database setup...")
    
    # Create database
    create_database()
    
    # Create tables
    create_tables()
    
    # Define columns for each CSV
    consolidated_columns = [
        'EVENT_UNIQUE_ID', 'OCC_DATE', 'NEIGHBOURHOOD_158', 'CSI_CATEGORY',
        'OFFENCE', 'DEATH', 'INJURIES', 'EVENT_TYPE', 'PREMISE_TYPE'
    ]
    
    traffic_columns = [
        'EVENT_UNIQUE_ID', 'OCC_DATE', 'FATALITIES', 'INJURY_COLLISIONS',
        'AUTOMOBILE', 'MOTORCYCLE', 'PASSENGER', 'BICYCLE', 'PEDESTRIAN',
        'NEIGHBOURHOOD_158', 'FTR_COLLISIONS', 'PD_COLLISIONS'
    ]
    
    # Define data type mappings
    consolidated_dtype_mapping = {
        'occ_date': 'date',
        'death': 'boolean',
        'injuries': 'boolean'
    }
    
    traffic_dtype_mapping = {
        'occ_date': 'date',
        'fatalities': 'integer',
        'injury_collisions': 'integer',
        'automobile': 'integer',
        'motorcycle': 'integer',
        'passenger': 'integer',
        'bicycle': 'integer',
        'pedestrian': 'integer',
        'ftr_collisions': 'integer',
        'pd_collisions': 'integer'
    }
    
    # Load CSV data
    print("\n" + "="*50)
    print("Loading Open_Consolidated_Data.csv...")
    print("="*50)
    load_csv_to_table(
        'project/DB_csv/Open_Consolidated_Data.csv',
        'open_consolidated_data',
        consolidated_columns,
        consolidated_dtype_mapping
    )
    
    print("\n" + "="*50)
    print("Loading Traffic_Collisions_Data.csv...")
    print("="*50)
    load_csv_to_table(
        'project/DB_csv/Traffic_Collisions_Data.csv',
        'traffic_collisions_data',
        traffic_columns,
        traffic_dtype_mapping
    )
    
    # Verify data
    verify_data()
    
    print("\n" + "="*50)
    print("Toronto_Crime database setup completed!")
    print("="*50)

if __name__ == "__main__":
    main()