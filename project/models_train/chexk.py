import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment
ENV_PATH = Path("project/.env")
load_dotenv(ENV_PATH)

# Database connection
db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}

connection_string = (
    f"postgresql://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
)

engine = create_engine(connection_string)

# Query to see all unique CSI categories
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT DISTINCT csi_category 
        FROM csi_break_and_enter 
        WHERE csi_category IS NOT NULL
        ORDER BY csi_category
    """))
    
    print("Unique CSI categories in csi_break_and_enter table:")
    for row in result:
        print(f"  - '{row[0]}'")
    
    # Also check total counts per category
    result2 = conn.execute(text("""
        SELECT csi_category, COUNT(*) as count
        FROM csi_break_and_enter 
        WHERE csi_category IS NOT NULL
        GROUP BY csi_category
        ORDER BY count DESC
    """))
    
    print("\nCounts per category:")
    for row in result2:
        print(f"  {row[0]}: {row[1]:,} records")

engine.dispose()