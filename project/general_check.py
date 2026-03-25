import pandas as pd
from collections import Counter
import os

# Specify the file path
file_path = r'DB_csv\Major_Crime_Indicators_Open_Data_-4289692410590149445 (1).csv'

def search_event_by_id(df):
    """
    Function to search for an event by EVENT_UNIQUE_ID and display relevant information
    """
    print("\n" + "="*60)
    print("EVENT UNIQUE ID SEARCH")
    print("="*60)
    
    # Check if EVENT_UNIQUE_ID column exists
    if 'EVENT_UNIQUE_ID' not in df.columns:
        print("Error: EVENT_UNIQUE_ID column not found in the dataset.")
        print("Available columns:", list(df.columns))
        return
    
    while True:
        print("\n" + "-"*40)
        print("Search Options:")
        print("1. Exact match search")
        print("2. Partial match search (contains)")
        print("3. Exit search")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '3':
            print("Exiting search function.")
            break
            
        event_id = input("Enter EVENT_UNIQUE_ID to search: ").strip()
        
        if not event_id:
            print("Please enter a valid EVENT_UNIQUE_ID.")
            continue
        
        if choice == '1':
            # Exact match
            result = df[df['EVENT_UNIQUE_ID'].astype(str) == event_id]
            match_type = "exact match"
        elif choice == '2':
            # Partial match (contains)
            result = df[df['EVENT_UNIQUE_ID'].astype(str).str.contains(event_id, case=False, na=False)]
            match_type = "partial match"
        else:
            print("Invalid choice. Please try again.")
            continue
        
        if len(result) == 0:
            print(f"No records found for {match_type} '{event_id}'")
        else:
            print(f"\nFound {len(result)} record(s) for {match_type} '{event_id}':")
            print("="*80)
            
            # Display the results
            for idx, row in result.iterrows():
                print(f"\nRecord #{idx + 1}:")
                print(f"  EVENT_UNIQUE_ID: {row.get('EVENT_UNIQUE_ID', 'N/A')}")
                print(f"  REPORT_DATE: {row.get('REPORT_DATE', 'N/A')}")
                print(f"  OFFENCE: {row.get('OFFENCE', 'N/A')}")
                print(f"  CSI_CATEGORY: {row.get('CSI_CATEGORY', 'N/A')}")
                print(f"  LOCATION_TYPE: {row.get('LOCATION_TYPE', 'N/A')}")
                print(f"  NEIGHBOURHOOD_158: {row.get('NEIGHBOURHOOD_158', 'N/A')}")
                print("-" * 40)

def display_summary_statistics(df):
    """
    Function to display summary statistics about the dataset
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nColumns with missing values:")
        for col in missing_values[missing_values > 0].index:
            print(f"  {col}: {missing_values[col]} missing values")

try:
    # Read the CSV file
    print("Loading CSV file...")
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
    
    # Display summary statistics
    display_summary_statistics(df)
    
    # Check if required columns exist
    required_columns = ['OFFENCE', 'CSI_CATEGORY']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if not missing_columns:
        # Get unique values and their counts for OFFENCE
        offence_counts = df['OFFENCE'].value_counts()
        
        # Get unique values and their counts for CSI_CATEGORY
        csi_counts = df['CSI_CATEGORY'].value_counts()
        
        # Print results for OFFENCE
        print("\n" + "="*60)
        print("UNIQUE OFFENCE VALUES AND OCCURRENCES")
        print("="*60)
        print(f"Total unique offence types: {len(offence_counts)}")
        print("-" * 40)
        for offence, count in offence_counts.items():
            print(f"{offence}: {count}")
        
        print("\n" + "="*60)
        print("UNIQUE CSI_CATEGORY VALUES AND OCCURRENCES")
        print("="*60)
        print(f"Total unique CSI categories: {len(csi_counts)}")
        print("-" * 40)
        for category, count in csi_counts.items():
            print(f"{category}: {count}")
            
    else:
        print(f"\nError: Required columns not found in the CSV file: {missing_columns}")
        print("Available columns:", list(df.columns))
    
    # Call the search function
    search_event_by_id(df)
    
except FileNotFoundError:
    print(f"\nError: File not found at {file_path}")
    print("Current working directory:", os.getcwd())
    print("\nFiles in current directory:")
    for file in os.listdir('.'):
        print(f"  - {file}")
    
    # Check if DB_csv folder exists
    db_csv_path = os.path.join(os.getcwd(), 'DB_csv')
    if os.path.exists(db_csv_path):
        print(f"\nDB_csv folder exists. Contents:")
        for file in os.listdir(db_csv_path):
            print(f"  - {file}")
    else:
        print(f"\nDB_csv folder does not exist at: {db_csv_path}")
        print("Please make sure the folder structure is correct:")
        print("  Current directory should contain:")
        print("  - your_script.py")
        print("  - DB_csv/ (folder)")
        print("    - Major_Crime_Indicators_Open_Data_-4289692410590149445 (1).csv")
        
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print(f"Error type: {type(e).__name__}")