import os
import re
import pandas as pd
from pathlib import Path

# 1. Configuration
# Update this path if your script is running from a different location
folder_path = Path(r'project\DB_csv')

# Define the exact columns you want to keep
target_columns = [
    'EVENT_UNIQUE_ID',
    'OCC_DATE',
    'NEIGHBOURHOOD_158',
    'CSI_CATEGORY',
    'OFFENCE',
    'DEATH',
    'INJURIES',
    'EVENT_TYPE',
    'PREMISE_TYPE'
]

# 2. Process Files
if not folder_path.exists():
    print(f"Error: Folder {folder_path} does not exist.")
else:
    files_processed = 0
    
    # Get all csv files in the folder
    csv_files = [f for f in folder_path.iterdir() if f.suffix.lower() == '.csv']
    
    for file_path in csv_files:
        try:
            # --- A. Read the CSV ---
            # We try utf-8 first, fallback to latin-1 if that fails (common with older CSVs)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            
            # --- B. Handle Columns ---
            # 1. Check for missing columns and add them with 'NA'
            for col in target_columns:
                if col not in df.columns:
                    df[col] = 'NA'
            
            # 2. Select only the target columns (this drops unwanted columns and orders them)
            df = df[target_columns]
            
            # --- C. Determine New Filename ---
            original_name = file_path.stem # Name without extension
            extension = file_path.suffix   # .csv
            
            # Regex looks for anything up to and including 'OPEN_DATA' (case-insensitive)
            # Group 1 captures the part we want to keep
            match = re.search(r'(.*OPEN_DATA).*', original_name, re.IGNORECASE)
            
            if match:
                new_name_base = match.group(1) + '_up1'
            else:
                # Fallback if keyword isn't found: just append _up1 to original name
                new_name_base = original_name + '_up1'
            
            new_filename = new_name_base + extension
            new_file_path = file_path.with_name(new_filename)
            
            # --- D. Save and Cleanup ---
            # Save the cleaned data to the NEW filename
            df.to_csv(new_file_path, index=False, encoding='utf-8-sig')
            
            # Delete the OLD filename
            file_path.unlink()
            
            print(f"Processed: {file_path.name} -> {new_filename}")
            files_processed += 1
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print(f"\nComplete. {files_processed} files processed.")