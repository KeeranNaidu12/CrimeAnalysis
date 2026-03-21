import pandas as pd
from pathlib import Path

# 1. Configuration
folder_path = Path(r'project\DB_csv')
output_filename = 'Open_Full_data.csv'
output_path = folder_path / output_filename

# Define the columns exactly as they appear in your processed files
# (Based on your previous request)
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

# 2. Collect Unique IDs
unique_ids = set()
files_processed = 0

print(f"Scanning folder: {folder_path.absolute()}")

if not folder_path.exists():
    print(f"Error: Folder {folder_path} does not exist.")
else:
    csv_files = [f for f in folder_path.iterdir() if f.suffix.lower() == '.csv']
    
    for file_path in csv_files:
        # Skip the output file if it already exists to avoid reading partial data
        if file_path.name == output_filename:
            continue
            
        try:
            # Read CSV (matching encoding from previous script)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            
            # Ensure the column exists before trying to read it
            if 'EVENT_UNIQUE_ID' in df.columns:
                # Get unique values, drop NaNs, and filter out the string 'NA' if present
                ids = df['EVENT_UNIQUE_ID'].dropna().unique()
                
                for id_val in ids:
                    # Ensure we don't add the placeholder 'NA' as a real ID
                    if str(id_val).strip().upper() != 'NA':
                        unique_ids.add(id_val)
            
            files_processed += 1
            
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # 3. Create the New DataFrame
    print(f"Found {len(unique_ids)} unique EVENT_UNIQUE_IDs across {files_processed} files.")
    
    # Create a dataframe with just the IDs first
    result_df = pd.DataFrame(list(unique_ids), columns=['EVENT_UNIQUE_ID'])
    
    # Add the remaining columns and fill them with 'NA'
    for col in target_columns:
        if col != 'EVENT_UNIQUE_ID':
            result_df[col] = 'NA'
    
    # Reorder columns to match the standard structure
    result_df = result_df[target_columns]
    
    # 4. Save the File
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Success! Created {output_filename} at {output_path.absolute()}")