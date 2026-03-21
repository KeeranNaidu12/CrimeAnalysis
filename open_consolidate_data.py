import pandas as pd
from pathlib import Path

# 1. Configuration
folder_path = Path(r'project\DB_csv')
source_list_file = folder_path / 'Open_Full_data.csv'
output_filename = 'Open_Consolidated_Data.csv'
output_path = folder_path / output_filename

# Define the standard columns
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

# 2. Load the List of IDs to Search For
print(f"Reading ID list from: {source_list_file}")

if not source_list_file.exists():
    print(f"Error: {source_list_file} does not exist. Please run the second script first.")
else:
    try:
        # Read the ID list
        df_ids = pd.read_csv(source_list_file, encoding='utf-8')
        
        # Get unique IDs as a set for fast lookup
        # Convert to string to ensure consistent matching
        target_ids = set(df_ids['EVENT_UNIQUE_ID'].dropna().astype(str))
        
        print(f"Found {len(target_ids)} unique IDs to search for.")
        
    except Exception as e:
        print(f"Error reading ID list: {e}")
        exit()

    # 3. Search Through All Other CSV Files
    print(f"\nScanning folder for matching data: {folder_path.absolute()}")
    
    all_matching_rows = []
    files_searched = 0
    matches_found = 0
    
    csv_files = [f for f in folder_path.iterdir() if f.suffix.lower() == '.csv']
    
    for file_path in csv_files:
        # Skip the source list file and the output file
        if file_path.name in ['Open_Full_data.csv', output_filename]:
            continue
            
        try:
            # Read CSV (matching encoding from previous scripts)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
            
            # Check if EVENT_UNIQUE_ID column exists
            if 'EVENT_UNIQUE_ID' not in df.columns:
                continue
            
            # Convert IDs to string for consistent matching
            df['EVENT_UNIQUE_ID'] = df['EVENT_UNIQUE_ID'].astype(str)
            
            # Filter rows where EVENT_UNIQUE_ID is in our target set
            matching_rows = df[df['EVENT_UNIQUE_ID'].isin(target_ids)]
            
            if len(matching_rows) > 0:
                # Ensure all target columns exist, add missing ones with 'NA'
                for col in target_columns:
                    if col not in matching_rows.columns:
                        matching_rows[col] = 'NA'
                
                # Select only target columns in correct order
                matching_rows = matching_rows[target_columns]
                
                # Add to our collection
                all_matching_rows.append(matching_rows)
                matches_found += len(matching_rows)
            
            files_searched += 1
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    # 4. Combine All Matching Rows
    print(f"\nSearched {files_searched} files.")
    print(f"Found {matches_found} total matching records.")
    
    if len(all_matching_rows) > 0:
        # Concatenate all dataframes
        final_df = pd.concat(all_matching_rows, ignore_index=True)
        
        # --- NEW: Sort by EVENT_UNIQUE_ID in ascending order ---
        final_df = final_df.sort_values(by='EVENT_UNIQUE_ID', ascending=True).reset_index(drop=True)
        # --------------------------------------------------------
        
        # 5. Save the Consolidated Data
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\nSuccess! Created {output_filename}")
        print(f"Output location: {output_path.absolute()}")
        print(f"Total rows in output: {len(final_df)}")
        
        # Show some stats about duplicates
        id_counts = final_df['EVENT_UNIQUE_ID'].value_counts()
        duplicates = id_counts[id_counts > 1]
        
        if len(duplicates) > 0:
            print(f"\nNote: {len(duplicates)} IDs appear more than once (total duplicate occurrences: {duplicates.sum() - len(duplicates)})")
        else:
            print("\nAll IDs in the output are unique.")
            
    else:
        print("\nWarning: No matching data found for any of the IDs.")
        print("Creating empty file with headers only.")
        
        # Create empty dataframe with correct columns
        empty_df = pd.DataFrame(columns=target_columns)
        empty_df.to_csv(output_path, index=False, encoding='utf-8-sig')