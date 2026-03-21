import pandas as pd
from pathlib import Path

# 1. Configuration
folder_path = Path(r'project\DB_csv')
file_path = folder_path / 'Open_Consolidated_Data.csv'

# Define the columns to check
columns_to_check = ['CSI_CATEGORY', 'OFFENCE', 'EVENT_TYPE']

# 2. Load the Data
print(f"Reading file: {file_path}")

if not file_path.exists():
    print(f"Error: {file_path} does not exist.")
else:
    try:
        # Read the CSV
        df = pd.read_csv(file_path, encoding='utf-8')
        original_row_count = len(df)
        print(f"Original row count: {original_row_count}")
        
        # 3. Define what counts as "empty"
        # We check for: NaN, empty string, whitespace only, or the string 'NA'
        def is_empty(value):
            if pd.isna(value):
                return True
            if str(value).strip() == '':
                return True
            if str(value).strip().upper() == 'NA':
                return True
            return False
        
        # 4. Filter Rows
        # We want to KEEP rows where AT LEAST ONE of the three columns has data
        # We DELETE rows where ALL THREE are empty
        
        mask_keep = []
        for idx, row in df.iterrows():
            # Check if at least one column has valid data
            has_data = False
            for col in columns_to_check:
                if col in df.columns:
                    if not is_empty(row[col]):
                        has_data = True
                        break
            
            mask_keep.append(has_data)
        
        # Apply the filter
        df_cleaned = df[mask_keep].reset_index(drop=True)
        
        new_row_count = len(df_cleaned)
        rows_removed = original_row_count - new_row_count
        
        print(f"\nRows removed: {rows_removed}")
        print(f"New row count: {new_row_count}")
        
        # 5. Save Back to the Same File
        df_cleaned.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        print(f"\nSuccess! Updated {file_path}")
        print(f"Removed {rows_removed} rows where CSI_CATEGORY, OFFENCE, and EVENT_TYPE were all empty.")
        
    except Exception as e:
        print(f"Error processing file: {e}")