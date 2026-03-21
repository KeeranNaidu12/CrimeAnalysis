import pandas as pd

# Define the columns to keep
columns_to_keep = [
    'EVENT_UNIQUE_ID',
    'OCC_DATE',
    'FATALITIES',
    'INJURY_COLLISIONS',
    'AUTOMOBILE',
    'MOTORCYCLE',
    'PASSENGER',
    'BICYCLE',
    'PEDESTRIAN',
    'NEIGHBOURHOOD_158',
    'FTR_COLLISIONS',
    'PD_COLLISIONS'
]

# Read the CSV file
input_file = r'project\DB_csv\Traffic_Collisions_Data.csv'
output_file = r'project\DB_csv\Traffic_Collisions_Filtered.csv'

# Load data
df = pd.read_csv(input_file)

# Check which columns exist
missing_columns = [col for col in columns_to_keep if col not in df.columns]
if missing_columns:
    print(f"Warning: The following columns were not found in the CSV: {missing_columns}")
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]

# Filter to keep only specified columns
df_filtered = df[columns_to_keep]

# Save to new CSV file
df_filtered.to_csv(output_file, index=False)

print(f"✓ Successfully filtered {len(df_filtered)} rows")
print(f"✓ Kept {len(columns_to_keep)} columns")
print(f"✓ Output saved to: {output_file}")