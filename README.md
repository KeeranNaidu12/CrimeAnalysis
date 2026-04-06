

data found from:
open collision: https://data.tps.ca/maps/bc4c72a793014a55a674984ef175a6f3

genral community safety indicators: https://data.tps.ca/datasets/0a239a5563a344a3bbf8452504ed8d68_0/explore
shioting and firearm discarge: https://data.tps.ca/datasets/64ddeca12da34403869968ec725e23c4_0/explore
Homicide: https://data.tps.ca/datasets/d96bf5b67c1c49879f354dad51cf81f9_0/explore
Assualt: https://data.tps.ca/datasets/b4d0398d37eb4aa184065ed625ddb922_0/explore
Auto theft: https://data.tps.ca/datasets/95ab41aee16847dba8453bf1688249d6_0/explore?location=21.926582%2C-39.819624%2C2
bicycle tehft: https://data.tps.ca/datasets/a89d10d5e28444ceb0c8d1d4c0ee39cc_0/explore
break and entering: https://data.tps.ca/datasets/040ead448df2412da252cfbb532e77ac_0/explore
robbery: https://data.tps.ca/datasets/d0e1e98de5f945faa2fe635dee3f4062_0/explore
theft over: https://data.tps.ca/datasets/7530d9b637c340059ccb81a782481c04_0/explore
theft from motor vehicle: https://data.tps.ca/datasets/d9303bc20f8a4351b7744a8703eecb80_0/explore?location=21.926582%2C-39.819624%2C2


that being said the data had alot of inconsistencies and wekanesses so we needed to clean it 

main cleaning:
1. removing any data pre 2014
2. removing any data whose neighborhood is NSA or null or date is null 
3. mainly for crime
    0. `update_missing_eventtype.py`:  
    - Clean up EVENT_TYPE and CSI_CATEGORY relationship: Some rows have EVENT_TYPE but NO CSI_CATEGORY and Other rows with the same EVENT_UNIQUE_ID have CSI_CATEGORY but missing EVENT_TYPE
    1. Scan all data and group rows by EVENT_UNIQUE_ID
    2. Identify correct EVENT_TYPE from rows that have it (even if CSI_CATEGORY is missing)
    3. Delete orphaned rows - Remove rows that have EVENT_TYPE but no CSI_CATEGORY (these are incomplete records)
    4. Update missing EVENT_TYPE - For rows that have CSI_CATEGORY but no EVENT_TYPE, fill in the EVENT_TYPE from other rows with the same EVENT_UNIQUE_ID
    5. Input:Open_Consolidated_Data.csv->Output:Open_Consolidated_Data_updated.csv

    0. `delete_dupes.py`:
    - Remove rows that are identical in every column
    1. Reads the CSV file
    2. Creates an MD5 hash for each row using all columns
    3. Keeps ONLY the first occurrence of each unique hash
    4. Removes all subsequent duplicate rows
    5. Asks for user confirmation before proceeding
    6. Input:Open_Consolidated_Data_updated.csv -> Output:Open_Consolidated_Data_updated_deduplicated.csv

    0. `add_traffic_features.py`
    - Remove rows from a specific neighborhood (NSA) & Enrich data with temporal features for analysis
    1. Filters out rows where NEIGHBOURHOOD_158 = 'NSA'
    2. Adds time-based features to the remaining data
    New columns added:
    week_day - Day name (e.g., "Monday", "Tuesday")
    season - Meteorological season (Spring/Summer/Autumn/Winter)
    holiday - Boolean (True if date is Canadian/Ontario holiday)
    5. Input: Open_Consolidated_Data_updated_deduplicated.csv -> Output: Open_Consolidated_Data_updated_deduplicated_enhanced.csv

## PostgreSQL Upload Process

| Script | Source File | Target Table | Filtering Applied |
|--------|-------------|--------------|-------------------|
| `DB_setup_crime.py` | `*_enhanced.csv` | `open_consolidated_data` | None (all rows) |
| `DB_setup_traffic.py` | `Traffic_Collisions_Data_enhanced.csv` | `traffic_collisions_data` | None (all rows) |
| `sub_crime_setup.py` | PostgreSQL table (not CSV) | Multiple `csi_*` tables | **Date filter: `occ_date >= '2014-01-01'`** |

---

### Upload Process (Same Pattern for Both CSV Scripts)

1. **Read CSV** → parse into Python tuples
2. **Connect to PostgreSQL** using `.env` credentials
3. **Drop existing table** (if exists) → `DROP TABLE IF EXISTS`
4. **Create fresh table** with predefined schema
5. **Batch insert** (500 rows at a time) using `executemany()`
6. **Commit** transaction

---

### Additional Filtering in `sub_crime_setup.py`

This script **does NOT read a CSV** — it reads from the already-loaded `open_consolidated_data` table and applies:

| Filter | Description |
|--------|-------------|
| `csi_category IS NOT NULL AND <> ''` | Only non-empty CSI categories |
| `occ_date >= '2014-01-01'` | **Only data from 2014 onward** |

It then creates **separate tables per CSI category** (e.g., `csi_auto_theft`, `csi_assault`).


### Summary of What Gets Uploaded

| Stage | Rows Included |
|-------|---------------|
| `DB_setup_crime.py` | All rows from `*_enhanced.csv` (including pre-2014) |
| `DB_setup_traffic.py` | All rows from traffic enhanced CSV |
| `sub_crime_setup.py` | **Only** rows with `csi_category` AND date ≥ 2014 |




