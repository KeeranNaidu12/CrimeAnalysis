# Toronto Crime & Traffic Collision Analysis

## Project Overview

This project provides a comprehensive data pipeline for analyzing crime and traffic collision data in Toronto. It includes data cleaning, PostgreSQL database integration, Power BI visualizations, machine learning predictions, and a full-stack web application for interactive exploration.

**Full Report:**  
A detailed project report is available at:  
[`301_crimeanalysis_report.pdf`](301_crimeanalysis_report.pdf)

---

## Data Sources

All raw data was sourced from the Toronto Police Service’s open data portal:

| Dataset | Link |
|---------|------|
| Open Collision | [data.tps.ca](https://data.tps.ca/maps/bc4c72a793014a55a674984ef175a6f3) |
| General Community Safety Indicators | [data.tps.ca](https://data.tps.ca/datasets/0a239a5563a344a3bbf8452504ed8d68_0/explore) |
| Shooting & Firearm Discharge | [data.tps.ca](https://data.tps.ca/datasets/64ddeca12da34403869968ec725e23c4_0/explore) |
| Homicide | [data.tps.ca](https://data.tps.ca/datasets/d96bf5b67c1c49879f354dad51cf81f9_0/explore) |
| Assault | [data.tps.ca](https://data.tps.ca/datasets/b4d0398d37eb4aa184065ed625ddb922_0/explore) |
| Auto Theft | [data.tps.ca](https://data.tps.ca/datasets/95ab41aee16847dba8453bf1688249d6_0/explore) |
| Bicycle Theft | [data.tps.ca](https://data.tps.ca/datasets/a89d10d5e28444ceb0c8d1d4c0ee39cc_0/explore) |
| Break & Enter | [data.tps.ca](https://data.tps.ca/datasets/040ead448df2412da252cfbb532e77ac_0/explore) |
| Robbery | [data.tps.ca](https://data.tps.ca/datasets/d0e1e98de5f945faa2fe635dee3f4062_0/explore) |
| Theft Over | [data.tps.ca](https://data.tps.ca/datasets/7530d9b637c340059ccb81a782481c04_0/explore) |
| Theft from Motor Vehicle | [data.tps.ca](https://data.tps.ca/datasets/d9303bc20f8a4351b7744a8703eecb80_0/explore) |

---

## Data Cleaning Process

Raw data contained inconsistencies and missing values. The following cleaning steps were applied **in order**:

### 1. Remove data before 2014
### 2. Remove rows with missing neighborhood (NSA or null) or missing date
### 3. Crime-specific cleaning scripts

| Script | Purpose |
|--------|---------|
| `update_missing_eventtype.py` | Fix missing EVENT_TYPE / CSI_CATEGORY relationships by grouping by EVENT_UNIQUE_ID, identifying correct values, deleting orphaned rows, and filling missing fields |
| `delete_dupes.py` | Remove exact duplicate rows using MD5 hashing (keeps first occurrence only, asks for confirmation) |
| `add_traffic_features.py` | Filter out 'NSA' neighborhood and add temporal features: `week_day`, `season`, `holiday` (Canadian/Ontario holidays) |

**Data flow after cleaning:**

1. `Open_Consolidated_Data.csv` → `update_missing_eventtype.py` → `Open_Consolidated_Data_updated.csv`
2. → `delete_dupes.py` → `Open_Consolidated_Data_updated_deduplicated.csv`
3. → `add_traffic_features.py` → `Open_Consolidated_Data_updated_deduplicated_enhanced.csv`

---

## PostgreSQL Upload Process

> **Important:** Create a `.env` file with your PostgreSQL credentials before running any database scripts.

### Upload Scripts (run in this order)

| Script | Source | Target Table | Filtering |
|--------|--------|--------------|-----------|
| `DB_setup_crime.py` | `*_enhanced.csv` | `open_consolidated_data` | None |
| `DB_setup_traffic.py` | `Traffic_Collisions_Data_enhanced.csv` | `traffic_collisions_data` | None |
| `sub_crime_setup.py` | PostgreSQL table (not CSV) | Multiple `csi_*` tables | `occ_date >= '2014-01-01'` AND `csi_category IS NOT NULL` |

### Common Upload Pattern (for CSV scripts)

1. Read CSV → parse into tuples
2. Connect to PostgreSQL using `.env`
3. Drop existing table (`DROP TABLE IF EXISTS`)
4. Create fresh table with schema
5. Batch insert (500 rows at a time) using `executemany()`
6. Commit transaction

### `sub_crime_setup.py` Special Behavior

This script reads from the already-loaded `open_consolidated_data` table, applies filtering, and creates separate tables per CSI category:

- `csi_auto_theft`
- `csi_assault`
- `csi_break_and_enter`
- (and others as defined in the data)

---

## Data Summary by Stage

| Stage | Rows Included |
|-------|----------------|
| `DB_setup_crime.py` | All rows from `*_enhanced.csv` (including pre-2014) |
| `DB_setup_traffic.py` | All rows from traffic enhanced CSV |
| `sub_crime_setup.py` | Only rows with `csi_category` AND date ≥ 2014 |

---

## Complete Setup Instructions

### Step 1: Download the Data
Download all CSV files from the links above and place them in the `data/` directory.

### Step 2: Run Cleaning Scripts
Run the cleaning scripts **in the order shown** in the Data Cleaning Process section above.

### Step 3: Upload to Database
Run the following scripts **in order**:

```bash
python project/DB_setup_crime.py
python project/DB_setup_traffic.py
python project/sub_crime_setup.py
```

### Step 4: Explore the Database (Optional)
Run this script to browse the database, view schemas, search for observations, and see statistics diagrams:

```bash
python project/DB_querying.py
```

### Step 5: Run Additional Safety Analysis (Optional)

```bash
python traffic_yearly_safety.py
```

Then run these in order (legacy — see Power BI note below):

```bash
python crime_category_long.py
python crime_subcategory_long.py
python crime_yearly_safety.py
```

 **Note:** The three scripts above are no longer needed for visualization. Power BI dashboards have replaced them.

---

## Power BI Dashboards

The primary visualization tools are Power BI files:

- `PowerBI-Visualizations/COSC 301 Consolidated.pbix`
- `PowerBI-Visualizations/COSC 301 Traffic.pbix`

Open these in Power BI Desktop to explore interactive dashboards.

---

## Machine Learning Models

### Train Models
Run each training script separately:

```bash
python project/models_train/collision_prediction.py
python project/models_train/break_and_enter_prediction.py
python project/models_train/auto_theft_prediction.py
python project/models_train/assault_prediction.py
```

These produce the following model bundles in `project/outputs/`:

- `assault_model_bundle.joblib`
- `auto_theft_model_bundle.joblib`
- `break_and_enter_model_bundle.joblib`
- `collision_model_bundle.joblib`

### Test Models in Terminal

```bash
python project/model_orchestrator.py
```

---

## Web Application

### Run Locally

**Backend:**

```bash
cd project/crime_prediction_api
python run.py
```

**Frontend:**

```bash
cd front-end
npm run dev
```

Then open your browser to: [http://localhost:3000/](http://localhost:3000/)

### Hosted Version

The application is also available online at:  
[g2-crimeanalysis.com](https://victorious-flower-0fa5e4b10.2.azurestaticapps.net/)

---

## Project Structure (Simplified)

```
.
├── 301_crimeanalysis_report.pdf
├── project/
│   ├── DB_setup_crime.py
│   ├── DB_setup_traffic.py
│   ├── sub_crime_setup.py
│   ├── DB_querying.py
│   ├── model_orchestrator.py
│   ├── outputs/               # Model bundles
│   ├── models_train/          # Training scripts
│   └── crime_prediction_api/  # Backend API
├── front-end/                 # NextJS frontend
├── PowerBI-Visualizations/
│   ├── COSC 301 Consolidated.pbix
│   └── COSC 301 Traffic.pbix
└── data/                      # Raw and cleaned CSVs
```

---

## Notes

- All cleaning steps must be run in the exact order shown to ensure data integrity.
- The `.env` file must contain valid PostgreSQL credentials before running any database scripts.
- Power BI dashboards are the recommended visualization method (the Python plotting scripts are legacy).
- The web app provides a user-friendly interface to model predictions.

