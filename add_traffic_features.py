import pandas as pd
import holidays
from pathlib import Path

# ================= CONFIGURATION =================
FILES_TO_PROCESS = [
    {'input': r'project\DB_csv\Traffic_Collisions_Data.csv', 'date_col': 'OCC_DATE'},
    {'input': r'project\DB_csv\Open_Consolidated_Data_updated_deduplicated.csv', 'date_col': 'OCC_DATE'}
]

# Filter settings (change if needed)
FILTER_COL = 'NEIGHBOURHOOD_158'
FILTER_VALUE = 'NSA'
# =================================================

def add_time_features(df, date_col, holiday_set):
    if date_col not in df.columns:
        raise ValueError(f"❌ Column '{date_col}' not found. Available: {list(df.columns[:5])}...")
        
    print(f"  🔍 Parsing '{date_col}' as datetime...")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # 1️⃣ Weekday
    df['week_day'] = df[date_col].dt.day_name().str.capitalize()
    
    # 2️⃣ Season (Meteorological)
    def get_season(month):
        if pd.isna(month): return None
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        if month in [9, 10, 11]: return 'Autumn'
        return 'Winter'
    df['season'] = df[date_col].dt.month.apply(get_season)
    
    # 3️⃣ Holiday (Vectorized check)
    valid_mask = df[date_col].notna()
    df.loc[valid_mask, 'holiday'] = df.loc[valid_mask, date_col].dt.date.isin(holiday_set)
    df['holiday'] = df['holiday'].fillna(False)
    
    return df

def main():
    # Precompute holiday set once (Canada + Ontario)
    print("📅 Loading Canada/Ontario holiday calendar...")
    ca_holidays = holidays.country_holidays('CA', subdiv='ON')
    holiday_set = set(ca_holidays.keys())
    print("✅ Holiday calendar loaded.\n")
    
    for config in FILES_TO_PROCESS:
        input_path = config['input']
        date_col = config['date_col']
        
        if not Path(input_path).exists():
            print(f"⚠️  File not found: {input_path} | Skipping...\n")
            continue
            
        print(f"📂 Processing: {Path(input_path).name}")
        try:
            df = pd.read_csv(input_path)
            print(f"  📊 Original shape: {df.shape}")
            
            # 🧹 Filter out NSA rows
            if FILTER_COL in df.columns:
                before_count = len(df)
                # Safe string comparison that handles NaNs gracefully
                df = df[df[FILTER_COL].astype(str).str.strip() != FILTER_VALUE]
                print(f"  🗑️  Removed {before_count - len(df)} rows where {FILTER_COL} == '{FILTER_VALUE}'")
            else:
                print(f"  ⚠️  Column '{FILTER_COL}' not found. Skipping NSA filter.")
            
            # ⏱️ Add time features
            df = add_time_features(df, date_col, holiday_set)
            
            # 💾 Save with _enhanced suffix
            path_obj = Path(input_path)
            output_path = path_obj.parent / f"{path_obj.stem}_enhanced{path_obj.suffix}"
            df.to_csv(output_path, index=False)
            
            print(f"✅ Saved: {output_path}")
            print("🔎 Preview:")
            cols_to_show = [date_col, 'week_day', 'season', 'holiday']
            if FILTER_COL in df.columns:
                cols_to_show.insert(1, FILTER_COL)
            print(df[cols_to_show].head(4), "\n" + "-"*60)
            
        except Exception as e:
            print(f"❌ Failed to process {input_path}: {e}\n")
            
    print("🎉 All files processed successfully!")

if __name__ == '__main__':
    main()