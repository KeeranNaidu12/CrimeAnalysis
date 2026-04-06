"""
Crime Prediction Orchestrator
==============================
Loads all trained crime prediction models and provides an interactive interface
for making predictions over custom date ranges.

Usage:
    python model_orchestrator.py

Features:
    - Load all 4 crime prediction models
    - Choose crime type to predict
    - Predict over custom date ranges (min 3 days)
    - Option to filter by specific neighbourhood
    - Support neighbourhood name or ID input (e.g., "20" or "Alderwood (20)")
    - Paginated neighbourhood list view
    - Aggregated results with statistics
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tabulate import tabulate
import re
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = Path("project/outputs")
MODEL_FILES = {
    "collision": "collision_model_bundle.joblib",
    "assault": "assault_model_bundle.joblib",
    "auto_theft": "auto_theft_model_bundle.joblib",
    "break_and_enter": "break_and_enter_model_bundle.joblib"
}

# Display names for crimes
CRIME_DISPLAY_NAMES = {
    "collision": "🚗 Traffic Collision",
    "assault": "👊 Assault",
    "auto_theft": "🚙 Auto Theft",
    "break_and_enter": "🏠 Break & Enter"
}

# Colour codes for terminal output
class Colours:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# ═══════════════════════════════════════════════════════════════════════════════
#  NEIGHBOURHOOD UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_neighbourhood_id(name_with_id):
    """Extract numeric ID from neighbourhood name like 'Alderwood (20)'"""
    match = re.search(r'\((\d+)\)', name_with_id)
    if match:
        return int(match.group(1))
    return None


def clean_neighbourhood_name(name_with_id):
    """Extract clean name without ID: 'Alderwood (20)' -> 'Alderwood'"""
    match = re.match(r'^(.+?)\s*\(\d+\)$', name_with_id)
    if match:
        return match.group(1).strip()
    return name_with_id


def create_neighbourhood_mappings(neighbourhoods):
    """
    Create mappings between neighbourhood names and IDs.
    Returns: 
        - id_to_name: dict {id: clean_name}
        - name_to_id: dict {clean_name: id}
        - display_list: list of formatted strings
    """
    id_to_name = {}
    name_to_id = {}
    display_list = []
    
    for nh in neighbourhoods:
        # Extract ID if present in format "Name (ID)"
        nh_id = extract_neighbourhood_id(nh)
        clean_name = clean_neighbourhood_name(nh)
        
        if nh_id is not None:
            id_to_name[nh_id] = clean_name
            name_to_id[clean_name] = nh_id
            display_list.append(f"{clean_name} ({nh_id})")
        else:
            # If no ID found, create a hash-based ID for consistency
            # This is less common but handle gracefully
            fallback_id = hash(nh) % 10000
            id_to_name[fallback_id] = nh
            name_to_id[nh] = fallback_id
            display_list.append(f"{nh} ({fallback_id})")
    
    return id_to_name, name_to_id, display_list


def find_neighbourhood_by_input(user_input, id_to_name, name_to_id, all_neighbourhoods):
    """
    Find neighbourhood by ID number or name.
    Returns the original neighbourhood string from all_neighbourhoods.
    """
    # Try as numeric ID first
    if user_input.isdigit():
        nh_id = int(user_input)
        if nh_id in id_to_name:
            clean_name = id_to_name[nh_id]
            # Find the original formatted name
            for nh in all_neighbourhoods:
                if clean_neighbourhood_name(nh) == clean_name:
                    return nh
        else:
            return None
    
    # Try as name (case-insensitive, partial match supported)
    user_input_lower = user_input.lower()
    matches = []
    
    for nh in all_neighbourhoods:
        clean_name = clean_neighbourhood_name(nh).lower()
        if clean_name == user_input_lower:
            return nh
        if user_input_lower in clean_name:
            matches.append(nh)
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return matches  # Return list of matches for disambiguation
    
    return None


def display_neighbourhood_list(all_neighbourhoods, page=1, page_size=20):
    """Display paginated neighbourhood list"""
    total = len(all_neighbourhoods)
    total_pages = (total + page_size - 1) // page_size
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)
    
    print(f"\n{Colours.BOLD}🏘️ Neighbourhoods ({start_idx + 1}-{end_idx} of {total}){Colours.END}")
    print(f"{Colours.HEADER}{'-'*60}{Colours.END}")
    
    for i, nh in enumerate(all_neighbourhoods[start_idx:end_idx], start=start_idx + 1):
        # Display with index number for easy reference
        nh_id = extract_neighbourhood_id(nh)
        if nh_id:
            print(f"  {i:3d}. {nh}  [{Colours.BLUE}ID: {nh_id}{Colours.END}]")
        else:
            print(f"  {i:3d}. {nh}")
    
    print(f"{Colours.HEADER}{'-'*60}{Colours.END}")
    print(f"  Page {page}/{total_pages}  |  Commands: [n]ext, [p]revious, [s]earch, [q]uit")
    return total_pages


def interactive_neighbourhood_selection(model_bundle):
    """
    Interactive neighbourhood selection with pagination and search.
    Returns selected neighbourhood name or None.
    """
    le = model_bundle['label_encoder']
    all_neighbourhoods = le.classes_
    
    # Create mappings
    id_to_name, name_to_id, display_list = create_neighbourhood_mappings(all_neighbourhoods)
    
    print(f"\n{Colours.BOLD}🏘️ Neighbourhood Filter{Colours.END}")
    print(f"{Colours.HEADER}{'-'*60}{Colours.END}")
    print(f"  You can select by:")
    print(f"    • Neighbourhood ID (e.g., '20')")
    print(f"    • Neighbourhood name (e.g., 'Alderwood')")
    print(f"    • Partial name (e.g., 'wood')")
    print(f"    • Browse the full list")
    
    choice = input(f"\n{Colours.BOLD}Do you want to filter by neighbourhood? (y/n/browse): {Colours.END}").strip().lower()
    
    if choice == 'n':
        return None
    
    if choice == 'browse' or choice == 'b':
        page = 1
        while True:
            total_pages = display_neighbourhood_list(all_neighbourhoods, page)
            
            cmd = input(f"\n{Colours.BOLD}Command: {Colours.END}").strip().lower()
            
            if cmd == 'n' or cmd == 'next':
                if page < total_pages:
                    page += 1
                else:
                    print(f"{Colours.YELLOW}Already on last page.{Colours.END}")
            elif cmd == 'p' or cmd == 'prev' or cmd == 'previous':
                if page > 1:
                    page -= 1
                else:
                    print(f"{Colours.YELLOW}Already on first page.{Colours.END}")
            elif cmd == 's' or cmd == 'search':
                search_term = input(f"Enter search term (name or ID): ").strip()
                result = find_neighbourhood_by_input(search_term, id_to_name, name_to_id, all_neighbourhoods)
                
                if result is None:
                    print(f"{Colours.RED}No neighbourhood found matching '{search_term}'{Colours.END}")
                elif isinstance(result, list):
                    print(f"\n{Colours.YELLOW}Multiple matches found:{Colours.END}")
                    for i, nh in enumerate(result[:10], 1):
                        nh_id = extract_neighbourhood_id(nh)
                        if nh_id:
                            print(f"    {i}. {nh} (ID: {nh_id})")
                        else:
                            print(f"    {i}. {nh}")
                    
                    # Let user choose from matches
                    if len(result) > 1:
                        idx_choice = input(f"\nEnter number (1-{min(10, len(result))}) or 'q' to cancel: ").strip()
                        if idx_choice.isdigit():
                            idx = int(idx_choice) - 1
                            if 0 <= idx < len(result):
                                selected = result[idx]
                                print(f"{Colours.GREEN}✓ Selected: {selected}{Colours.END}")
                                return selected
                else:
                    print(f"{Colours.GREEN}✓ Found: {result}{Colours.END}")
                    return result
            elif cmd == 'q' or cmd == 'quit':
                return None
            else:
                # Try to interpret as a neighbourhood selection
                result = find_neighbourhood_by_input(cmd, id_to_name, name_to_id, all_neighbourhoods)
                if result and not isinstance(result, list):
                    print(f"{Colours.GREEN}✓ Selected: {result}{Colours.END}")
                    return result
                elif result and isinstance(result, list):
                    print(f"{Colours.YELLOW}Multiple matches found. Please be more specific.{Colours.END}")
                else:
                    print(f"{Colours.RED}Invalid command. Use n/p/s/q or enter neighbourhood name/ID.{Colours.END}")
    
    else:
        # Direct input mode
        while True:
            neighbourhood_input = input(f"\nEnter neighbourhood name or ID: ").strip()
            result = find_neighbourhood_by_input(neighbourhood_input, id_to_name, name_to_id, all_neighbourhoods)
            
            if result is None:
                print(f"{Colours.RED}Neighbourhood not found. Try 'browse' to see the list.{Colours.END}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
            elif isinstance(result, list):
                print(f"\n{Colours.YELLOW}Multiple matches found:{Colours.END}")
                for i, nh in enumerate(result[:10], 1):
                    nh_id = extract_neighbourhood_id(nh)
                    if nh_id:
                        print(f"    {i}. {nh} (ID: {nh_id})")
                    else:
                        print(f"    {i}. {nh}")
                
                if len(result) > 1:
                    idx_choice = input(f"\nEnter number (1-{min(10, len(result))}) or 'q' to cancel: ").strip()
                    if idx_choice.isdigit():
                        idx = int(idx_choice) - 1
                        if 0 <= idx < len(result):
                            selected = result[idx]
                            print(f"{Colours.GREEN}✓ Selected: {selected}{Colours.END}")
                            return selected
                    else:
                        continue
            else:
                print(f"{Colours.GREEN}✓ Selected: {result}{Colours.END}")
                return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(crime_type):
    """Load a specific crime prediction model bundle"""
    model_path = MODELS_DIR / MODEL_FILES[crime_type]
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"{Colours.BLUE}    Loading {CRIME_DISPLAY_NAMES[crime_type]} model...{Colours.END}")
    bundle = joblib.load(model_path)
    
    return {
        "model": bundle["model"],
        "label_encoder": bundle["label_encoder"],
        "threshold": bundle["threshold"],
        "feature_cols": bundle["feature_cols"],
        "window_days": bundle.get("window_days", 3),  # Default to 3 if not specified
        "crime_type": crime_type
    }


def load_all_models():
    """Load all available crime prediction models"""
    models = {}
    
    print(f"\n{Colours.HEADER}{'='*60}{Colours.END}")
    print(f"{Colours.BOLD}Loading Crime Prediction Models{Colours.END}")
    print(f"{Colours.HEADER}{'='*60}{Colours.END}\n")
    
    for crime_type in MODEL_FILES.keys():
        try:
            models[crime_type] = load_model(crime_type)
            print(f"    ✅ Loaded {CRIME_DISPLAY_NAMES[crime_type]} (window: {models[crime_type]['window_days']} days)")
        except Exception as e:
            print(f"    ❌ Failed to load {crime_type}: {e}")
    
    print(f"\n{Colours.GREEN}✅ Successfully loaded {len(models)} models{Colours.END}\n")
    return models


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def to_timestamp(date_obj):
    """Convert datetime or string to pandas Timestamp for consistent attribute access"""
    if isinstance(date_obj, datetime):
        return pd.Timestamp(date_obj)
    elif isinstance(date_obj, str):
        return pd.Timestamp(date_obj)
    else:
        return date_obj


def create_prediction_row(neighbourhood, window_start, model_bundle, latest_state=None):
    """
    Create a feature row for a single neighbourhood and window start date.
    This replicates the logic from the training scripts.
    """
    from math import sin, cos, pi
    
    # Convert to pandas Timestamp if needed
    window_start = to_timestamp(window_start)
    
    le = model_bundle['label_encoder']
    
    # Helper functions
    def date_to_season(dt):
        m = dt.month
        if m in (12, 1, 2): return 0
        if m in (3, 4, 5): return 1
        if m in (6, 7, 8): return 2
        return 3
    
    def is_holiday(dt):
        statutory = {(1, 1), (2, 14), (7, 1), (11, 11), (12, 25), (12, 26)}
        return 1 if (dt.month, dt.day) in statutory else 0
    
    def weekday_to_num(weekday):
        weekday_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6,
        }
        return weekday_map.get(weekday, 0)
    
    # Build the feature row
    row = {}
    
    # Basic features
    row['neighbourhood_enc'] = le.transform([neighbourhood])[0]
    row['week_day_num'] = weekday_to_num(window_start.strftime("%A"))
    row['season_num'] = date_to_season(window_start)
    row['holiday'] = is_holiday(window_start)
    
    # Cyclic features
    row['month_sin'] = sin(2 * pi * window_start.month / 12)
    row['month_cos'] = cos(2 * pi * window_start.month / 12)
    row['period_sin'] = sin(2 * pi * window_start.isocalendar().week / 52)
    row['period_cos'] = cos(2 * pi * window_start.isocalendar().week / 52)
    row['year_continuous'] = window_start.year + window_start.dayofyear / 365.25
    
    # Historical features (use latest_state if available, otherwise defaults)
    if latest_state is not None and neighbourhood in latest_state.index:
        hist_row = latest_state.loc[neighbourhood]
        row['hist_rate_3w'] = hist_row.get('hist_rate_3w', 0.0)
        row['hist_rate_6w'] = hist_row.get('hist_rate_6w', 0.0)
        row['periods_since_last'] = hist_row.get('periods_since_last', 0)
        row['ewma_3w'] = hist_row.get('ewma_3w', 0.0)
        row['rate_change_3w'] = hist_row.get('rate_change_3w', 0.0)
        row['seasonal_pattern'] = hist_row.get('seasonal_pattern', 0.0)
        row['global_yoy_change'] = hist_row.get('global_yoy_change', 0.0)
        
        # Handle crime-specific trend/density fields
        crime_type = model_bundle['crime_type']
        if crime_type == 'collision':
            row['collision_trend'] = hist_row.get('collision_trend', 0.0)
            row['recent_collision_density'] = hist_row.get('recent_collision_density', 0.0)
            row['recent_assault_density'] = 0.0
            row['assault_trend'] = 0.0
        elif crime_type == 'assault':
            row['assault_trend'] = hist_row.get('assault_trend', 0.0)
            row['recent_assault_density'] = hist_row.get('recent_assault_density', 0.0)
            row['collision_trend'] = 0.0
            row['recent_collision_density'] = 0.0
        elif crime_type == 'auto_theft':
            row['auto_theft_trend'] = hist_row.get('auto_theft_trend', 0.0)
            row['recent_auto_theft_density'] = hist_row.get('recent_auto_theft_density', 0.0)
        elif crime_type == 'break_and_enter':
            row['break_and_enter_trend'] = hist_row.get('break_and_enter_trend', 0.0)
            row['recent_break_and_enter_density'] = hist_row.get('recent_break_and_enter_density', 0.0)
    else:
        # Default values
        row['hist_rate_3w'] = 0.0
        row['hist_rate_6w'] = 0.0
        row['periods_since_last'] = 0
        row['ewma_3w'] = 0.0
        row['rate_change_3w'] = 0.0
        row['seasonal_pattern'] = 0.0
        row['global_yoy_change'] = 0.0
        
        crime_type = model_bundle['crime_type']
        if crime_type == 'collision':
            row['collision_trend'] = 0.0
            row['recent_collision_density'] = 0.0
            row['recent_assault_density'] = 0.0
            row['assault_trend'] = 0.0
        elif crime_type == 'assault':
            row['assault_trend'] = 0.0
            row['recent_assault_density'] = 0.0
            row['collision_trend'] = 0.0
            row['recent_collision_density'] = 0.0
        elif crime_type == 'auto_theft':
            row['auto_theft_trend'] = 0.0
            row['recent_auto_theft_density'] = 0.0
        elif crime_type == 'break_and_enter':
            row['break_and_enter_trend'] = 0.0
            row['recent_break_and_enter_density'] = 0.0
    
    return row


def predict_for_date_range(model_bundle, start_date, end_date, neighbourhood_filter=None):
    """
    Make predictions for all 3-day windows within a date range.
    
    Args:
        model_bundle: Loaded model bundle
        start_date: Start date (datetime or string YYYY-MM-DD)
        end_date: End date (datetime or string YYYY-MM-DD)
        neighbourhood_filter: Optional specific neighbourhood to predict for
    
    Returns:
        DataFrame with predictions for each window
    """
    # Convert to pandas Timestamp
    start_date = to_timestamp(start_date)
    end_date = to_timestamp(end_date)
    
    window_days = model_bundle['window_days']
    le = model_bundle['label_encoder']
    crime_type = model_bundle['crime_type']
    
    # Get all neighbourhoods
    all_neighbourhoods = le.classes_
    
    # Filter if specific neighbourhood requested
    if neighbourhood_filter:
        if neighbourhood_filter not in all_neighbourhoods:
            # Try to find by ID or name
            id_to_name, name_to_id, _ = create_neighbourhood_mappings(all_neighbourhoods)
            found = find_neighbourhood_by_input(neighbourhood_filter, id_to_name, name_to_id, all_neighbourhoods)
            if found and not isinstance(found, list):
                neighbourhood_filter = found
            else:
                raise ValueError(f"Neighbourhood '{neighbourhood_filter}' not found.")
        neighbourhoods = [neighbourhood_filter]
    else:
        neighbourhoods = all_neighbourhoods
    
    # Generate windows
    current = start_date
    windows = []
    
    while current <= end_date:
        window_end = current + timedelta(days=window_days - 1)
        windows.append({
            "window_start": current,
            "window_end": window_end,
            "window_label": f"{current.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}"
        })
        current += timedelta(days=window_days)
    
    print(f"\n{Colours.BLUE}    Generating predictions for {len(windows)} x {len(neighbourhoods)} = {len(windows) * len(neighbourhoods)} combinations...{Colours.END}")
    
    # For historical features, we need to simulate the latest state
    # Since we don't have full historical data, we'll use a simplified approach
    # In production, you'd want to load historical data from the database
    latest_state = None  # Simplified: use defaults
    
    # Generate predictions for each window
    all_predictions = []
    
    for i, window in enumerate(windows):
        window_start = window["window_start"]
        
        # Build feature matrix for this window
        rows = []
        for neighbourhood in neighbourhoods:
            row = create_prediction_row(neighbourhood, window_start, model_bundle, latest_state)
            rows.append(row)
        
        # Convert to DataFrame
        feature_cols = model_bundle['feature_cols']
        X_pred = pd.DataFrame(rows)[feature_cols]
        
        # Make predictions
        probs = model_bundle['model'].predict_proba(X_pred)[:, 1]
        predictions = (probs >= model_bundle['threshold']).astype(int)
        
        # Store results
        for j, neighbourhood in enumerate(neighbourhoods):
            all_predictions.append({
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "neighbourhood": neighbourhood,
                "probability": probs[j],
                "predicted": predictions[j],
                "risk_level": "HIGH" if probs[j] >= 0.7 else "MEDIUM" if probs[j] >= 0.4 else "LOW"
            })
    
    return pd.DataFrame(all_predictions)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT DISPLAY AND AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def display_aggregated_results(results_df, crime_display_name, neighbourhood_filter=None):
    """Display aggregated prediction results"""
    
    if results_df.empty:
        print(f"{Colours.RED}No predictions generated.{Colours.END}")
        return
    
    print(f"\n{Colours.HEADER}{'='*80}{Colours.END}")
    print(f"{Colours.BOLD}{crime_display_name} - Prediction Results{Colours.END}")
    print(f"{Colours.HEADER}{'='*80}{Colours.END}")
    
    # Summary statistics
    total_windows = results_df['window_start'].nunique()
    total_predictions = len(results_df)
    positive_predictions = results_df['predicted'].sum()
    
    print(f"\n{Colours.BOLD}📊 Summary Statistics:{Colours.END}")
    print(f"    Time periods analyzed: {total_windows}")
    print(f"    Total predictions made: {total_predictions:,}")
    print(f"    Predicted incidents: {positive_predictions:,} ({positive_predictions/total_predictions*100:.1f}%)")
    
    if not neighbourhood_filter:
        # Show top neighbourhoods by risk
        neighbourhood_risk = results_df.groupby('neighbourhood').agg({
            'probability': 'mean',
            'predicted': 'sum'
        }).sort_values('probability', ascending=False).head(10)
        
        print(f"\n{Colours.BOLD}🎯 Top 10 Highest Risk Neighbourhoods:{Colours.END}")
        # Extract IDs for display
        display_data = []
        for idx, row in neighbourhood_risk.reset_index().iterrows():
            nh_name = row['neighbourhood']
            nh_id = extract_neighbourhood_id(nh_name)
            if nh_id:
                display_name = f"{clean_neighbourhood_name(nh_name)} (ID:{nh_id})"
            else:
                display_name = nh_name
            display_data.append([display_name, row['probability'], row['predicted']])
        
        print(tabulate(
            display_data,
            headers=['Neighbourhood', 'Avg Risk Prob', 'Predicted Count'],
            tablefmt='grid',
            floatfmt=['', '.1%', 'd']
        ))
    
    # Show results by time window
    print(f"\n{Colours.BOLD}📅 Results by Time Window:{Colours.END}")
    
    window_summary = results_df.groupby(['window_start', 'window_end']).agg({
        'probability': ['mean', 'max'],
        'predicted': 'sum'
    }).round(3)
    window_summary.columns = ['Avg Risk', 'Max Risk', 'Predicted Count']
    window_summary = window_summary.reset_index()
    window_summary['Window'] = window_summary['window_start'].dt.strftime('%Y-%m-%d') + ' → ' + window_summary['window_end'].dt.strftime('%Y-%m-%d')
    
    # Display with risk colour coding
    for _, row in window_summary.iterrows():
        risk_indicator = "🔴" if row['Max Risk'] >= 0.7 else "🟡" if row['Max Risk'] >= 0.4 else "🟢"
        print(f"    {risk_indicator} {row['Window']:<30} | Avg Risk: {row['Avg Risk']:.1%} | Max Risk: {row['Max Risk']:.1%} | Predicted: {row['Predicted Count']}")
    
    # High-risk alerts
    high_risk = results_df[results_df['risk_level'] == 'HIGH'].sort_values('probability', ascending=False)
    if not high_risk.empty:
        print(f"\n{Colours.RED}{Colours.BOLD}⚠️ HIGH RISK ALERTS (Probability ≥ 70%):{Colours.END}")
        high_risk_display = high_risk.head(20).copy()
        high_risk_display['window_label'] = high_risk_display['window_start'].dt.strftime('%Y-%m-%d') + ' → ' + high_risk_display['window_end'].dt.strftime('%Y-%m-%d')
        
        # Clean neighbourhood names for display
        high_risk_display['neighbourhood_clean'] = high_risk_display['neighbourhood'].apply(
            lambda x: f"{clean_neighbourhood_name(x)} (ID:{extract_neighbourhood_id(x)})" if extract_neighbourhood_id(x) else x
        )
        
        print(tabulate(
            high_risk_display[['window_label', 'neighbourhood_clean', 'probability']].head(10),
            headers=['Time Window', 'Neighbourhood', 'Probability'],
            tablefmt='grid',
            floatfmt=['', '', '.1%']
        ))
    
    return window_summary


def export_results(results_df, crime_type, start_date, end_date, output_format="csv"):
    """Export prediction results to file"""
    output_dir = Path("project/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{crime_type}_predictions_{start_date}_to_{end_date}_{timestamp}"
    
    if output_format == "csv":
        filepath = output_dir / f"{filename}.csv"
        results_df.to_csv(filepath, index=False)
        print(f"\n{Colours.GREEN}✅ Results exported to: {filepath}{Colours.END}")
    elif output_format == "excel":
        filepath = output_dir / f"{filename}.xlsx"
        results_df.to_excel(filepath, index=False)
        print(f"\n{Colours.GREEN}✅ Results exported to: {filepath}{Colours.END}")
    
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
#  USER INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def select_crime_type(models):
    """Interactive crime type selection"""
    print(f"\n{Colours.BOLD}Available Crime Prediction Models:{Colours.END}")
    print(f"{Colours.HEADER}{'-'*50}{Colours.END}")
    
    crime_list = list(models.keys())
    for i, crime in enumerate(crime_list, 1):
        model_info = models[crime]
        print(f"  {i}. {CRIME_DISPLAY_NAMES[crime]} (window: {model_info['window_days']} days)")
    
    print(f"{Colours.HEADER}{'-'*50}{Colours.END}")
    
    while True:
        try:
            choice = input(f"\n{Colours.BOLD}Select crime type (1-{len(crime_list)}): {Colours.END}").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(crime_list):
                return crime_list[idx]
            else:
                print(f"{Colours.RED}Invalid selection. Please enter a number between 1 and {len(crime_list)}.{Colours.END}")
        except ValueError:
            print(f"{Colours.RED}Please enter a valid number.{Colours.END}")


def get_date_range():
    """Get date range from user"""
    print(f"\n{Colours.BOLD}📅 Date Range Selection{Colours.END}")
    print(f"{Colours.HEADER}{'-'*50}{Colours.END}")
    
    while True:
        try:
            start_str = input(f"Enter start date (YYYY-MM-DD): ").strip()
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            
            end_str = input(f"Enter end date (YYYY-MM-DD): ").strip()
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            
            if end_date < start_date:
                print(f"{Colours.RED}End date must be after start date.{Colours.END}")
                continue
            
            days_diff = (end_date - start_date).days + 1
            if days_diff < 3:
                print(f"{Colours.RED}Date range must be at least 3 days (minimum window size).{Colours.END}")
                continue
            
            print(f"\n{Colours.GREEN}✓ Selected range: {days_diff} days ({start_str} to {end_str}){Colours.END}")
            return start_date, end_date
            
        except ValueError:
            print(f"{Colours.RED}Invalid date format. Please use YYYY-MM-DD.{Colours.END}")


def get_export_option():
    """Ask user if they want to export results"""
    print(f"\n{Colours.BOLD}💾 Export Options{Colours.END}")
    print(f"{Colours.HEADER}{'-'*50}{Colours.END}")
    
    choice = input(f"Do you want to export results to file? (y/n): ").strip().lower()
    
    if choice == 'y':
        print(f"  Export formats: csv, excel")
        format_choice = input(f"Choose format (csv/excel): ").strip().lower()
        if format_choice in ['csv', 'excel']:
            return format_choice
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main orchestrator function"""
    print(f"\n{Colours.HEADER}{'='*60}{Colours.END}")
    print(f"{Colours.BOLD}{'🔮 CRIME PREDICTION ORCHESTRATOR'.center(60)}{Colours.END}")
    print(f"{Colours.HEADER}{'='*60}{Colours.END}")
    
    # Load all models
    models = load_all_models()
    
    if not models:
        print(f"{Colours.RED}No models loaded. Please ensure models exist in {MODELS_DIR}{Colours.END}")
        return
    
    # Main interaction loop
    while True:
        # Select crime type
        crime_type = select_crime_type(models)
        model_bundle = models[crime_type]
        
        # Get date range
        start_date, end_date = get_date_range()
        
        # Get neighbourhood filter (optional) - with enhanced selection
        neighbourhood_filter = interactive_neighbourhood_selection(model_bundle)
        
        if neighbourhood_filter:
            print(f"{Colours.GREEN}✓ Predicting for: {neighbourhood_filter}{Colours.END}")
        else:
            print(f"{Colours.BLUE}✓ Predicting for ALL neighbourhoods{Colours.END}")
        
        # Make predictions
        print(f"\n{Colours.YELLOW}⏳ Generating predictions...{Colours.END}")
        try:
            results_df = predict_for_date_range(
                model_bundle, 
                start_date, 
                end_date, 
                neighbourhood_filter
            )
            
            # Display results
            display_aggregated_results(results_df, CRIME_DISPLAY_NAMES[crime_type], neighbourhood_filter)
            
            # Export if requested
            export_format = get_export_option()
            if export_format:
                start_str = start_date.strftime("%Y%m%d")
                end_str = end_date.strftime("%Y%m%d")
                export_results(results_df, crime_type, start_str, end_str, export_format)
            
        except Exception as e:
            print(f"{Colours.RED}Error during prediction: {e}{Colours.END}")
            import traceback
            traceback.print_exc()
        
        # Ask if user wants to make another prediction
        print(f"\n{Colours.HEADER}{'-'*50}{Colours.END}")
        again = input(f"\n{Colours.BOLD}Make another prediction? (y/n): {Colours.END}").strip().lower()
        if again != 'y':
            print(f"\n{Colours.GREEN}Thank you for using Crime Prediction Orchestrator!{Colours.END}")
            break


if __name__ == "__main__":
    main()