"""
Traffic Collision Prediction Query System
==========================================
Interactive tool for making predictions using the trained XGBoost model.

Usage:
    python query_predictions.py

Features:
    1. Predict collisions for a specific neighborhood and date range
    2. Predict collisions across all neighborhoods for a specific date
    3. Interactive menu-driven interface
    4. Confidence levels and risk assessment
"""

import os
import pickle
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

# ============================================
# CONFIGURATION
# ============================================

MODEL_PATH = os.path.join("model_data", "collision_xgboost_model.pkl")
OUTPUT_DIR = "query_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature columns (must match training)
FEATURE_COLS = [
    "neighbourhood_enc",
    "month",
    "quarter",
    "week_of_year",
    "year",
    "season",
    "is_holiday_season",
    "hist_rate_4w",
    "hist_rate_8w",
    "weeks_since_last",
]

# Helper functions for feature engineering
def _season(month: int) -> int:
    """Convert month to season (0=Winter, 1=Spring, 2=Summer, 3=Fall)"""
    if month in [12, 1, 2]:
        return 0
    if month in [3, 4, 5]:
        return 1
    if month in [6, 7, 8]:
        return 2
    return 3

def _holiday_season(month: int) -> int:
    """Check if month is in holiday season (Nov-Jan)"""
    return int(month in [11, 12, 1])

def get_confidence_band(probability: float) -> str:
    """Convert probability to confidence band"""
    if probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Medium"
    elif probability < 0.75:
        return "High"
    else:
        return "Very High"

def get_risk_level(probability: float) -> str:
    """Get risk level description"""
    if probability < 0.3:
        return "🟢 Low Risk"
    elif probability < 0.5:
        return "🟡 Moderate Risk"
    elif probability < 0.7:
        return "🟠 Elevated Risk"
    else:
        return "🔴 High Risk"


# ============================================
# MODEL LOADING
# ============================================

class CollisionPredictor:
    """Wrapper class for the trained collision prediction model"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """Load the trained model and required components"""
        print(f"\n📂 Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.feature_cols = bundle["feature_cols"]
        self.cutoff_year = bundle.get("cutoff_year", 2024)
        self.trained_at = bundle.get("trained_at", "Unknown")
        
        print(f"✅ Model loaded successfully")
        print(f"   - Trained on data before {self.cutoff_year}")
        print(f"   - Training date: {self.trained_at}")
        print(f"   - Features: {len(self.feature_cols)}")
        
        # Load historical data for lag features
        self.historical_data = self._load_historical_data()
        
    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical collision data for computing lag features"""
        try:
            # Try to load the weekly predictions file if it exists
            weekly_path = os.path.join("model_data", "collision_weekly_predictions.csv")
            if os.path.exists(weekly_path):
                df = pd.read_csv(weekly_path)
                df['week'] = pd.to_datetime(df['week'])
                print(f"   - Loaded historical data from {weekly_path}")
                return df
            else:
                print("   - Warning: No historical data found. Using default lag values.")
                return None
        except Exception as e:
            print(f"   - Warning: Could not load historical data: {e}")
            return None
    
    def get_neighbourhoods(self) -> List[str]:
        """Get list of all neighbourhoods the model knows"""
        return self.label_encoder.classes_.tolist()
    
    def get_historical_rate(self, neighbourhood: str, date: datetime) -> Tuple[float, float, int]:
        """Get historical collision rate for a neighbourhood up to a given date
        
        Returns:
            Tuple of (hist_rate_4w, hist_rate_8w, weeks_since_last)
        """
        if self.historical_data is None:
            # Return default values if no historical data
            return (0.1, 0.1, 4)
        
        # Filter data for this neighbourhood and dates before the target
        nh_data = self.historical_data[
            (self.historical_data['neighbourhood_158'] == neighbourhood) &
            (self.historical_data['week'] < date)
        ].sort_values('week')
        
        if len(nh_data) == 0:
            return (0.05, 0.05, 8)
        
        # Calculate rolling averages from actual collisions
        # For simplicity, use probability if available, otherwise use actual collisions
        if 'collision_probability' in nh_data.columns:
            # Use predicted probabilities as proxy for historical rates
            hist_4w = nh_data['collision_probability'].tail(4).mean()
            hist_8w = nh_data['collision_probability'].tail(8).mean()
        else:
            # Use actual collisions
            hist_4w = nh_data['actual_collision'].tail(4).mean()
            hist_8w = nh_data['actual_collision'].tail(8).mean()
        
        # Calculate weeks since last collision
        if 'actual_collision' in nh_data.columns:
            last_collision = nh_data[nh_data['actual_collision'] == 1]
            if len(last_collision) > 0:
                last_date = last_collision['week'].iloc[-1]
                weeks_since = (date - last_date).days // 7
            else:
                weeks_since = 8
        else:
            weeks_since = 4
        
        return (hist_4w, hist_8w, weeks_since)
    
    def prepare_features(self, neighbourhood: str, date: datetime) -> pd.DataFrame:
        """Prepare features for a single prediction"""
        month = date.month
        iso = date.isocalendar()
        
        # Get historical rates
        hist_4w, hist_8w, weeks_since = self.get_historical_rate(neighbourhood, date)
        
        # Encode neighbourhood
        try:
            nh_enc = self.label_encoder.transform([neighbourhood])[0]
        except ValueError:
            # Handle unknown neighbourhood
            similar_nh = self.label_encoder.transform([neighbourhood])[0] if neighbourhood in self.label_encoder.classes_ else None
            if similar_nh is None:
                # Fallback to first neighbourhood (not ideal, but better than crashing)
                print(f"⚠️ Warning: Neighbourhood '{neighbourhood}' not in training data. Using fallback.")
                nh_enc = 0
        
        features = {
            "neighbourhood_enc": nh_enc,
            "month": month,
            "quarter": (month - 1) // 3 + 1,
            "week_of_year": iso[1],
            "year": iso[0],
            "season": _season(month),
            "is_holiday_season": _holiday_season(month),
            "hist_rate_4w": hist_4w,
            "hist_rate_8w": hist_8w,
            "weeks_since_last": weeks_since,
        }
        
        return pd.DataFrame([features])
    
    def predict(self, neighbourhood: str, date: datetime) -> Dict:
        """Make a single prediction for a neighbourhood and date"""
        # Prepare features
        X = self.prepare_features(neighbourhood, date)
        
        # Make prediction
        proba = self.model.predict_proba(X[self.feature_cols])[0][1]
        pred_class = int(proba >= 0.5)
        
        return {
            "neighbourhood": neighbourhood,
            "date": date,
            "probability": proba,
            "prediction": pred_class,
            "confidence_band": get_confidence_band(proba),
            "risk_level": get_risk_level(proba),
        }
    
    def predict_week_range(self, neighbourhood: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Predict collisions for a neighbourhood over a date range (weekly intervals)"""
        results = []
        
        # Generate weekly dates
        current = start_date
        while current <= end_date:
            # Ensure we're predicting for the start of the week
            week_start = current - timedelta(days=current.weekday())
            prediction = self.predict(neighbourhood, week_start)
            results.append(prediction)
            current += timedelta(weeks=1)
        
        return pd.DataFrame(results)
    
    def predict_all_neighbourhoods(self, date: datetime) -> pd.DataFrame:
        """Predict collisions for all neighbourhoods on a specific date"""
        results = []
        
        for neighbourhood in self.get_neighbourhoods():
            try:
                prediction = self.predict(neighbourhood, date)
                results.append(prediction)
            except Exception as e:
                print(f"⚠️ Error predicting for {neighbourhood}: {e}")
                continue
        
        df = pd.DataFrame(results)
        return df.sort_values("probability", ascending=False)


# ============================================
# INTERACTIVE QUERY SYSTEM
# ============================================

def print_header():
    """Print application header"""
    print("\n" + "="*70)
    print("  🚗 TRAFFIC COLLISION PREDICTION QUERY SYSTEM 🚦")
    print("="*70)
    print(f"  Model trained on data before 2024 | Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print("="*70)

def print_menu():
    """Print main menu"""
    print("\n📋 QUERY OPTIONS:")
    print("  1. Predict collisions for a specific neighbourhood (date range)")
    print("  2. Predict collisions across ALL neighbourhoods (specific date)")
    print("  3. Compare multiple neighbourhoods (date range)")
    print("  4. Find high-risk dates for a neighbourhood")
    print("  5. Export results to CSV")
    print("  0. Exit")
    print("-"*70)

def validate_date(date_str: str) -> Optional[datetime]:
    """Validate and parse date input"""
    try:
        # Try multiple date formats
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        raise ValueError("Invalid date format")
    except Exception:
        print("❌ Invalid date format. Please use YYYY-MM-DD (e.g., 2024-03-15)")
        return None

def select_neighbourhood(predictor: CollisionPredictor) -> Optional[str]:
    """Interactive neighbourhood selection"""
    neighbourhoods = predictor.get_neighbourhoods()
    
    print("\n🏘️ AVAILABLE NEIGHBOURHOODS:")
    # Display neighbourhoods in columns
    for i, nh in enumerate(neighbourhoods[:20], 1):  # Show first 20
        print(f"  {i:2}. {nh}")
    
    if len(neighbourhoods) > 20:
        print(f"  ... and {len(neighbourhoods) - 20} more")
    
    print("\n💡 You can:")
    print("   - Enter a neighbourhood name (case-sensitive)")
    print("   - Enter a number (1-20) to select from the list")
    print("   - Type 'list' to see all neighbourhoods")
    print("   - Press Enter to go back")
    
    while True:
        choice = input("\n👉 Enter neighbourhood: ").strip()
        
        if choice == "":
            return None
        
        if choice.lower() == "list":
            print("\n📋 ALL NEIGHBOURHOODS:")
            for i, nh in enumerate(neighbourhoods, 1):
                print(f"  {i:3}. {nh}")
            continue
        
        # Check if it's a number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(neighbourhoods):
                return neighbourhoods[idx]
            else:
                print(f"❌ Number must be between 1 and {len(neighbourhoods)}")
                continue
        
        # Check if it's a neighbourhood name
        if choice in neighbourhoods:
            return choice
        else:
            print(f"❌ Neighbourhood '{choice}' not found")
            print("   Tip: Use 'list' to see all neighbourhoods")
            continue

def display_prediction(prediction: Dict):
    """Display a single prediction result"""
    print("\n" + "="*70)
    print(f"📍 NEIGHBOURHOOD: {prediction['neighbourhood']}")
    print(f"📅 DATE: {prediction['date'].strftime('%Y-%m-%d')} (Week starting)")
    print("-"*70)
    print(f"🎯 COLLISION PROBABILITY: {prediction['probability']:.2%}")
    print(f"📊 CONFIDENCE BAND: {prediction['confidence_band']}")
    print(f"⚠️  RISK LEVEL: {prediction['risk_level']}")
    
    # Visual indicator
    prob = prediction['probability']
    bar_length = 40
    filled = int(prob * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"   [{bar}] {prob:.1%}")
    
    if prediction['prediction'] == 1:
        print("\n🚨 PREDICTION: Collision LIKELY to occur")
    else:
        print("\n✅ PREDICTION: Collision UNLIKELY to occur")
    print("="*70)

def display_results_table(df: pd.DataFrame, title: str = "Predictions"):
    """Display results in a formatted table"""
    if df.empty:
        print("❌ No results to display")
        return
    
    # Prepare display dataframe
    display_df = df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.1%}")
    
    # Select columns for display
    cols = ['neighbourhood', 'date', 'probability', 'confidence_band', 'risk_level']
    if all(col in display_df.columns for col in cols):
        display_df = display_df[cols]
    
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print('='*70)
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    print('='*70)
    
    # Summary statistics
    avg_prob = df['probability'].mean()
    high_risk = len(df[df['probability'] >= 0.5])
    print(f"\n📈 Summary:")
    print(f"   - Total predictions: {len(df)}")
    print(f"   - Average probability: {avg_prob:.1%}")
    print(f"   - High-risk predictions (≥50%): {high_risk} ({high_risk/len(df)*100:.1f}%)")

def query_neighbourhood_range(predictor: CollisionPredictor):
    """Option 1: Predict for specific neighbourhood over date range"""
    print("\n" + "="*70)
    print("  PREDICT FOR SPECIFIC NEIGHBOURHOOD (DATE RANGE)")
    print("="*70)
    
    # Select neighbourhood
    neighbourhood = select_neighbourhood(predictor)
    if neighbourhood is None:
        return
    
    # Get date range
    print("\n📅 Enter date range (YYYY-MM-DD format):")
    
    start_str = input("Start date: ").strip()
    start_date = validate_date(start_str)
    if start_date is None:
        return
    
    end_str = input("End date: ").strip()
    end_date = validate_date(end_str)
    if end_date is None:
        return
    
    if start_date > end_date:
        print("❌ Start date must be before end date")
        return
    
    # Make predictions
    print(f"\n🔮 Predicting for {neighbourhood} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    results = predictor.predict_week_range(neighbourhood, start_date, end_date)
    
    # Display results
    display_results_table(results, f"Predictions for {neighbourhood}")
    
    # Ask to export
    if input("\n💾 Export to CSV? (y/n): ").lower() == 'y':
        filename = f"{neighbourhood}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        results.to_csv(filepath, index=False)
        print(f"✅ Results saved to: {filepath}")

def query_all_neighbourhoods(predictor: CollisionPredictor):
    """Option 2: Predict for all neighbourhoods on a specific date"""
    print("\n" + "="*70)
    print("  PREDICT FOR ALL NEIGHBOURHOODS (SPECIFIC DATE)")
    print("="*70)
    
    # Get date
    date_str = input("\n📅 Enter date (YYYY-MM-DD): ").strip()
    date = validate_date(date_str)
    if date is None:
        return
    
    # Make predictions
    print(f"\n🔮 Predicting for all neighbourhoods on {date.strftime('%Y-%m-%d')}...")
    results = predictor.predict_all_neighbourhoods(date)
    
    # Display results
    display_results_table(results, f"Predictions for {date.strftime('%Y-%m-%d')}")
    
    # Show top 5 high-risk neighbourhoods
    if len(results) > 0:
        print("\n🔥 TOP 5 HIGHEST RISK NEIGHBOURHOODS:")
        top5 = results.head(5)
        for i, row in top5.iterrows():
            print(f"   {i+1}. {row['neighbourhood']}: {row['probability']:.1%} ({row['risk_level']})")
    
    # Ask to export
    if input("\n💾 Export to CSV? (y/n): ").lower() == 'y':
        filename = f"all_neighbourhoods_{date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        results.to_csv(filepath, index=False)
        print(f"✅ Results saved to: {filepath}")

def compare_neighbourhoods(predictor: CollisionPredictor):
    """Option 3: Compare multiple neighbourhoods over date range"""
    print("\n" + "="*70)
    print("  COMPARE MULTIPLE NEIGHBOURHOODS")
    print("="*70)
    
    # Select neighbourhoods
    neighbourhoods = []
    print("\nSelect neighbourhoods to compare (enter 'done' when finished):")
    
    while True:
        nh = select_neighbourhood(predictor)
        if nh is None:
            if len(neighbourhoods) == 0:
                print("❌ At least one neighbourhood is required")
                continue
            break
        if nh not in neighbourhoods:
            neighbourhoods.append(nh)
            print(f"✅ Added: {nh}")
        else:
            print(f"⚠️ {nh} already in list")
        
        if input("Add another? (y/n): ").lower() != 'y':
            break
    
    if len(neighbourhoods) == 0:
        return
    
    # Get date range
    print("\n📅 Enter date range (YYYY-MM-DD format):")
    start_str = input("Start date: ").strip()
    start_date = validate_date(start_str)
    if start_date is None:
        return
    
    end_str = input("End date: ").strip()
    end_date = validate_date(end_str)
    if end_date is None:
        return
    
    if start_date > end_date:
        print("❌ Start date must be before end date")
        return
    
    # Make predictions for each neighbourhood
    all_results = []
    for nh in neighbourhoods:
        print(f"\n🔮 Predicting for {nh}...")
        results = predictor.predict_week_range(nh, start_date, end_date)
        all_results.append(results)
    
    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    display_results_table(combined, f"Comparison: {', '.join(neighbourhoods)}")
    
    # Create pivot table for comparison
    pivot = combined.pivot_table(
        values='probability',
        index='date',
        columns='neighbourhood',
        aggfunc='first'
    )
    
    print("\n📊 COMPARISON MATRIX (Probability %):")
    print(tabulate(pivot * 100, headers='keys', tablefmt='grid', floatfmt=".1f"))
    
    # Ask to export
    if input("\n💾 Export results to CSV? (y/n): ").lower() == 'y':
        filename = f"comparison_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        combined.to_csv(filepath, index=False)
        print(f"✅ Results saved to: {filepath}")
        
        # Also export pivot
        pivot_path = os.path.join(OUTPUT_DIR, f"pivot_{filename}")
        pivot.to_csv(pivot_path)
        print(f"✅ Comparison matrix saved to: {pivot_path}")

def find_high_risk_dates(predictor: CollisionPredictor):
    """Option 4: Find high-risk dates for a neighbourhood"""
    print("\n" + "="*70)
    print("  FIND HIGH-RISK DATES FOR NEIGHBOURHOOD")
    print("="*70)
    
    # Select neighbourhood
    neighbourhood = select_neighbourhood(predictor)
    if neighbourhood is None:
        return
    
    # Get date range
    print("\n📅 Enter date range to scan (YYYY-MM-DD format):")
    start_str = input("Start date: ").strip()
    start_date = validate_date(start_str)
    if start_date is None:
        return
    
    end_str = input("End date: ").strip()
    end_date = validate_date(end_str)
    if end_date is None:
        return
    
    # Get risk threshold
    threshold_input = input("\n🎯 Risk threshold (0-1, default 0.5): ").strip()
    threshold = float(threshold_input) if threshold_input else 0.5
    
    # Make predictions
    print(f"\n🔮 Scanning for {neighbourhood} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    results = predictor.predict_week_range(neighbourhood, start_date, end_date)
    
    # Filter high-risk dates
    high_risk = results[results['probability'] >= threshold].copy()
    
    if len(high_risk) == 0:
        print(f"\n✅ No dates found with risk ≥ {threshold:.0%}")
    else:
        print(f"\n🚨 Found {len(high_risk)} high-risk dates:")
        display_results_table(high_risk, f"High-risk dates for {neighbourhood} (≥{threshold:.0%})")
        
        # Show risk trend
        if len(results) > 1:
            print("\n📈 RISK TREND:")
            for _, row in results.iterrows():
                bar = "█" * int(row['probability'] * 40)
                print(f"  {row['date'].strftime('%Y-%m-%d')}: {bar} {row['probability']:.1%}")
    
    # Ask to export
    if input("\n💾 Export results to CSV? (y/n): ").lower() == 'y':
        filename = f"high_risk_{neighbourhood}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        high_risk.to_csv(filepath, index=False)
        print(f"✅ Results saved to: {filepath}")

def main():
    """Main application entry point"""
    print_header()
    
    try:
        # Load predictor
        predictor = CollisionPredictor()
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("   Please ensure the model has been trained first.")
        print("   Run the training script: python collision_prediction.py")
        sys.exit(1)
    
    # Interactive menu
    while True:
        print_menu()
        choice = input("👉 Select option (0-5): ").strip()
        
        if choice == "0":
            print("\n👋 Goodbye! Stay safe on the roads!")
            break
        
        elif choice == "1":
            query_neighbourhood_range(predictor)
        
        elif choice == "2":
            query_all_neighbourhoods(predictor)
        
        elif choice == "3":
            compare_neighbourhoods(predictor)
        
        elif choice == "4":
            find_high_risk_dates(predictor)
        
        elif choice == "5":
            print("\n📁 Results are automatically saved to the 'query_results' directory")
            print(f"   Location: {os.path.abspath(OUTPUT_DIR)}")
        
        else:
            print("❌ Invalid option. Please choose 0-5")
        
        input("\n⏎ Press Enter to continue...")

if __name__ == "__main__":
    main()