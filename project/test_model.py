"""
Traffic Collision Prediction Model
===================================
Inspired by: Zhang et al. (2022) "Interpretable machine learning models for crime prediction"

Split strategy (temporal, no data leakage)
-------------------------------------------
  TRAIN  : data before 2024  -> 75% used for training
  VAL    : data before 2024  -> 25% used for validation  (random split within pre-2024)
  TEST   : data from 2024 onwards  (held-out, never seen during training)

Outputs  (all saved to project/model_data/)
-------------------------------------------
  collision_weekly_predictions.csv    per (neighbourhood x week) probability
  next_collision_forecast.csv         next likely collision per neighbourhood
  neighbourhood_risk_ranking.csv      neighbourhoods ranked by avg risk
  model_accuracy_report.csv           val + test metrics side by side
  collision_xgboost_model.pkl         saved model bundle
  confusion_matrix_val.png            validation confusion matrix
  confusion_matrix_test.png           test confusion matrix
  shap_global_bar.png                 SHAP feature importance bar chart
  shap_beeswarm.png                   SHAP beeswarm distribution
"""

import os
import warnings
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

CUTOFF_YEAR          = 2024     # rows < 2024 = train/val | rows >= 2024 = test
VAL_SIZE             = 0.25     # fraction of pre-2024 data held out for validation
PREDICTION_THRESHOLD = 0.50     # binary classification cutoff
FORECAST_THRESHOLD   = 0.40     # min probability to flag a future week as "likely"
FORECAST_WEEKS_AHEAD = 12       # weeks ahead to forecast per neighbourhood

# Output dir: project/model_data/ relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, "model_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "neighbourhood_enc",
    "month",
    "quarter",
    "week_of_year",
    "year",
    "season",
    "is_holiday_season",
    "hist_rate_4w",       # rolling 4-week avg collision rate (lag, no leakage)
    "hist_rate_8w",       # rolling 8-week avg collision rate (lag, no leakage)
    "weeks_since_last",   # weeks since last collision in this neighbourhood
]

FEATURE_LABELS = {
    "neighbourhood_enc":  "Neighbourhood",
    "month":              "Month",
    "quarter":            "Quarter",
    "week_of_year":       "Week of Year",
    "year":               "Year",
    "season":             "Season",
    "is_holiday_season":  "Holiday Season",
    "hist_rate_4w":       "4-Week Historical Rate",
    "hist_rate_8w":       "8-Week Historical Rate",
    "weeks_since_last":   "Weeks Since Last Collision",
}


# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    load_dotenv()
    db_config = {
        "dbname":   os.getenv("DB_NAME"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host":     os.getenv("DB_HOST"),
        "port":     os.getenv("DB_PORT"),
    }

    print("[1/7] Connecting to database...")
    conn  = psycopg2.connect(**db_config)
    query = """
        SELECT occ_date, neighbourhood_158
        FROM traffic_collisions_data
        WHERE neighbourhood_158 IS NOT NULL
          AND neighbourhood_158 <> 'NSA'
          AND occ_date IS NOT NULL
        ORDER BY occ_date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["occ_date"] = pd.to_datetime(df["occ_date"])
    print(f"      Loaded {len(df):,} rows  |  "
          f"Date range: {df['occ_date'].min().date()} to {df['occ_date'].max().date()}")

    pre  = (df["occ_date"].dt.year < CUTOFF_YEAR).sum()
    post = (df["occ_date"].dt.year >= CUTOFF_YEAR).sum()
    print(f"      Pre-{CUTOFF_YEAR} (train/val): {pre:,} rows  |  "
          f">= {CUTOFF_YEAR} (test): {post:,} rows")
    return df


# ═══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def _season(month: int) -> int:
    if month in [12, 1, 2]: return 0   # Winter
    if month in [3, 4, 5]:  return 1   # Spring
    if month in [6, 7, 8]:  return 2   # Summer
    return 3                            # Fall

def _holiday_season(month: int) -> int:
    return int(month in [11, 12, 1])   # Nov-Jan higher risk

def engineer_features(df: pd.DataFrame):
    print("[2/7] Engineering features...")

    df = df.copy()
    df["week_start"] = df["occ_date"].dt.to_period("W").apply(lambda p: p.start_time)

    # ── Aggregate to (neighbourhood x week) ───────────────────────────
    obs = (
        df.groupby(["neighbourhood_158", "week_start"])
        .size()
        .reset_index(name="collision_count")
    )
    obs["collision"] = (obs["collision_count"] >= 1).astype(int)

    # Full cross-join so every neighbourhood has a row for every week
    all_nh  = df["neighbourhood_158"].unique()
    all_wks = obs["week_start"].unique()

    full_grid = (
        pd.MultiIndex.from_product(
            [all_nh, all_wks], names=["neighbourhood_158", "week_start"]
        ).to_frame(index=False)
    )
    full_grid = full_grid.merge(
        obs[["neighbourhood_158", "week_start", "collision"]],
        on=["neighbourhood_158", "week_start"], how="left"
    )
    full_grid["collision"] = full_grid["collision"].fillna(0).astype(int)
    full_grid = (
        full_grid
        .sort_values(["neighbourhood_158", "week_start"])
        .reset_index(drop=True)
    )

    # ── Temporal features ──────────────────────────────────────────────
    ws = full_grid["week_start"]
    full_grid["month"]             = ws.dt.month
    full_grid["quarter"]           = ws.dt.quarter
    full_grid["week_of_year"]      = ws.dt.isocalendar().week.astype(int)
    full_grid["year"]              = ws.dt.year
    full_grid["season"]            = full_grid["month"].apply(_season)
    full_grid["is_holiday_season"] = full_grid["month"].apply(_holiday_season)

    # ── Neighbourhood encoding (fit on full data so test set has no unknowns) ──
    le = LabelEncoder()
    full_grid["neighbourhood_enc"] = le.fit_transform(full_grid["neighbourhood_158"])

    # ── Lag features (shift(1) ensures no leakage into the target week) ──
    grp = full_grid.groupby("neighbourhood_158")["collision"]

    full_grid["hist_rate_4w"] = (
        grp.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)
    )
    full_grid["hist_rate_8w"] = (
        grp.transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean()).fillna(0)
    )

    def _weeks_since_last(series: pd.Series) -> pd.Series:
        result, last = [], np.nan
        for i in range(len(series)):
            result.append(np.nan if i == 0 else last)
            if series.iloc[i] == 1:
                last = 0
            elif not np.isnan(last):
                last += 1
        return pd.Series(result, index=series.index)

    full_grid["weeks_since_last"] = (
        full_grid.groupby("neighbourhood_158")["collision"]
        .transform(_weeks_since_last)
        .fillna(0)
    )

    # ── Temporal split flag ────────────────────────────────────────────
    full_grid["is_test"] = (full_grid["week_start"].dt.year >= CUTOFF_YEAR).astype(int)

    pos_rate = full_grid["collision"].mean()
    print(f"      Grid: {len(full_grid):,} cells  |  "
          f"Neighbourhoods: {full_grid['neighbourhood_158'].nunique()}  |  "
          f"Weeks: {full_grid['week_start'].nunique()}  |  "
          f"Positive rate: {pos_rate:.2%}")
    print(f"      Train/val cells: {(full_grid['is_test']==0).sum():,}  |  "
          f"Test cells (>={CUTOFF_YEAR}): {(full_grid['is_test']==1).sum():,}")

    return full_grid, le


# ═══════════════════════════════════════════════════════════════
# 3. TRAIN  /  VALIDATE  /  TEST
# ═══════════════════════════════════════════════════════════════

def _print_metrics(label: str, y_true, y_pred, y_prob) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    roc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n      [{label}]")
    print(f"        Accuracy  : {acc:.4f}")
    print(f"        ROC-AUC   : {roc:.4f}")
    print(f"        F1        : {f1:.4f}")
    print(f"        Precision : {prec:.4f}")
    print(f"        Recall    : {rec:.4f}")
    print(f"\n      Classification Report ({label}):")
    print(classification_report(y_true, y_pred,
                                target_names=["No Collision", "Collision"]))
    return {
        "split":     label,
        "accuracy":  round(acc,  4),
        "roc_auc":   round(roc,  4),
        "f1":        round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
    }

def _save_confusion_matrix(y_true, y_pred, title: str, filename: str):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Collision", "Collision"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()

def train_model(full_grid: pd.DataFrame):
    print("[3/7] Training XGBoost — temporal split...")
    print(f"      Train/Val : week_start < {CUTOFF_YEAR}")
    print(f"      Test      : week_start >= {CUTOFF_YEAR}\n")

    # ── Temporal split ─────────────────────────────────────────────────
    pre_2024  = full_grid[full_grid["is_test"] == 0]
    test_data = full_grid[full_grid["is_test"] == 1]

    X_pre = pre_2024[FEATURE_COLS]
    y_pre = pre_2024["collision"]

    X_test_final = test_data[FEATURE_COLS]
    y_test_final = test_data["collision"]

    # ── Random val split within pre-2024 ──────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_pre, y_pre,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=y_pre,
    )

    print(f"      Train rows : {len(X_train):,}")
    print(f"      Val rows   : {len(X_val):,}")
    print(f"      Test rows  : {len(X_test_final):,}")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators     = 400,
        max_depth        = 6,
        learning_rate    = 0.04,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        gamma            = 0.1,
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "auc",
        random_state     = 42,
        verbosity        = 0,
        early_stopping_rounds = 30,
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose  = False,
    )

    # ── Validation metrics ─────────────────────────────────────────────
    val_pred  = model.predict(X_val)
    val_prob  = model.predict_proba(X_val)[:, 1]
    val_metrics = _print_metrics("VALIDATION (pre-2024 hold-out)", y_val, val_pred, val_prob)
    _save_confusion_matrix(
        y_val, val_pred,
        title    = f"Confusion Matrix - Validation (pre-{CUTOFF_YEAR})",
        filename = "confusion_matrix_val.png",
    )

    # ── Test metrics (2024+) ───────────────────────────────────────────
    test_pred  = model.predict(X_test_final)
    test_prob  = model.predict_proba(X_test_final)[:, 1]
    test_metrics = _print_metrics(f"TEST (>= {CUTOFF_YEAR})", y_test_final, test_pred, test_prob)
    _save_confusion_matrix(
        y_test_final, test_pred,
        title    = f"Confusion Matrix - Test (>= {CUTOFF_YEAR})",
        filename = "confusion_matrix_test.png",
    )

    # ── Save accuracy report CSV ───────────────────────────────────────
    report_df = pd.DataFrame([val_metrics, test_metrics])
    report_path = os.path.join(OUTPUT_DIR, "model_accuracy_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\n      Accuracy report saved -> model_accuracy_report.csv")
    print(f"\n  {'='*55}")
    print(f"  SUMMARY")
    print(f"  {'='*55}")
    print(f"  Validation  accuracy : {val_metrics['accuracy']:.4f}  |  ROC-AUC : {val_metrics['roc_auc']:.4f}")
    print(f"  Test        accuracy : {test_metrics['accuracy']:.4f}  |  ROC-AUC : {test_metrics['roc_auc']:.4f}")
    print(f"  {'='*55}\n")

    return model, X_train, X_val, X_test_final, val_metrics, test_metrics


# ═══════════════════════════════════════════════════════════════
# 4. SHAP INTERPRETABILITY  (run on val set — same distribution as train)
# ═══════════════════════════════════════════════════════════════

def run_shap(model, X_train, X_val):
    print("[4/7] Running SHAP analysis...")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    X_plot      = X_val.rename(columns=FEATURE_LABELS)

    # Bar chart
    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X_plot, plot_type="bar", show=False, max_display=10)
    plt.title("Global Feature Importance - Mean |SHAP Value|", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_global_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_plot, show=False, max_display=10)
    plt.title("SHAP Value Distribution per Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("      SHAP plots saved.")
    return explainer, shap_values


# ═══════════════════════════════════════════════════════════════
# 5. WEEKLY PROBABILITY PREDICTIONS  (full grid)
# ═══════════════════════════════════════════════════════════════

def _confidence_band(p: float) -> str:
    if p < 0.25: return "Low"
    if p < 0.50: return "Medium"
    if p < 0.75: return "High"
    return "Very High"

def generate_weekly_predictions(model, full_grid: pd.DataFrame) -> pd.DataFrame:
    print("[5/7] Generating weekly probability predictions...")

    probs = model.predict_proba(full_grid[FEATURE_COLS])[:, 1]

    out = full_grid[["neighbourhood_158", "week_start", "collision", "is_test"]].copy()
    out.rename(columns={
        "week_start": "week",
        "collision":  "actual_collision",
        "is_test":    "is_test_period",
    }, inplace=True)
    out["collision_probability"] = probs.round(4)
    out["predicted_collision"]   = (probs >= PREDICTION_THRESHOLD).astype(int)
    out["confidence_band"]       = out["collision_probability"].apply(_confidence_band)
    out = out.sort_values(["neighbourhood_158", "week"]).reset_index(drop=True)

    out.to_csv(os.path.join(OUTPUT_DIR, "collision_weekly_predictions.csv"), index=False)
    print(f"      Saved collision_weekly_predictions.csv  ({len(out):,} rows)")
    return out


# ═══════════════════════════════════════════════════════════════
# 6. NEXT LIKELY COLLISION FORECAST
# ═══════════════════════════════════════════════════════════════

def forecast_next_collision(model, full_grid: pd.DataFrame) -> pd.DataFrame:
    print(f"[6/7] Forecasting next likely collision ({FORECAST_WEEKS_AHEAD} weeks ahead)...")

    last_week      = full_grid["week_start"].max()
    all_nh         = full_grid["neighbourhood_158"].unique()
    future_records = []

    for nh in all_nh:
        nh_hist = (
            full_grid[full_grid["neighbourhood_158"] == nh]
            .sort_values("week_start")
        )
        last_4w      = nh_hist["hist_rate_4w"].iloc[-1]
        last_8w      = nh_hist["hist_rate_8w"].iloc[-1]
        last_wsl     = nh_hist["weeks_since_last"].iloc[-1]
        nh_enc       = nh_hist["neighbourhood_enc"].iloc[-1]

        for w in range(1, FORECAST_WEEKS_AHEAD + 1):
            fw    = last_week + pd.Timedelta(weeks=w)
            month = fw.month
            iso   = fw.isocalendar()
            future_records.append({
                "neighbourhood_158": nh,
                "week_start":        fw,
                "neighbourhood_enc": nh_enc,
                "month":             month,
                "quarter":           (month - 1) // 3 + 1,
                "week_of_year":      iso[1],
                "year":              iso[0],
                "season":            _season(month),
                "is_holiday_season": _holiday_season(month),
                "hist_rate_4w":      last_4w,
                "hist_rate_8w":      last_8w,
                "weeks_since_last":  last_wsl + w,
            })

    future_df              = pd.DataFrame(future_records)
    future_probs           = model.predict_proba(future_df[FEATURE_COLS])[:, 1]
    future_df["prob"]      = future_probs.round(4)
    future_df["conf_band"] = future_df["prob"].apply(_confidence_band)

    results = []
    today   = pd.Timestamp(datetime.now().date())

    for nh in all_nh:
        nh_fut = future_df[future_df["neighbourhood_158"] == nh].sort_values("week_start")
        above  = nh_fut[nh_fut["prob"] >= FORECAST_THRESHOLD]

        if len(above) == 0:
            best = nh_fut.loc[nh_fut["prob"].idxmax()]
            note = f"Best guess (no week >= {FORECAST_THRESHOLD:.0%})"
        else:
            best = above.iloc[0]
            note = "Predicted"

        results.append({
            "neighbourhood":         nh,
            "next_predicted_week":   best["week_start"].date(),
            "collision_probability": best["prob"],
            "confidence_band":       best["conf_band"],
            "weeks_from_now":        max(int((best["week_start"] - today).days // 7), 1),
            "note":                  note,
        })

    forecast_df = (
        pd.DataFrame(results)
        .sort_values("collision_probability", ascending=False)
        .reset_index(drop=True)
    )

    forecast_df.to_csv(os.path.join(OUTPUT_DIR, "next_collision_forecast.csv"), index=False)
    print(f"      Saved next_collision_forecast.csv  ({len(forecast_df)} neighbourhoods)")

    print("\n  -- Highest-Risk Next Predicted Collisions (Top 10) ---------")
    print(forecast_df[
        ["neighbourhood", "next_predicted_week",
         "collision_probability", "confidence_band", "weeks_from_now"]
    ].head(10).to_string(index=False))
    print()

    return forecast_df


# ═══════════════════════════════════════════════════════════════
# 7. RISK RANKING  +  SAVE MODEL
# ═══════════════════════════════════════════════════════════════

def generate_risk_ranking(weekly_preds: pd.DataFrame) -> pd.DataFrame:
    ranking = (
        weekly_preds
        .groupby("neighbourhood_158")["collision_probability"]
        .agg(avg_weekly_risk="mean", max_weekly_risk="max",
             predicted_collision_weeks="sum")
        .reset_index()
        .sort_values("avg_weekly_risk", ascending=False)
        .reset_index(drop=True)
    )
    ranking["risk_rank"]       = ranking.index + 1
    ranking["avg_weekly_risk"] = ranking["avg_weekly_risk"].round(4)
    ranking["max_weekly_risk"] = ranking["max_weekly_risk"].round(4)
    ranking["confidence_band"] = ranking["avg_weekly_risk"].apply(_confidence_band)

    ranking.to_csv(os.path.join(OUTPUT_DIR, "neighbourhood_risk_ranking.csv"), index=False)
    print("      Saved neighbourhood_risk_ranking.csv")

    print("\n  -- Top 10 Highest Risk Neighbourhoods ----------------------")
    print(ranking[
        ["risk_rank", "neighbourhood_158", "avg_weekly_risk", "confidence_band"]
    ].head(10).to_string(index=False))
    print()
    return ranking

def save_model(model, le: LabelEncoder):
    bundle = {
        "model":         model,
        "label_encoder": le,
        "feature_cols":  FEATURE_COLS,
        "cutoff_year":   CUTOFF_YEAR,
        "granularity":   "weekly",
        "trained_at":    datetime.now().isoformat(),
    }
    path = os.path.join(OUTPUT_DIR, "collision_xgboost_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print("      Saved collision_xgboost_model.pkl")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Traffic Collision Prediction -- XGBoost + SHAP (Weekly)")
    print(f"  Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print("=" * 65 + "\n")

    df                                              = load_data()
    full_grid, le                                   = engineer_features(df)
    model, X_train, X_val, X_test, val_m, test_m   = train_model(full_grid)
    explainer, shap_values                          = run_shap(model, X_train, X_val)
    weekly_preds                                    = generate_weekly_predictions(model, full_grid)
    forecast_df                                     = forecast_next_collision(model, full_grid)

    print("[7/7] Building neighbourhood risk ranking & saving model...")
    ranking = generate_risk_ranking(weekly_preds)
    save_model(model, le)

    print("\n" + "=" * 65)
    print("  FINAL ACCURACY SUMMARY")
    print(f"  Validation accuracy : {val_m['accuracy']:.4f}  |  ROC-AUC : {val_m['roc_auc']:.4f}  (pre-{CUTOFF_YEAR} hold-out)")
    print(f"  Test       accuracy : {test_m['accuracy']:.4f}  |  ROC-AUC : {test_m['roc_auc']:.4f}  (>= {CUTOFF_YEAR})")
    print("=" * 65)
    print(f"\n  Output files in: {OUTPUT_DIR}")
    print("     collision_weekly_predictions.csv")
    print("     next_collision_forecast.csv")
    print("     neighbourhood_risk_ranking.csv")
    print("     model_accuracy_report.csv")
    print("     collision_xgboost_model.pkl")
    print("     confusion_matrix_val.png")
    print("     confusion_matrix_test.png")
    print("     shap_global_bar.png")
    print("     shap_beeswarm.png")
    print("=" * 65)


if __name__ == "__main__":
    main()