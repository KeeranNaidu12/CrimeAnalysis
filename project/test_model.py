"""
Traffic Collision Prediction Model
====================================
5-day XGBoost classifier with:
  - Optuna hyperparameter tuning
  - Time-series cross-validation
  - Probability calibration (isotonic, prefit)
  - Optimal threshold selection (F1)
  - SHAP analysis suite
  - Structured logging

Usage:
    python collision_model_5day.py
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import shap
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CUTOFF_YEAR           = 2024     # rows with year >= this form the test set
VAL_SIZE              = 0.25     # fraction of pre-cutoff data held out for val
DEFAULT_THRESHOLD     = 0.50     # used only if threshold optimisation is off
COLLISION_COUNT_THRESHOLD = 0.65  # Minimum confidence to count as collision (65%)
FORECAST_THRESHOLD    = 0.40     # minimum prob to call a forecast "Predicted"
FORECAST_PERIODS_AHEAD = 17      # 12 weeks = 17 periods (5-day intervals, 85 days)

USE_OPTUNA            = True     # set False if optuna is not installed
N_OPTUNA_TRIALS       = 100
USE_TIMESERIES_CV     = True
CALIBRATE_MODEL       = True
FIND_OPTIMAL_THRESHOLD = True

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, "model_data_5day")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants for 5-day periods
DAYS_PER_PERIOD = 5
PERIODS_PER_YEAR = 365 / DAYS_PER_PERIOD  # = 73 periods per year

# ── Canadian statutory holidays used for "days_to_holiday" feature ──────────
_CA_HOLIDAYS_MD = [          # (month, day) tuples — year is added at runtime
    (1,  1),   # New Year's Day
    (2, 17),   # Family Day (3rd Monday Feb, approximated)
    (4, 18),   # Good Friday (approximated)
    (5, 19),   # Victoria Day (3rd Monday May, approximated)
    (7,  1),   # Canada Day
    (8,  4),   # BC Day (1st Monday Aug, approximated)
    (9,  1),   # Labour Day (1st Monday Sep, approximated)
    (10, 13),  # Thanksgiving (2nd Monday Oct, approximated)
    (11, 11),  # Remembrance Day
    (12, 25),  # Christmas Day
    (12, 26),  # Boxing Day
]

# ─────────────────────────────────────────────────────────────────────────────
# Feature schema
# ─────────────────────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "neighbourhood_enc",
    "month", "quarter", "period_of_year", "year",
    "season", "is_holiday_season",
    "hist_rate_3w", "hist_rate_6w",
    "periods_since_last",
]

ADVANCED_FEATURES = [
    "month_sin", "month_cos",
    "period_sin", "period_cos",
    "ewma_3w",
    "rate_change_3w",
    "collision_trend",
    "high_risk_cluster",
    "days_to_holiday",
    "recent_collision_density",
    "seasonal_pattern",
]

FEATURE_COLS = BASE_FEATURES + ADVANCED_FEATURES

FEATURE_LABELS: dict[str, str] = {
    "neighbourhood_enc":        "Neighbourhood",
    "month":                    "Month",
    "month_sin":                "Month sin",
    "month_cos":                "Month cos",
    "quarter":                  "Quarter",
    "period_of_year":           "Period of Year (5-day)",
    "period_sin":               "Period sin",
    "period_cos":               "Period cos",
    "year":                     "Year",
    "season":                   "Season",
    "is_holiday_season":        "Holiday Season",
    "hist_rate_3w":             "3-Week Historical Rate",
    "hist_rate_6w":             "6-Week Historical Rate",
    "ewma_3w":                  "EWMA (3-week)",
    "periods_since_last":       "Periods Since Last Collision",
    "rate_change_3w":           "Rate Change (3w)",
    "collision_trend":          "Collision Trend",
    "high_risk_cluster":        "High Risk Cluster",
    "days_to_holiday":          "Days to Holiday",
    "recent_collision_density": "Recent Collision Density",
    "seasonal_pattern":         "Seasonal Pattern",
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "Temporal": [
        "month", "month_sin", "month_cos",
        "quarter", "period_of_year", "period_sin", "period_cos",
        "year", "season", "is_holiday_season", "days_to_holiday",
    ],
    "Historical rates": [
        "hist_rate_3w", "hist_rate_6w", "ewma_3w",
        "rate_change_3w", "collision_trend",
        "periods_since_last", "recent_collision_density",
        "seasonal_pattern",
    ],
    "Spatial / risk": [
        "neighbourhood_enc", "high_risk_cluster",
    ],
}

FEATURE_GROUP_COLORS: dict[str, str] = {
    "Temporal":          "#7F77DD",
    "Historical rates":  "#1D9E75",
    "Spatial / risk":    "#D85A30",
}


def _feature_color(feature_name: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if feature_name in members:
            return FEATURE_GROUP_COLORS[group]
    return "#888780"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    load_dotenv()
    db_config = {
        "dbname":   os.getenv("DB_NAME"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host":     os.getenv("DB_HOST"),
        "port":     os.getenv("DB_PORT"),
    }

    log.info("Connecting to database …")
    conn = psycopg2.connect(**db_config)
    query = """
        SELECT occ_date, neighbourhood_158
        FROM   traffic_collisions_data
        WHERE  neighbourhood_158 IS NOT NULL
          AND  neighbourhood_158 <> 'NSA'
          AND  occ_date          IS NOT NULL
        ORDER  BY occ_date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["occ_date"] = pd.to_datetime(df["occ_date"])
    log.info(
        "Loaded %s rows | %s → %s",
        f"{len(df):,}",
        df["occ_date"].min().date(),
        df["occ_date"].max().date(),
    )

    pre  = (df["occ_date"].dt.year <  CUTOFF_YEAR).sum()
    post = (df["occ_date"].dt.year >= CUTOFF_YEAR).sum()
    log.info("Pre-%d (train/val): %s rows | >= %d (test): %s rows",
             CUTOFF_YEAR, f"{pre:,}", CUTOFF_YEAR, f"{post:,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _season(month: int) -> int:
    if month in (12, 1, 2): return 0   # Winter
    if month in (3,  4, 5): return 1   # Spring
    if month in (6,  7, 8): return 2   # Summer
    return 3                            # Fall


def _is_holiday_season(month: int) -> int:
    return int(month in (11, 12, 1))


def _build_holiday_index(years: np.ndarray) -> pd.DatetimeIndex:
    """Return a sorted DatetimeIndex of Canadian statutory holidays."""
    dates: list[pd.Timestamp] = []
    for year in years:
        for month, day in _CA_HOLIDAYS_MD:
            try:
                dates.append(pd.Timestamp(year=int(year), month=month, day=day))
            except ValueError:
                pass   # skip invalid dates (e.g. Feb 30)
    return pd.DatetimeIndex(sorted(dates))


def _days_to_nearest_holiday(
    dates: pd.Series,
    holiday_index: pd.DatetimeIndex,
) -> pd.Series:
    """Vectorised: minimum absolute days from each date to any holiday."""
    hol_ns = holiday_index.astype(np.int64).values
    dt_ns  = dates.astype(np.int64).values
    diff_days = np.abs(dt_ns[:, None] - hol_ns[None, :]) // (86_400 * 1_000_000_000)
    return pd.Series(diff_days.min(axis=1), index=dates.index).clip(0, 30)


def _calculate_trend(series: pd.Series) -> float:
    """Linear regression slope over the last ≤14 observations (~10 weeks)."""
    vals = series.dropna().values
    n    = min(len(vals), 14)  # 14 periods = 10 weeks (14 * 5 days = 70 days)
    if n < 3:
        return 0.0
    x = np.arange(n)
    y = vals[-n:]
    return float(np.polyfit(x, y, 1)[0])


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    log.info("Engineering features for 5-day periods …")
    
    df = df.copy()
    
    # Create 5-day period bins
    min_date = df["occ_date"].min()
    df["period_start"] = min_date + pd.to_timedelta(
        ((df["occ_date"] - min_date).dt.days // DAYS_PER_PERIOD) * DAYS_PER_PERIOD, 
        unit="D"
    )
    
    # Also add a floor to start of day for consistent grouping
    df["period_start"] = pd.to_datetime(df["period_start"].dt.date)
    
    # ── 5-day aggregate ──────────────────────────────────────────────────
    obs = (
        df.groupby(["neighbourhood_158", "period_start"])
        .size()
        .reset_index(name="collision_count")
    )
    obs["collision"] = (obs["collision_count"] >= 1).astype(int)
    
    log.info(f"Created {len(obs)} observations across {obs['period_start'].nunique()} periods")
    
    # ── Full neighbourhood × period grid ───────────────────────────────────
    all_nh     = df["neighbourhood_158"].unique()
    all_periods = obs["period_start"].unique()
    
    grid = (
        pd.MultiIndex
        .from_product([all_nh, all_periods], names=["neighbourhood_158", "period_start"])
        .to_frame(index=False)
        .merge(obs[["neighbourhood_158", "period_start", "collision"]],
               on=["neighbourhood_158", "period_start"], how="left")
    )
    grid["collision"] = grid["collision"].fillna(0).astype(int)
    grid = grid.sort_values(["neighbourhood_158", "period_start"]).reset_index(drop=True)
    
    # ── Temporal features (updated for 5-day periods) ─────────────────────
    ps = grid["period_start"]
    grid["month"]        = ps.dt.month
    grid["quarter"]      = ps.dt.quarter
    grid["year"]         = ps.dt.year
    grid["season"]       = grid["month"].map(_season)
    grid["is_holiday_season"] = grid["month"].map(_is_holiday_season)
    
    # Calculate period of year (1-based, 73 periods per year)
    year_start = pd.to_datetime(grid["year"].astype(str) + "-01-01")
    days_since_year_start = (ps - year_start).dt.days
    grid["period_of_year"] = (days_since_year_start // DAYS_PER_PERIOD) + 1
    
    # Cyclical encoding for periods within year
    grid["period_sin"] = np.sin(2 * np.pi * grid["period_of_year"] / PERIODS_PER_YEAR)
    grid["period_cos"] = np.cos(2 * np.pi * grid["period_of_year"] / PERIODS_PER_YEAR)
    
    # Keep month sin/cos for seasonal patterns
    grid["month_sin"] = np.sin(2 * np.pi * grid["month"] / 12)
    grid["month_cos"] = np.cos(2 * np.pi * grid["month"] / 12)
    
    # ── Neighbourhood encoding ────────────────────────────────────────────
    le = LabelEncoder()
    grid["neighbourhood_enc"] = le.fit_transform(grid["neighbourhood_158"])
    
    # ── Rolling features (adjusted for 5-day periods) ─────────────────────
    # 3 weeks = 4.2 periods ≈ 4 periods (3 weeks * 7 days / 5 days per period = 4.2)
    # Using 4 periods to capture ~3 weeks
    grp = grid.groupby("neighbourhood_158")["collision"]
    
    # 3-week historical rate (4 periods)
    grid["hist_rate_3w"] = (
        grp.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)
    )
    
    # 6-week historical rate (8 periods)
    grid["hist_rate_6w"] = (
        grp.transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean()).fillna(0)
    )
    
    # EWMA with 4 periods span (~3 weeks)
    grid["ewma_3w"] = (
        grp.transform(lambda x: x.shift(1).ewm(span=4, adjust=False).mean()).fillna(0)
    )
    
    # Rate change: 3-week vs 6-week
    grid["rate_change_3w"] = grid["hist_rate_3w"] - grid["hist_rate_6w"]
    
    # Recent density: last 2 periods (~10 days)
    grid["recent_collision_density"] = (
        grp.transform(lambda x: x.shift(1).rolling(2, min_periods=1).sum()).fillna(0)
    )
    
    # Seasonal pattern: shift by 73 periods (≈1 year)
    grid["seasonal_pattern"] = grp.transform(lambda x: x.shift(73)).fillna(0)
    
    # ── Periods since last collision ──────────────────────────────────────
    def _periods_since_last_vec(s: pd.Series) -> pd.Series:
        s = s.reset_index(drop=True)
        run_id = (s == 1).cumsum()
        counter = s.groupby(run_id).cumcount()
        return counter.shift(1).fillna(0)
    
    grid["periods_since_last"] = (
        grid.groupby("neighbourhood_158")["collision"]
        .transform(_periods_since_last_vec)
        .fillna(0)
    )
    
    # ── Collision trend (slope of last 14 periods ≈ 10 weeks) ────────────
    grid["collision_trend"] = (
        grid.groupby("neighbourhood_158")["collision"]
        .transform(
            lambda x: x.shift(1)
                       .rolling(14, min_periods=4)
                       .apply(_calculate_trend, raw=False)
        )
        .fillna(0)
    )
    
    # ── Days to nearest Canadian holiday ─────────────────────────────────
    holiday_idx = _build_holiday_index(grid["year"].unique())
    grid["days_to_holiday"] = _days_to_nearest_holiday(grid["period_start"], holiday_idx)
    
    # ── High-risk neighbourhood cluster (top 30% by avg 3-week rate) ─────
    nh_avg = grid.groupby("neighbourhood_158")["hist_rate_3w"].mean()
    threshold = nh_avg.quantile(0.70)
    high_risk_set = set(nh_avg[nh_avg > threshold].index)
    grid["high_risk_cluster"] = grid["neighbourhood_158"].isin(high_risk_set).astype(int)
    
    # ── Test flag ─────────────────────────────────────────────────────────
    grid["is_test"] = (grid["period_start"].dt.year >= CUTOFF_YEAR).astype(int)
    
    log.info(
        "Grid: %s cells | %d neighbourhoods | %d periods | positive rate: %.2f%%",
        f"{len(grid):,}",
        grid["neighbourhood_158"].nunique(),
        grid["period_start"].nunique(),
        100 * grid["collision"].mean(),
    )
    return grid, le


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameter optimisation (Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_objective(
    trial: "optuna.Trial",
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
) -> float:
    params = dict(
        n_estimators       = trial.suggest_int  ("n_estimators",    300, 1500, step=50),
        max_depth          = trial.suggest_int  ("max_depth",         3,   12),
        learning_rate      = trial.suggest_float("learning_rate",  5e-3,  0.2, log=True),
        subsample          = trial.suggest_float("subsample",        0.6,  1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.6,  1.0),
        min_child_weight   = trial.suggest_int  ("min_child_weight",   1,   10),
        gamma              = trial.suggest_float("gamma",             0.0,  0.5),
        reg_alpha          = trial.suggest_float("reg_alpha",        1e-8, 10.0, log=True),
        reg_lambda         = trial.suggest_float("reg_lambda",       1e-8, 10.0, log=True),
        scale_pos_weight   = trial.suggest_float("scale_pos_weight",  1.0, 20.0),
        eval_metric        = "auc",
        early_stopping_rounds = 30,
        random_state       = 42,
        verbosity          = 0,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])


def optimize_hyperparameters(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_va: pd.DataFrame, y_va: pd.Series,
) -> dict:
    if not OPTUNA_AVAILABLE:
        log.warning("Optuna not installed — skipping hyperparameter tuning.")
        return {}
    log.info("Optuna: running %d trials …", N_OPTUNA_TRIALS)
    study = optuna.create_study(direction="maximize", study_name="xgb_collision")
    study.optimize(
        lambda t: _xgb_objective(t, X_tr, y_tr, X_va, y_va),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    log.info("Optuna best AUC: %.4f | params: %s", study.best_value, study.best_params)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training (time-series CV or single split)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PARAMS: dict = dict(
    n_estimators     = 800,
    max_depth        = 7,
    learning_rate    = 0.03,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma            = 0.1,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    eval_metric      = "auc",
    random_state     = 42,
    verbosity        = 0,
)


def _scale_pos_weight(y: pd.Series) -> float:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg) / max(n_pos, 1)


def train_model(
    grid: pd.DataFrame,
    best_params: Optional[dict] = None,
) -> tuple[xgb.XGBClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train the final XGBoost model.

    Returns
    -------
    model, X_pre, y_pre, X_test, y_test
    """
    pre  = grid[grid["is_test"] == 0]
    test = grid[grid["is_test"] == 1]

    X_pre,  y_pre  = pre[FEATURE_COLS],  pre["collision"]
    X_test, y_test = test[FEATURE_COLS], test["collision"]

    params = {**_DEFAULT_PARAMS, **(best_params or {})}
    # Always remove early_stopping_rounds from final training params
    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    final_params["scale_pos_weight"] = _scale_pos_weight(y_pre)

    if USE_TIMESERIES_CV:
        log.info("Training with time-series cross-validation (5 folds, gap=7 periods) …")
        tscv = TimeSeriesSplit(n_splits=5, gap=7)  # gap of ~5 weeks
        cv_aucs: list[float] = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_pre), 1):
            Xf_tr, yf_tr = X_pre.iloc[tr_idx], y_pre.iloc[tr_idx]
            Xf_va, yf_va = X_pre.iloc[va_idx],  y_pre.iloc[va_idx]

            fold_params = {**params, "scale_pos_weight": _scale_pos_weight(yf_tr)}
            m = xgb.XGBClassifier(**fold_params)
            m.fit(Xf_tr, yf_tr, eval_set=[(Xf_va, yf_va)], verbose=False)

            auc = roc_auc_score(yf_va, m.predict_proba(Xf_va)[:, 1])
            cv_aucs.append(auc)
            log.info("  Fold %d AUC = %.4f", fold, auc)

        log.info("CV AUC: %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))
    else:
        log.info("Training with single random split …")

    # Final model trained on all pre-cutoff data
    log.info("Training final model on all pre-%d data …", CUTOFF_YEAR)
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_pre, y_pre)

    test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
    log.info("Test AUC (raw model): %.4f", test_auc)

    return final_model, X_pre, y_pre, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 5. Probability calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_model(
    model: xgb.XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV | xgb.XGBClassifier:
    """
    Calibrate using isotonic regression (prefit).
    Falls back gracefully to the original model if calibration fails.
    """
    if not CALIBRATE_MODEL:
        return model

    log.info("Calibrating model probabilities (isotonic, prefit) …")
    try:
        cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        cal.fit(X_val, y_val)

        brier_before = brier_score_loss(y_val, model.predict_proba(X_val)[:, 1])
        brier_after  = brier_score_loss(y_val, cal.predict_proba(X_val)[:, 1])
        log.info(
            "Brier score — before: %.4f  after: %.4f  (improvement: %+.4f)",
            brier_before, brier_after, brier_before - brier_after,
        )
        return cal
    except Exception as exc:
        log.warning("Calibration failed (%s) — continuing with uncalibrated model.", exc)
        return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. Optimal threshold (now used for model prediction, not counting)
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Return the probability threshold that maximises F1 on the validation set."""
    if not FIND_OPTIMAL_THRESHOLD:
        return DEFAULT_THRESHOLD

    y_prob = model.predict_proba(X_val)[:, 1]
    precs, recs, thresholds = precision_recall_curve(y_val, y_prob)

    # Avoid divide-by-zero; arrays are (n+1,) so trim the last element
    f1s = np.where(
        (precs[:-1] + recs[:-1]) == 0,
        0.0,
        2 * precs[:-1] * recs[:-1] / (precs[:-1] + recs[:-1]),
    )
    best_idx       = int(np.argmax(f1s))
    optimal_thresh = float(thresholds[best_idx])
    log.info(
        "Optimal threshold: %.3f  (F1 = %.3f, precision = %.3f, recall = %.3f)",
        optimal_thresh, f1s[best_idx], precs[best_idx], recs[best_idx],
    )

    # ── Save precision-recall-threshold curve ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precs[:-1], label="Precision", linewidth=2, color="#7F77DD")
    ax.plot(thresholds, recs[:-1],  label="Recall",    linewidth=2, color="#1D9E75")
    ax.plot(thresholds, f1s,        label="F1",        linewidth=2, color="#D85A30",
            linestyle="--")
    ax.axvline(optimal_thresh, color="black", linestyle=":", alpha=0.6,
               label=f"Optimal ({optimal_thresh:.3f})")
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision / Recall / F1 vs. Decision Threshold", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "optimal_threshold.png"), dpi=150)
    plt.close()

    return optimal_thresh


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation helpers (modified to separate prediction from counting)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    label: str,
    y_true: pd.Series,
    y_prob: np.ndarray,
    prediction_threshold: float,
    counting_threshold: float,
) -> dict:
    """
    Evaluate model with two thresholds:
    - prediction_threshold: used for binary prediction (model says collision or not)
    - counting_threshold: used for determining if we COUNT as collision (requires 65%+ confidence)
    """
    # Standard model prediction (for comparison purposes)
    y_pred_model = (y_prob >= prediction_threshold).astype(int)
    
    # Collision counting prediction (requires high confidence to count)
    y_pred_count = (y_prob >= counting_threshold).astype(int)

    metrics = dict(
        split      = label,
        accuracy   = round(accuracy_score(y_true, y_pred_model), 4),
        roc_auc    = round(roc_auc_score(y_true, y_prob),  4),
        f1         = round(f1_score(y_true, y_pred_model, zero_division=0), 4),
        precision  = round(precision_score(y_true, y_pred_model, zero_division=0), 4),
        recall     = round(recall_score(y_true, y_pred_model, zero_division=0), 4),
        brier_score= round(brier_score_loss(y_true, y_prob), 4),
        prediction_threshold = round(prediction_threshold, 4),
        counting_threshold = round(counting_threshold, 4),
        # Metrics for counting-based predictions (high confidence only)
        count_accuracy = round(accuracy_score(y_true, y_pred_count), 4),
        count_f1 = round(f1_score(y_true, y_pred_count, zero_division=0), 4),
        count_precision = round(precision_score(y_true, y_pred_count, zero_division=0), 4),
        count_recall = round(recall_score(y_true, y_pred_count, zero_division=0), 4),
    )

    log.info(
        "[%s] Model Pred: AUC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f | Count Pred (>=%.0f%%): F1=%.4f  Prec=%.4f  Rec=%.4f",
        label, metrics["roc_auc"], metrics["f1"],
        metrics["precision"], metrics["recall"],
        counting_threshold*100, metrics["count_f1"],
        metrics["count_precision"], metrics["count_recall"],
    )
    
    print(f"\n── Model Prediction Report ({label}) ──────────────────────────────")
    print("Using threshold:", prediction_threshold)
    print(classification_report(y_true, y_pred_model,
                                target_names=["No Collision", "Collision"]))
    
    print(f"\n── Collision Counting Report ({label}) ──────────────────────────")
    print(f"Counting only predictions with confidence >= {counting_threshold*100}%")
    print(classification_report(y_true, y_pred_count,
                                target_names=["No Collision", "Collision"]))

    # Confusion matrix for model predictions
    cm_model = confusion_matrix(y_true, y_pred_model)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model predictions confusion matrix
    ConfusionMatrixDisplay(cm_model, display_labels=["No Collision", "Collision"]).plot(
        ax=axes[0], colorbar=True, cmap="Blues"
    )
    total_model = cm_model.sum()
    for i in range(2):
        for j in range(2):
            axes[0].text(
                j, i,
                f"\n{cm_model[i, j]}\n({100 * cm_model[i, j] / total_model:.1f}%)",
                ha="center", va="center", fontsize=9,
                color="white" if cm_model[i, j] > cm_model.max() * 0.6 else "black",
            )
    axes[0].set_title(f"Model Predictions (threshold={prediction_threshold})", fontsize=10, fontweight="bold")
    
    # Collision counting confusion matrix
    cm_count = confusion_matrix(y_true, y_pred_count)
    total_count = cm_count.sum()
    ConfusionMatrixDisplay(cm_count, display_labels=["No Collision", "Collision"]).plot(
        ax=axes[1], colorbar=True, cmap="Greens"
    )
    for i in range(2):
        for j in range(2):
            axes[1].text(
                j, i,
                f"\n{cm_count[i, j]}\n({100 * cm_count[i, j] / total_count:.1f}%)",
                ha="center", va="center", fontsize=9,
                color="white" if cm_count[i, j] > cm_count.max() * 0.6 else "black",
            )
    axes[1].set_title(f"Collision Counting (threshold={counting_threshold})", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    fname = f"confusion_matrices_{label.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 8. SHAP analysis
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linewidth=0.6)


def run_shap_analysis(model, X_val: pd.DataFrame, y_val: pd.Series) -> None:
    log.info("SHAP: computing values …")

    n        = min(600, len(X_val))
    X_sample = X_val.sample(n=n, random_state=42)
    y_sample = y_val.loc[X_sample.index]

    # Extract the base XGBoost estimator from a calibrated wrapper if needed
    base_model = (
        model.calibrated_classifiers_[0].estimator
        if hasattr(model, "calibrated_classifiers_")
        else model
    )

    explainer   = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_sample)
    # Older SHAP returns a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feat_names     = X_sample.columns.tolist()
    readable_names = [FEATURE_LABELS.get(f, f) for f in feat_names]
    X_readable     = X_sample.rename(columns=FEATURE_LABELS)

    mean_abs = np.abs(shap_values).mean(axis=0)

    # ── 1. Bar chart ──────────────────────────────────────────────────────
    top15_idx    = np.argsort(mean_abs)[::-1][:15]
    bar_names    = [readable_names[i] for i in top15_idx]
    bar_values   = mean_abs[top15_idx]
    bar_colors   = [_feature_color(feat_names[i]) for i in top15_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top15_idx)), bar_values, color=bar_colors, edgecolor="none")
    ax.set_yticks(range(len(top15_idx)))
    ax.set_yticklabels(bar_names, fontsize=9)
    ax.invert_yaxis()
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=g)
        for g, c in FEATURE_GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7)
    _style_ax(ax,
              title="Top-15 features by mean |SHAP value|",
              xlabel="Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved shap_bar.png")

    # ── 2. Beeswarm ───────────────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    shap.summary_plot(
        shap_values, X_readable,
        plot_type="dot", max_display=15, show=False,
        color_bar_label="Feature value (low → high)",
    )
    ax = plt.gca()
    _style_ax(ax,
              title="How feature values shift collision probability",
              xlabel="SHAP value  (negative = lowers risk, positive = raises risk)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved shap_beeswarm.png")

    # ── 3. Dependence plots — top 3 features ──────────────────────────────
    top3_idx = np.argsort(mean_abs)[-3:][::-1]
    for i in top3_idx:
        raw_name  = feat_names[i]
        read_name = readable_names[i]
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.dependence_plot(
            raw_name, shap_values, X_sample,
            interaction_index="auto", ax=ax, show=False,
            dot_size=14, alpha=0.55,
        )
        ax.set_xlabel(read_name, fontsize=10)
        ax.set_ylabel(f"SHAP value for\n{read_name}", fontsize=10)
        _style_ax(ax, title=f"Effect of '{read_name}' on collision risk")
        plt.tight_layout()
        safe = read_name.replace(" ", "_").replace("/", "-")
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"shap_dependence_{safe}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
    log.info("  Saved dependence plots for top-3 features")

    # ── 4. Waterfall — highest-risk example ───────────────────────────────
    probs   = model.predict_proba(X_sample)[:, 1]
    pos_idx = int(np.argmax(probs))

    base_val = (
        float(explainer.expected_value)
        if not isinstance(explainer.expected_value, np.ndarray)
        else float(explainer.expected_value[1])
    )
    explanation = shap.Explanation(
        values       = shap_values[pos_idx],
        base_values  = base_val,
        data         = X_readable.iloc[pos_idx].values,
        feature_names= readable_names,
    )
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.suptitle(
        f"Why the model flagged this period as high-risk  "
        f"(predicted probability: {probs[pos_idx]:.0%})",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "shap_waterfall.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    log.info("  Saved shap_waterfall.png  (example prob = %.0f%%)", 100 * probs[pos_idx])

    # ── 5. Group importance (stacked bar) ─────────────────────────────────
    group_totals = {
        g: mean_abs[[i for i, f in enumerate(feat_names) if f in members]].sum()
        for g, members in FEATURE_GROUPS.items()
    }
    labels  = list(group_totals.keys())
    values  = list(group_totals.values())
    colors  = [FEATURE_GROUP_COLORS[g] for g in labels]
    total   = sum(values) or 1.0

    fig, ax = plt.subplots(figsize=(8, 3))
    left = 0.0
    for lbl, val, col in zip(labels, values, colors):
        ax.barh(0, val, left=left, color=col, label=lbl, height=0.45)
        if val / total > 0.04:
            ax.text(
                left + val / 2, 0,
                f"{lbl}\n{val / total:.0%}",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold",
            )
        left += val
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    _style_ax(ax,
              title="Which category of features drives the model?",
              xlabel="Total mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "shap_group_importance.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    log.info("  Saved shap_group_importance.png")

    # ── 6. Calibration curve ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    prob_before = base_model.predict_proba(X_sample)[:, 1]
    frac_b, pred_b = calibration_curve(y_sample, prob_before, n_bins=10)
    ax.plot(pred_b, frac_b, "o-", color="#D85A30",
            label="Before calibration", linewidth=1.5)

    prob_after = model.predict_proba(X_sample)[:, 1]
    frac_a, pred_a = calibration_curve(y_sample, prob_after, n_bins=10)
    ax.plot(pred_a, frac_a, "s-", color="#1D9E75",
            label="After calibration", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    _style_ax(ax,
              title="Are the model's probabilities trustworthy?",
              xlabel="Mean predicted probability",
              ylabel="Fraction of actual collisions")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "shap_calibration.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    log.info("  Saved shap_calibration.png")

    log.info("SHAP: all plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Predictions, forecasts, rankings (modified to use counting threshold)
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_band(p: float) -> str:
    if p < 0.25: return "Low"
    if p < 0.50: return "Medium"
    if p < 0.75: return "High"
    return "Very High"


def generate_weekly_predictions(
    model,
    grid: pd.DataFrame,
    prediction_threshold: float,
    counting_threshold: float,
) -> pd.DataFrame:
    log.info("Generating 5-day period probability predictions …")
    probs = model.predict_proba(grid[FEATURE_COLS])[:, 1]

    out = grid[["neighbourhood_158", "period_start", "collision", "is_test"]].copy()
    out = out.rename(columns={
        "period_start": "period",
        "collision":    "actual_collision",
        "is_test":      "is_test_period",
    })
    out["collision_probability"] = probs.round(4)
    out["model_prediction"]   = (probs >= prediction_threshold).astype(int)
    out["count_as_collision"] = (probs >= counting_threshold).astype(int)  # Only count if 65%+ confidence
    out["confidence_band"]       = out["collision_probability"].apply(_confidence_band)
    out = out.sort_values(["neighbourhood_158", "period"]).reset_index(drop=True)

    path = os.path.join(OUTPUT_DIR, "collision_period_predictions.csv")
    out.to_csv(path, index=False)
    log.info("Saved collision_period_predictions.csv  (%s rows)", f"{len(out):,}")
    
    # Log summary of counting threshold impact
    pred_count = out["model_prediction"].sum()
    count_count = out["count_as_collision"].sum()
    log.info("Model predicts collisions in %d periods, but only %d periods have confidence >= %.0f%% to count as collisions",
             pred_count, count_count, counting_threshold*100)
    
    return out


def forecast_next_collision(
    model,
    grid: pd.DataFrame,
    counting_threshold: float,
) -> pd.DataFrame:
    log.info("Forecasting next %d periods (5-day intervals) for all neighbourhoods …", 
             FORECAST_PERIODS_AHEAD)

    last_period = grid["period_start"].max()
    all_nh    = grid["neighbourhood_158"].unique()
    today     = pd.Timestamp(datetime.now().date())
    future_records: list[dict] = []

    for nh in all_nh:
        nh_hist = (
            grid[grid["neighbourhood_158"] == nh]
            .sort_values("period_start")
        )
        if nh_hist.empty:
            continue

        # Carry forward the most recent feature values
        last_row = nh_hist.iloc[-1]

        for p in range(1, FORECAST_PERIODS_AHEAD + 1):
            fp    = last_period + pd.Timedelta(days=DAYS_PER_PERIOD * p)
            month = fp.month
            period_num = ((fp - pd.Timestamp(fp.year, 1, 1)).days // DAYS_PER_PERIOD) + 1
            yr    = fp.year

            future_records.append({
                "neighbourhood_158":        nh,
                "period_start":             fp,
                "neighbourhood_enc":        last_row["neighbourhood_enc"],
                "month":                    month,
                "month_sin":                np.sin(2 * np.pi * month / 12),
                "month_cos":                np.cos(2 * np.pi * month / 12),
                "quarter":                  (month - 1) // 3 + 1,
                "period_of_year":           period_num,
                "period_sin":               np.sin(2 * np.pi * period_num / PERIODS_PER_YEAR),
                "period_cos":               np.cos(2 * np.pi * period_num / PERIODS_PER_YEAR),
                "year":                     yr,
                "season":                   _season(month),
                "is_holiday_season":        _is_holiday_season(month),
                "hist_rate_3w":             last_row["hist_rate_3w"],
                "hist_rate_6w":             last_row["hist_rate_6w"],
                "ewma_3w":                  last_row["ewma_3w"],
                "rate_change_3w":           last_row["rate_change_3w"],
                "periods_since_last":       float(last_row["periods_since_last"]) + p,
                "collision_trend":          last_row["collision_trend"],
                "high_risk_cluster":        last_row["high_risk_cluster"],
                "days_to_holiday":          30,     # unknown future; capped value
                "recent_collision_density": 0.0,
                "seasonal_pattern":         last_row["seasonal_pattern"],
            })

    if not future_records:
        log.warning("No future records generated — returning empty DataFrame.")
        return pd.DataFrame()

    future_df = pd.DataFrame(future_records)
    # Compute days_to_holiday for the actual future dates
    holiday_idx = _build_holiday_index(future_df["year"].unique())
    future_df["days_to_holiday"] = _days_to_nearest_holiday(
        future_df["period_start"], holiday_idx
    )

    future_df["prob"] = model.predict_proba(future_df[FEATURE_COLS])[:, 1].round(4)
    future_df["conf_band"] = future_df["prob"].apply(_confidence_band)
    future_df["count_as_collision"] = (future_df["prob"] >= counting_threshold).astype(int)

    results: list[dict] = []
    for nh in all_nh:
        nh_fut = future_df[future_df["neighbourhood_158"] == nh].sort_values("period_start")
        if nh_fut.empty:
            continue

        # For forecasting, we want the first period where we would COUNT it as a collision (>=65% confidence)
        count_periods = nh_fut[nh_fut["prob"] >= counting_threshold]
        if count_periods.empty:
            # If no period reaches counting threshold, take the highest probability period
            best = nh_fut.loc[nh_fut["prob"].idxmax()]
            note = f"Best guess (no period ≥ {counting_threshold:.0%} confidence)"
        else:
            best = count_periods.iloc[0]
            note = f"Predicted (≥{counting_threshold:.0%} confidence)"

        results.append({
            "neighbourhood":          nh,
            "next_predicted_period":  best["period_start"].date(),
            "collision_probability":  best["prob"],
            "confidence_band":        best["conf_band"],
            "days_from_now":          max(int((best["period_start"] - today).days), 1),
            "note":                   note,
        })

    forecast_df = (
        pd.DataFrame(results)
        .sort_values("collision_probability", ascending=False)
        .reset_index(drop=True)
    )
    path = os.path.join(OUTPUT_DIR, "next_collision_forecast.csv")
    forecast_df.to_csv(path, index=False)
    log.info("Saved next_collision_forecast.csv  (%d neighbourhoods)", len(forecast_df))
    return forecast_df


def generate_risk_ranking(period_preds: pd.DataFrame, counting_threshold: float) -> pd.DataFrame:
    ranking = (
        period_preds
        .groupby("neighbourhood_158")
        .agg(
            avg_period_risk         = ("collision_probability", "mean"),
            max_period_risk         = ("collision_probability", "max"),
            risk_std                = ("collision_probability", "std"),
            predicted_collision_periods = ("model_prediction", "sum"),
            counted_collision_periods = ("count_as_collision", "sum"),  # Periods with 65%+ confidence
        )
        .reset_index()
        .sort_values("avg_period_risk", ascending=False)
        .reset_index(drop=True)
    )
    ranking["risk_rank"]        = ranking.index + 1
    ranking["avg_period_risk"]  = ranking["avg_period_risk"].round(4)
    ranking["max_period_risk"]  = ranking["max_period_risk"].round(4)
    ranking["risk_std"]         = ranking["risk_std"].round(4)
    ranking["confidence_band"]  = ranking["avg_period_risk"].apply(_confidence_band)

    path = os.path.join(OUTPUT_DIR, "neighbourhood_risk_ranking.csv")
    ranking.to_csv(path, index=False)
    log.info("Saved neighbourhood_risk_ranking.csv")
    return ranking


# ─────────────────────────────────────────────────────────────────────────────
# 10. Persist model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model,
    le: LabelEncoder,
    prediction_threshold: float,
    counting_threshold: float,
    best_params: Optional[dict],
) -> None:
    bundle = dict(
        model            = model,
        label_encoder    = le,
        feature_cols     = FEATURE_COLS,
        cutoff_year      = CUTOFF_YEAR,
        granularity      = "5day",
        prediction_threshold = prediction_threshold,
        counting_threshold = counting_threshold,
        best_params      = best_params,
        trained_at       = datetime.now().isoformat(),
        version          = "4.0_5day_counting",
    )
    path = os.path.join(OUTPUT_DIR, "collision_xgboost_model_5day.pkl")
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    log.info("Saved collision_xgboost_model_5day.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  Traffic Collision Prediction — XGBoost + SHAP  (v4.0 - 5-day periods)")
    print(f"  Started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Counting threshold: {COLLISION_COUNT_THRESHOLD*100}% (collisions counted only with >=65% confidence)")
    print("=" * 70)

    # ── 1. Data ───────────────────────────────────────────────────────────
    df   = load_data()
    grid, le = engineer_features(df)

    # ── 2. Validation split (reused for calibration + threshold) ─────────
    pre     = grid[grid["is_test"] == 0]
    X_pre   = pre[FEATURE_COLS]
    y_pre   = pre["collision"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pre, y_pre, test_size=VAL_SIZE, random_state=42, stratify=y_pre
    )

    # ── 3. Optional: Optuna hyperparameter search ─────────────────────────
    best_params: Optional[dict] = None
    if USE_OPTUNA and OPTUNA_AVAILABLE:
        best_params = optimize_hyperparameters(X_tr, y_tr, X_val, y_val)
    elif USE_OPTUNA and not OPTUNA_AVAILABLE:
        log.warning("USE_OPTUNA=True but optuna is not installed — skipping.")

    # ── 4. Train ──────────────────────────────────────────────────────────
    model, X_pre, y_pre, X_test, y_test = train_model(grid, best_params)

    # ── 5. Calibrate ──────────────────────────────────────────────────────
    model = calibrate_model(model, X_val, y_val)

    # ── 6. Find optimal threshold (for model prediction, NOT counting) ────
    prediction_threshold = find_optimal_threshold(model, X_val, y_val)
    
    # Use fixed counting threshold (65% confidence required to count as collision)
    counting_threshold = COLLISION_COUNT_THRESHOLD

    # ── 7. Evaluate (showing both model prediction and counting metrics) ──
    log.info("Evaluating with separate thresholds for prediction (%.3f) and counting (%.0f%%)...", 
             prediction_threshold, counting_threshold*100)
    val_prob  = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    val_metrics  = evaluate("Validation", y_val,  val_prob,  prediction_threshold, counting_threshold)
    test_metrics = evaluate("Test",       y_test, test_prob, prediction_threshold, counting_threshold)

    pd.DataFrame([val_metrics, test_metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "model_accuracy_report.csv"), index=False
    )

    # ── 8. SHAP ───────────────────────────────────────────────────────────
    try:
        run_shap_analysis(model, X_val, y_val)
    except Exception as exc:
        log.error("SHAP analysis failed: %s", exc, exc_info=True)

    # ── 9. Period predictions (with both prediction and counting flags) ───
    period_preds = generate_weekly_predictions(model, grid, prediction_threshold, counting_threshold)

    # ── 10. Forecast (using counting threshold) ───────────────────────────
    forecast_df  = forecast_next_collision(model, grid, counting_threshold)

    # ── 11. Risk ranking (shows both predicted and counted periods) ───────
    ranking      = generate_risk_ranking(period_preds, counting_threshold)

    # ── 12. Persist ───────────────────────────────────────────────────────
    save_model(model, le, prediction_threshold, counting_threshold, best_params)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print(f"  Validation  AUC={val_metrics['roc_auc']:.4f}  F1={val_metrics['f1']:.4f}"
          f"  Brier={val_metrics['brier_score']:.4f}")
    print(f"  Test        AUC={test_metrics['roc_auc']:.4f}  F1={test_metrics['f1']:.4f}"
          f"  Brier={test_metrics['brier_score']:.4f}")
    print(f"  Prediction Threshold (model says 'collision'): {prediction_threshold:.3f}")
    print(f"  Counting Threshold (count as collision): {counting_threshold*100}%")
    print(f"  Output dir  {OUTPUT_DIR}")
    
    # Summary of counting impact
    counted_periods = period_preds[period_preds['count_as_collision'] == 1].shape[0]
    predicted_periods = period_preds[period_preds['model_prediction'] == 1].shape[0]
    print(f"\n  IMPACT OF COUNTING THRESHOLD:")
    print(f"  Model predicts collisions in {predicted_periods:,} periods")
    print(f"  Only {counted_periods:,} periods have >= {counting_threshold*100}% confidence to count as collisions")
    print(f"  {predicted_periods - counted_periods:,} periods are below confidence threshold")
    print("=" * 70)


if __name__ == "__main__":
    main()