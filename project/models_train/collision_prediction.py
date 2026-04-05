"""
Traffic Collision Prediction — LightGBM + SHAP (Priority: Collision Recall)
============================================================================
Just edit the CONFIG section below, then run:

    pip install lightgbm joblib
    python collision_prediction_lgbm.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    f1_score, recall_score, precision_score
)
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import shap
import lightgbm as lgb       # ← LightGBM replaces XGBoost
import joblib
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← edit everything here, then just run the script
# ═══════════════════════════════════════════════════════════════════════════════

CSV_PATH = Path("project/DB_csv/Traffic_Collisions_Data_enhanced.csv")
OUT_DIR  = Path("project/outputs")

PREDICT_START = None
WINDOW_DAYS   = 3
TOP_N         = 10

TRAIN_YEARS_UP_TO = 2024
VALIDATION_YEARS  = [2025]

# ── LightGBM base hyper-parameters ───────────────────────────────────────────
LGB_N_ESTIMATORS     = 500
LGB_LEARNING_RATE    = 0.05
LGB_NUM_LEAVES       = 63        # LightGBM uses num_leaves instead of max_depth
LGB_MAX_DEPTH        = -1        # -1 = no limit; num_leaves controls complexity
LGB_MIN_CHILD_SAMPLES = 20       # LightGBM equivalent of XGB min_child_weight
LGB_SUBSAMPLE        = 0.8       # row subsampling (bagging_fraction)
LGB_COLSAMPLE_BYTREE = 0.8       # feature subsampling (feature_fraction)
LGB_REG_ALPHA        = 0.1       # L1 regularisation
LGB_REG_LAMBDA       = 1.0       # L2 regularisation
LGB_RANDOM_STATE     = 42
COLLISION_BOOST      = 1.5       # Multiplier on scale_pos_weight to force higher collision recall

# ── Hyperparameter Tuning & Optimization ─────────────────────────────────────
TUNE_ENABLED    = True
TUNE_ITERATIONS = 30
TUNE_OBJECTIVE  = "recall_1"   # "recall_1", "f1_1", or "composite"
MIN_RECALL_0    = 0.25         # Minimum acceptable non-collision recall floor
TUNE_THRESHOLD  = 0.35         # Threshold used during tuning (biases toward collision)

# ── SHAP sample sizes ────────────────────────────────────────────────────────
SHAP_BG_SAMPLES  = 500
SHAP_VAL_SAMPLES = 300

# ═══════════════════════════════════════════════════════════════════════════════
#  END OF CONFIG — nothing below needs to be changed
# ═══════════════════════════════════════════════════════════════════════════════

SEASON_MAP  = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3, "Autumn": 3}
WEEKDAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

FEATURE_COLS = [
    "neighbourhood_enc", "week_day_num", "season_num", "holiday",
    "hist_rate_3w", "hist_rate_6w", "periods_since_last",
    "month_sin", "month_cos", "period_sin", "period_cos",
    "ewma_3w", "rate_change_3w", "collision_trend",
    "recent_collision_density", "seasonal_pattern",
    "year_continuous", "global_yoy_change"
]

FEATURE_DISPLAY_NAMES = [
    "Neighbourhood", "Weekday", "Season", "Holiday",
    "Hist Rate 3w", "Hist Rate 6w", "Periods Since Last",
    "Month Sin", "Month Cos", "Period Sin", "Period Cos",
    "EWMA 3w", "Rate Change 3w", "Collision Trend",
    "Recent Density", "Seasonal Pattern",
    "Year Continuous", "Global YoY Change"
]


def _date_to_season(dt):
    m = dt.month
    if m in (12, 1, 2): return 0
    if m in (3, 4, 5):  return 1
    if m in (6, 7, 8):  return 2
    return 3


def _is_holiday(dt):
    statutory = {(1,1),(2,14),(7,1),(11,11),(12,25),(12,26)}
    return int((dt.month, dt.day) in statutory)


def _save(filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path):
    print(f"\n[1/8] Loading data from: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\nCSV not found at '{csv_path}'.\n"
            "Please update CSV_PATH in the CONFIG section at the top of this script."
        )

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()

    df["OCC_DATE"] = pd.to_datetime(df["OCC_DATE"], errors="coerce")
    df = df.dropna(subset=["OCC_DATE"])
    df["DATE"] = df["OCC_DATE"].dt.normalize()

    df = df[["DATE", "NEIGHBOURHOOD_158", "WEEK_DAY", "SEASON", "HOLIDAY"]].copy()
    df["NEIGHBOURHOOD_158"] = df["NEIGHBOURHOOD_158"].str.strip()
    df["HOLIDAY"] = (
        df["HOLIDAY"].astype(str).str.strip().str.lower()
        .map({"true": 1, "false": 0, "yes": 1, "no": 0})
        .fillna(0).astype(int)
    )

    print(f"    {len(df):,} records  ({df['DATE'].min().date()} -> {df['DATE'].max().date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD WINDOWS
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(df):
    print(f"[2/8] Building {WINDOW_DAYS}-day prediction windows ...")

    neighbourhoods = df["NEIGHBOURHOOD_158"].dropna().unique()
    date_range = pd.date_range(df["DATE"].min(), df["DATE"].max(), freq=f"{WINDOW_DAYS}D")

    records = []
    for start in date_range:
        end   = start + pd.Timedelta(days=WINDOW_DAYS - 1)
        chunk = df[(df["DATE"] >= start) & (df["DATE"] <= end)]

        wd_val   = WEEKDAY_MAP.get(start.day_name(), -1)
        seas_val = _date_to_season(start)
        hol_val  = int(chunk["HOLIDAY"].max()) if len(chunk) > 0 else 0

        for neigh in neighbourhoods:
            label = 1 if len(chunk[chunk["NEIGHBOURHOOD_158"] == neigh]) > 0 else 0
            records.append({
                "window_start":  start,
                "neighbourhood": neigh,
                "week_day_num":  wd_val,
                "season_num":    seas_val,
                "holiday":       hol_val,
                "collision":     label,
                "year":          start.year,
            })

    result = pd.DataFrame(records)
    print(f"    {len(result):,} samples. Collision rate: {result['collision'].mean():.1%}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENGINEER TEMPORAL & SPATIAL FEATURES (STRICTLY CAUSAL)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(windows_df):
    print("[3/8] Computing historical, periodic & cross-year trend features (STRICTLY CAUSAL) ...")
    df = windows_df.copy().sort_values(['neighbourhood', 'window_start']).reset_index(drop=True)

    df['month_sin'] = np.sin(2 * np.pi * df['window_start'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['window_start'].dt.month / 12)
    df['period_sin'] = np.sin(2 * np.pi * df['window_start'].dt.isocalendar().week.astype(float) / 52)
    df['period_cos'] = np.cos(2 * np.pi * df['window_start'].dt.isocalendar().week.astype(float) / 52)
    df['year_continuous'] = df['window_start'].dt.year + df['window_start'].dt.dayofyear / 365.25

    for col in ['hist_rate_3w', 'hist_rate_6w', 'periods_since_last',
                'ewma_3w', 'rate_change_3w', 'collision_trend',
                'recent_collision_density', 'seasonal_pattern']:
        df[col] = 0.0

    city_rates = df.groupby('window_start')['collision'].mean()
    city_lagged = city_rates.shift(1)
    city_ly     = city_rates.shift(53)
    df['global_yoy_change'] = df['window_start'].map(city_lagged - city_ly).fillna(0.0)

    for neigh, grp in df.groupby('neighbourhood'):
        idx = grp.index
        coll = grp['collision'].values.astype(float)
        n = len(coll)

        past = np.full(n, np.nan)
        past[1:] = coll[:-1]
        s_past = pd.Series(past)

        df.loc[idx, 'hist_rate_3w'] = s_past.rolling(4, min_periods=1).mean().values
        df.loc[idx, 'hist_rate_6w'] = s_past.rolling(8, min_periods=1).mean().values

        recency = np.zeros(n, dtype=int)
        cnt = 0
        for i in range(n):
            if np.isnan(past[i]) or past[i] == 1.0:
                cnt = 0
            else:
                cnt += 1
            recency[i] = cnt
        df.loc[idx, 'periods_since_last'] = recency

        df.loc[idx, 'ewma_3w'] = s_past.ewm(span=4, adjust=False).mean().values
        df.loc[idx, 'rate_change_3w'] = df.loc[idx, 'hist_rate_3w'] - df.loc[idx, 'hist_rate_6w']
        df.loc[idx, 'recent_collision_density'] = s_past.rolling(2, min_periods=1).sum().values

        seasonal = np.full(n, 0.0)
        if n > 72:
            seasonal[72:] = coll[:-72]
        df.loc[idx, 'seasonal_pattern'] = seasonal

        slopes = np.zeros(n)
        for i in range(n):
            w = past[max(0, i-9):i+1]
            valid = w[~np.isnan(w)]
            if len(valid) >= 3:
                slopes[i] = np.polyfit(np.arange(len(valid)), valid, 1)[0]
        df.loc[idx, 'collision_trend'] = slopes

    print("    Feature engineering complete (no target leakage).")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODE & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def encode_and_split(windows_enriched):
    print(f"[4/8] Encoding and splitting (train <= {TRAIN_YEARS_UP_TO} | val = {VALIDATION_YEARS}) ...")

    le = LabelEncoder()
    windows_enriched["neighbourhood_enc"] = le.fit_transform(windows_enriched["neighbourhood"])

    X = windows_enriched[FEATURE_COLS].copy()
    y = windows_enriched["collision"]

    train_mask = windows_enriched["year"] <= TRAIN_YEARS_UP_TO
    val_mask   = windows_enriched["year"].isin(VALIDATION_YEARS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"    Train: {len(X_train):,}  |  Validation: {len(X_val):,}")

    if len(X_val) == 0:
        print(f"    WARNING: No {VALIDATION_YEARS} data — using last 10% of training as validation.")
        split    = int(len(X_train) * 0.9)
        X_val,   y_val   = X_train.iloc[split:],  y_train.iloc[split:]
        X_train, y_train = X_train.iloc[:split],  y_train.iloc[:split]

    return X_train, y_train, X_val, y_val, le, windows_enriched


# ─────────────────────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (CONSTRAINT-AWARE)
# ─────────────────────────────────────────────────────────────────────────────

def tune_lgbm(X_train, y_train, X_val, y_val):
    print(f"[5/8] Hyperparameter Tuning ({TUNE_ITERATIONS} iterations, objective='{TUNE_OBJECTIVE}') ...")

    # LightGBM-specific param space
    param_space = {
        'n_estimators':      stats.randint(100, 800),
        'learning_rate':     stats.uniform(0.01, 0.15),
        'num_leaves':        stats.randint(15, 255),   # key LightGBM complexity knob
        'max_depth':         stats.randint(3, 12),
        'min_child_samples': stats.randint(5, 100),    # LightGBM name for min_child_weight
        'subsample':         stats.uniform(0.6, 0.4),
        'colsample_bytree':  stats.uniform(0.6, 0.4),
        'reg_alpha':         stats.uniform(0, 5),
        'reg_lambda':        stats.uniform(0, 5),
        'min_split_gain':    stats.uniform(0, 1),      # LightGBM-specific regularisation
    }

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_w = (n_neg / max(n_pos, 1)) * COLLISION_BOOST

    results = []
    for i in range(TUNE_ITERATIONS):
        params = {k: v.rvs() for k, v in param_space.items()}
        params['n_estimators']      = int(params['n_estimators'])
        params['num_leaves']        = int(params['num_leaves'])
        params['max_depth']         = int(params['max_depth'])
        params['min_child_samples'] = int(params['min_child_samples'])
        params['scale_pos_weight']  = scale_w
        params['random_state']      = LGB_RANDOM_STATE
        params['n_jobs']            = -1
        params['verbose']           = -1           # LightGBM uses verbose=-1 for silence

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= TUNE_THRESHOLD).astype(int)

        rec_1  = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        prec_1 = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec_0  = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
        prec_0 = precision_score(y_val, y_pred, pos_label=0, zero_division=0)
        f1_1   = 2 * (prec_1 * rec_1) / (prec_1 + rec_1 + 1e-9)

        passes_floor = (rec_0 >= MIN_RECALL_0)
        if TUNE_OBJECTIVE == "recall_1":   score = rec_1
        elif TUNE_OBJECTIVE == "f1_1":     score = f1_1
        elif TUNE_OBJECTIVE == "composite": score = 0.7 * rec_1 + 0.3 * rec_0
        else:                               score = rec_1

        score = score if passes_floor else score * 0.4

        results.append({
            'iter': i+1, 'score': score,
            'recall_1': rec_1, 'prec_1': prec_1, 'f1_1': f1_1,
            'recall_0': rec_0, 'prec_0': prec_0,
            'params': params.copy()
        })
        if (i+1) % 5 == 0:
            print(f"    Iter {i+1}/{TUNE_ITERATIONS} | Rec@1={rec_1:.3f} | Rec@0={rec_0:.3f} | Score={score:.3f}")

    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]
    print(f"\n    ✅ Best Config (Iter {best['iter']}): Score={best['score']:.3f} | Rec@1={best['recall_1']:.3f} | Rec@0={best['recall_0']:.3f}")
    print("    Top 3 Configurations:")
    for r in results[:3]:
        p = r['params']
        print(f"      {r['iter']:2d}. Rec@1={r['recall_1']:.3f} | Rec@0={r['recall_0']:.3f} | lr={p['learning_rate']:.3f} leaves={p['num_leaves']}")

    best_params = {k: v for k, v in best['params'].items()
                   if k not in ['random_state', 'n_jobs', 'verbose']}
    return best_params, results


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN (LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, best_params=None):
    print("[6/8] Training LightGBM ...")

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_w = (n_neg / max(n_pos, 1)) * COLLISION_BOOST

    defaults = {
        'n_estimators':      LGB_N_ESTIMATORS,
        'learning_rate':     LGB_LEARNING_RATE,
        'num_leaves':        LGB_NUM_LEAVES,
        'max_depth':         LGB_MAX_DEPTH,
        'min_child_samples': LGB_MIN_CHILD_SAMPLES,
        'subsample':         LGB_SUBSAMPLE,
        'colsample_bytree':  LGB_COLSAMPLE_BYTREE,
        'reg_alpha':         LGB_REG_ALPHA,
        'reg_lambda':        LGB_REG_LAMBDA,
        'scale_pos_weight':  scale_w,
        'random_state':      LGB_RANDOM_STATE,
        'n_jobs':            -1,
        'verbose':           -1,
    }
    params = {**defaults, **(best_params or {})}

    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    print("    Training complete.")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATE & CONSTRAINED THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(clf, X_val, y_val):
    val_label = f"{VALIDATION_YEARS[0]}-{VALIDATION_YEARS[-1]}"
    print("[7/8] Evaluating & Optimizing Threshold (Priority: Collision Recall) ...")

    y_prob = clf.predict_proba(X_val)[:, 1]

    best_thresh, best_score = 0.5, -1.0
    for t in np.arange(0.15, 0.85, 0.01):
        pred = (y_prob >= t).astype(int)
        rec1 = recall_score(y_val, pred, pos_label=1, zero_division=0)
        rec0 = recall_score(y_val, pred, pos_label=0, zero_division=0)

        if rec0 >= MIN_RECALL_0:
            score = rec1
        else:
            score = rec1 * (rec0 / MIN_RECALL_0)

        if score > best_score:
            best_score, best_thresh = score, t

    threshold = best_thresh
    y_pred = (y_prob >= threshold).astype(int)

    rec1 = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    rec0 = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
    print(f"    Optimized threshold: {threshold:.2f} (Rec@1={rec1:.3f} | Rec@0={rec0:.3f})")

    print("\n" + "="*55)
    print(f"  CLASSIFICATION REPORT ({val_label}) — threshold={threshold:.2f}")
    print("="*55)
    print(classification_report(y_val, y_pred, target_names=["No Collision", "Collision"]))

    try:
        auc = roc_auc_score(y_val, y_prob)
        print(f"  ROC-AUC: {auc:.4f}")
    except Exception:
        auc = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Collision Prediction — {val_label} Validation", fontsize=15, fontweight="bold")
    ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred),
                           display_labels=["No Collision", "Collision"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix (Counts)")
    ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred, normalize="true"),
                           display_labels=["No Collision", "Collision"]).plot(
        ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion Matrix (Normalised)")
    axes[1].set_ylabel("")
    plt.tight_layout()
    _save("confusion_matrix.png")

    if auc is not None:
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.plot(fpr, tpr, lw=2, label=f"LightGBM (AUC = {auc:.3f})")
        ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve"); ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)
        plt.tight_layout()
        _save("roc_curve.png")

    # LightGBM feature importances (gain) — same API as XGBoost
    importances = pd.Series(clf.feature_importances_, index=FEATURE_DISPLAY_NAMES).sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    importances.plot(kind="barh", ax=ax3, color="#4C72B0", edgecolor="white")
    ax3.set_title("Feature Importances (Gain)"); ax3.set_xlabel("Total Gain"); ax3.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save("feature_importances.png")

    return y_pred, y_prob, threshold


# ─────────────────────────────────────────────────────────────────────────────
# 8. SHAP
# ─────────────────────────────────────────────────────────────────────────────

def run_shap(clf, X_train, X_val):
    print("[8/8] Computing SHAP values ...")

    bg_sample  = X_train.sample(min(SHAP_BG_SAMPLES,  len(X_train)), random_state=42)
    val_sample = X_val.sample(  min(SHAP_VAL_SAMPLES, len(X_val)),   random_state=42)

    # shap.TreeExplainer works natively with LightGBM — no API change needed
    explainer   = shap.TreeExplainer(clf, bg_sample)
    shap_values = explainer.shap_values(val_sample, check_additivity=False)

    # LightGBM TreeExplainer returns a list [shap_class0, shap_class1]
    if isinstance(shap_values, list):
        shap_pos = shap_values[1]
    elif shap_values.ndim == 3:
        shap_pos = shap_values[:, :, 1]
    else:
        shap_pos = shap_values

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])

    shap.summary_plot(shap_pos, val_sample, feature_names=FEATURE_DISPLAY_NAMES, show=False, plot_size=(10, 6))
    plt.title("SHAP Summary — Impact on Collision Probability", fontsize=12, pad=10)
    plt.tight_layout(); _save("shap_summary.png")

    shap.summary_plot(shap_pos, val_sample, feature_names=FEATURE_DISPLAY_NAMES, plot_type="bar", show=False, plot_size=(8, 5))
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=12, pad=10)
    plt.tight_layout(); _save("shap_bar.png")

    top_idx  = int(np.abs(shap_pos).mean(axis=0).argmax())
    top_name = FEATURE_DISPLAY_NAMES[top_idx]
    fig_d, ax_d = plt.subplots(figsize=(7, 4))
    ax_d.scatter(val_sample.iloc[:, top_idx].values, shap_pos[:, top_idx],
                 alpha=0.5, edgecolors="none", color="#4C72B0", s=20)
    ax_d.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_d.set_xlabel(top_name); ax_d.set_ylabel(f"SHAP value for '{top_name}'")
    ax_d.set_title(f"SHAP Dependence — '{top_name}'"); ax_d.grid(alpha=0.3)
    plt.tight_layout(); _save("shap_dependence_top.png")

    exp = shap.Explanation(
        values=shap_pos[0], base_values=base_val,
        data=val_sample.iloc[0].values, feature_names=FEATURE_DISPLAY_NAMES,
    )
    shap.plots.waterfall(exp, show=False, max_display=10)
    plt.title("SHAP Waterfall — Single Prediction Explained", fontsize=11, pad=10)
    plt.tight_layout(); _save("shap_waterfall.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PREDICT WINDOW
# ─────────────────────────────────────────────────────────────────────────────

def predict_window(clf, le, start_date, windows_enriched, threshold=0.5):
    start = pd.Timestamp(start_date)
    end   = start + pd.Timedelta(days=WINDOW_DAYS - 1)

    print(f"\n{'='*55}")
    print(f"  PREDICTION WINDOW: {start.date()} -> {end.date()}")
    print(f"{'='*55}")

    historical = windows_enriched[windows_enriched["window_start"] < start].copy()
    if historical.empty:
        print("    ⚠️ No historical data available before prediction window. Using defaults.")
        historical = windows_enriched[windows_enriched["year"] < VALIDATION_YEARS[0]].copy()

    latest_state = historical.sort_values('window_start').groupby('neighbourhood').last()

    rows = []
    for n in le.classes_:
        if n in latest_state.index:
            row = latest_state.loc[n].to_dict()
            row['week_day_num']      = WEEKDAY_MAP.get(start.day_name(), 0)
            row['season_num']        = _date_to_season(start)
            row['holiday']           = _is_holiday(start)
            row['month_sin']         = np.sin(2 * np.pi * start.month / 12)
            row['month_cos']         = np.cos(2 * np.pi * start.month / 12)
            row['period_sin']        = np.sin(2 * np.pi * start.isocalendar().week / 52)
            row['period_cos']        = np.cos(2 * np.pi * start.isocalendar().week / 52)
            row['neighbourhood_enc'] = le.transform([n])[0]
            row['neighbourhood']     = n
            row['year_continuous']   = start.year + start.dayofyear / 365.25
        else:
            row = {col: 0.0 for col in FEATURE_COLS}
            row['neighbourhood_enc'] = le.transform([n])[0]
            row['week_day_num']      = WEEKDAY_MAP.get(start.day_name(), 0)
            row['season_num']        = _date_to_season(start)
            row['holiday']           = _is_holiday(start)
            row['month_sin']         = np.sin(2 * np.pi * start.month / 12)
            row['month_cos']         = np.cos(2 * np.pi * start.month / 12)
            row['period_sin']        = np.sin(2 * np.pi * start.isocalendar().week / 52)
            row['period_cos']        = np.cos(2 * np.pi * start.isocalendar().week / 52)
            row['neighbourhood']     = n
            row['year_continuous']   = start.year + start.dayofyear / 365.25

        rows.append(row)

    pred_df = pd.DataFrame(rows)[FEATURE_COLS + ["neighbourhood"]]
    probs   = clf.predict_proba(pred_df[FEATURE_COLS])[:, 1]
    pred_df["collision_probability"] = probs
    pred_df["predicted_collision"]   = (probs >= threshold).astype(int)

    top = pred_df.sort_values("collision_probability", ascending=False).head(TOP_N)
    print(f"\n  Top {TOP_N} highest-risk neighbourhoods:")
    print(f"  {'Neighbourhood':<45} {'Prob':>6}  {'Collision?':>10}")
    print(f"  {'-'*65}")
    for _, row in top.iterrows():
        flag = "RED YES" if row["predicted_collision"] else "GREEN NO"
        print(f"  {row['neighbourhood']:<45} {row['collision_probability']:>5.1%}  {flag:>10}")

    return pred_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df               = load_and_preprocess(CSV_PATH)
    windows          = build_windows(df)
    windows_enriched = engineer_features(windows)
    X_train, y_train, X_val, y_val, le, windows_enriched = encode_and_split(windows_enriched)

    best_params = None
    if TUNE_ENABLED:
        best_params, _ = tune_lgbm(X_train, y_train, X_val, y_val)

    clf = train_model(X_train, y_train, best_params)
    y_pred, y_prob, threshold = evaluate(clf, X_val, y_val)
    run_shap(clf, X_train, X_val)

    # ── SAVE MODEL BUNDLE FOR REUSE ──────────────────────────────────────────
    model_bundle = {
        "model": clf,
        "label_encoder": le,
        "threshold": threshold,
        "feature_cols": FEATURE_COLS,
        "feature_display_names": FEATURE_DISPLAY_NAMES,
        "window_days": WINDOW_DAYS,
        "train_years_up_to": TRAIN_YEARS_UP_TO,
        "validation_years": VALIDATION_YEARS
    }
    model_path = OUT_DIR / "collision_model_bundle.joblib"
    joblib.dump(model_bundle, model_path)
    print(f"\n✅ Model bundle saved to: {model_path}")
    print("   To load in another app:")
    print("   >>> import joblib")
    print("   >>> from pathlib import Path")
    print("   >>> bundle = joblib.load(Path('path/to/collision_model_bundle.joblib'))")
    print("   >>> clf, le, thresh = bundle['model'], bundle['label_encoder'], bundle['threshold']")

    if PREDICT_START:
        start_date = PREDICT_START
    else:
        val_mask   = windows_enriched["year"].isin(VALIDATION_YEARS)
        first_val  = windows_enriched[val_mask]["window_start"].min()
        start_date = str(first_val.date()) if not pd.isnull(first_val) else f"{VALIDATION_YEARS[0]}-01-01"

    predict_window(clf, le, start_date, windows_enriched, threshold)

    print(f"\nDone. All outputs saved to: {OUT_DIR.absolute()}/")
    print("  collision_model_bundle.joblib  ← Load this in other apps!")
    print("  confusion_matrix.png | roc_curve.png | feature_importances.png")
    print("  shap_summary.png | shap_bar.png | shap_dependence_top.png | shap_waterfall.png")


if __name__ == "__main__":
    main()