"""
train_model.py — Standalone ML training script for E-Commerce Sales Prediction
===============================================================================
Trains, tunes, evaluates, and saves the best regression model to predict
order-level sales revenue.

Usage (from the project root):
    python ecommerce_sales/train_model.py
    python ecommerce_sales/train_model.py --data path/to/custom.csv
    python ecommerce_sales/train_model.py --tune          # enable GridSearchCV tuning
    python ecommerce_sales/train_model.py --tune --no-plot  # skip plots

Outputs (written to ecommerce_sales/models/):
    sales_model.joblib       — serialised best estimator
    feature_columns.json     — ordered feature column list for inference
    model_meta.json          — metrics, CV scores, feature importances
    training_report.txt      — human-readable training summary
"""

import os
import sys
import json
import time
import argparse
import logging
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    KFold,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "clean_final_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sales_model.joblib")
META_PATH  = os.path.join(MODELS_DIR, "model_meta.json")
COLS_PATH  = os.path.join(MODELS_DIR, "feature_columns.json")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.txt")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load and lightly validate the raw CSV."""
    log.info(f"Loading data from: {path}")
    df = pd.read_csv(path, parse_dates=["OrderDate", "SignupDate"])

    required = [
        "OrderDate", "SignupDate", "Quantity", "Discount",
        "PaymentMethod", "Status", "Age", "City",
        "CustomerSegment", "Category", "UnitPrice", "Sales",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    log.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    log.info(f"  Date range: {df['OrderDate'].min().date()} → {df['OrderDate'].max().date()}")
    log.info(f"  Sales range: ${df['Sales'].min():.2f} – ${df['Sales'].max():.2f}  "
             f"(mean ${df['Sales'].mean():.2f})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features from the raw DataFrame.

    New columns added
    -----------------
    order_year, order_month, order_dayofweek, order_quarter, order_weekofyear
    tenure_days      : days between signup and order
    effective_price  : unit_price after discount
    revenue          : effective_price × quantity
    is_weekend       : 1 if order placed Sat/Sun
    is_discounted    : 1 if discount > 0
    One-hot dummies  : Category, PaymentMethod, Status, CustomerSegment, City
    """
    log.info("Engineering features …")
    df = df.copy()

    # Date features
    df["order_year"]       = df["OrderDate"].dt.year
    df["order_month"]      = df["OrderDate"].dt.month
    df["order_dayofweek"]  = df["OrderDate"].dt.dayofweek      # 0=Mon, 6=Sun
    df["order_quarter"]    = df["OrderDate"].dt.quarter
    df["order_weekofyear"] = df["OrderDate"].dt.isocalendar().week.astype(int)
    df["is_weekend"]       = df["order_dayofweek"].isin([5, 6]).astype(int)

    # Customer tenure
    df["tenure_days"] = (
        (df["OrderDate"] - df["SignupDate"]).dt.days.clip(lower=0)
    )

    # Price / discount features
    df["effective_price"] = df["UnitPrice"] * (1.0 - df["Discount"] / 100.0)
    df["revenue"]         = df["effective_price"] * df["Quantity"]
    df["is_discounted"]   = (df["Discount"] > 0).astype(int)

    # One-hot encode categoricals
    cat_cols = ["Category", "PaymentMethod", "Status", "CustomerSegment", "City"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    log.info(f"  Feature matrix shape after encoding: {df.shape}")
    return df


def build_feature_matrix(df_eng: pd.DataFrame):
    """
    Split engineered DataFrame into feature matrix X and target y.
    Drops ID columns, raw dates, and the target itself.
    """
    drop_cols = [
        "Sales", "OrderValue",      # target & duplicate
        "OrderID", "CustomerID",    # identifiers
        "ProductID",                # high-cardinality ID
        "OrderDate", "SignupDate",  # raw dates (already extracted)
        "ProductName",              # high-cardinality text
    ]
    drop_cols = [c for c in drop_cols if c in df_eng.columns]
    X = df_eng.drop(columns=drop_cols)
    y = df_eng["Sales"]
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_candidate_models() -> dict:
    """Return a dict of {name: estimator} for baseline comparison."""
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            min_samples_split=10,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ]),
    }


def get_tuning_grids() -> dict:
    """Hyperparameter grids for GridSearchCV (used only with --tune flag)."""
    return {
        "GradientBoosting": {
            "n_estimators":  [200, 300, 400],
            "learning_rate": [0.05, 0.08, 0.1],
            "max_depth":     [4, 5, 6],
            "subsample":     [0.7, 0.8, 1.0],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth":    [8, 12, None],
            "min_samples_split": [2, 5, 10],
        },
        "Ridge": {
            "ridge__alpha": [0.1, 1.0, 10.0, 100.0],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics for one set of predictions."""
    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = float(r2_score(y_true, y_pred))
    mape  = float(mean_absolute_percentage_error(y_true, y_pred)) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def cross_validate_all(candidates: dict, X_train, y_train, cv: int = 5) -> dict:
    """Run 5-fold CV on all candidates and return sorted results."""
    log.info(f"Running {cv}-fold cross-validation on {len(candidates)} models …")
    cv_results = {}
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in candidates.items():
        t0     = time.time()
        scores = cross_val_score(model, X_train, y_train,
                                 cv=kf, scoring="r2", n_jobs=-1)
        elapsed = time.time() - t0
        cv_results[name] = {
            "cv_r2_mean": float(scores.mean()),
            "cv_r2_std":  float(scores.std()),
            "cv_r2_min":  float(scores.min()),
            "cv_r2_max":  float(scores.max()),
            "cv_time_s":  round(elapsed, 2),
        }
        log.info(
            f"  {name:<20s}  R² = {scores.mean():.4f} ± {scores.std():.4f}"
            f"  ({elapsed:.1f}s)"
        )

    return cv_results


def tune_model(model, param_grid: dict, X_train, y_train, cv: int = 3):
    """GridSearchCV tuning — returns the best estimator."""
    log.info(f"  Tuning with GridSearchCV (cv={cv}) …")
    gs = GridSearchCV(
        model, param_grid, cv=cv,
        scoring="r2", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    log.info(f"  Best params : {gs.best_params_}")
    log.info(f"  Best CV R²  : {gs.best_score_:.4f}")
    return gs.best_estimator_


def get_feature_importances(model, feature_cols: list) -> dict:
    """Extract top-20 feature importances from tree-based models."""
    base = model[-1] if hasattr(model, "__len__") else model
    if not hasattr(base, "feature_importances_"):
        return {}
    imp = base.feature_importances_
    ranked = sorted(zip(feature_cols, imp.tolist()), key=lambda x: -x[1])
    return dict(ranked[:20])


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_divider(char: str = "─", width: int = 60):
    print(char * width)


def build_report(meta: dict) -> str:
    """Return a formatted text report from the meta dict."""
    lines = []
    lines.append("=" * 60)
    lines.append("  E-COMMERCE SALES PREDICTION — TRAINING REPORT")
    lines.append("=" * 60)
    lines.append(f"  Model selected  : {meta['model_name']}")
    lines.append(f"  Training samples: {meta['n_train']:,}")
    lines.append(f"  Test samples    : {meta['n_test']:,}")
    lines.append(f"  Features used   : {meta['n_features']}")
    lines.append("")
    lines.append("  ── Test-set metrics ──────────────────────────")
    lines.append(f"  R²    : {meta['test_r2']:.4f}")
    lines.append(f"  MAE   : ${meta['test_mae']:.2f}")
    lines.append(f"  RMSE  : ${meta['test_rmse']:.2f}")
    lines.append(f"  MAPE  : {meta['test_mape']:.2f}%")
    lines.append("")
    lines.append("  ── Cross-validation results (5-fold R²) ──────")
    for name, res in meta["cv_results"].items():
        marker = " ◀ selected" if name == meta["model_name"] else ""
        lines.append(
            f"  {name:<22s} {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}{marker}"
        )
    lines.append("")
    fi = meta.get("feature_importances", {})
    if fi:
        lines.append("  ── Top-10 feature importances ────────────────")
        for i, (feat, score) in enumerate(list(fi.items())[:10], 1):
            bar = "█" * int(score * 200)
            lines.append(f"  {i:2}. {feat:<30s} {score:.4f}  {bar}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(data_path: str, tune: bool = False, show_plot: bool = True):
    os.makedirs(MODELS_DIR, exist_ok=True)
    t_start = time.time()

    # ── Load & engineer ───────────────────────────────────────────────────────
    df_raw = load_data(data_path)
    df_eng = engineer_features(df_raw)
    X, y   = build_feature_matrix(df_eng)

    feature_cols = list(X.columns)
    with open(COLS_PATH, "w") as f:
        json.dump(feature_cols, f)
    log.info(f"  Saved feature columns ({len(feature_cols)}) → {COLS_PATH}")

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(f"  Split: {len(X_train):,} train / {len(X_test):,} test")

    # ── Cross-validate candidates ─────────────────────────────────────────────
    candidates  = get_candidate_models()
    cv_results  = cross_validate_all(candidates, X_train, y_train)

    # ── Pick best model by mean CV R² ─────────────────────────────────────────
    best_name = max(cv_results, key=lambda k: cv_results[k]["cv_r2_mean"])
    best_model = candidates[best_name]
    log.info(f"\nBest model: {best_name}  "
             f"(CV R²={cv_results[best_name]['cv_r2_mean']:.4f})")

    # ── Optional hyperparameter tuning ────────────────────────────────────────
    if tune:
        log.info(f"Tuning {best_name} …")
        grids = get_tuning_grids()
        if best_name in grids:
            best_model = tune_model(
                best_model, grids[best_name], X_train, y_train
            )
        else:
            log.info(f"  No tuning grid defined for {best_name}, skipping.")

    # ── Final fit & evaluation ────────────────────────────────────────────────
    log.info("Fitting best model on full training split …")
    best_model.fit(X_train, y_train)
    y_pred  = best_model.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)

    log.info(
        f"  R²={metrics['r2']:.4f}  MAE=${metrics['mae']:.2f}"
        f"  RMSE=${metrics['rmse']:.2f}  MAPE={metrics['mape']:.2f}%"
    )

    # ── Feature importances ───────────────────────────────────────────────────
    importances = get_feature_importances(best_model, feature_cols)

    # ── Build metadata ────────────────────────────────────────────────────────
    meta = {
        "model_name":          best_name,
        "tuned":               tune,
        "cv_results":          cv_results,
        "test_r2":             metrics["r2"],
        "test_mae":            metrics["mae"],
        "test_rmse":           metrics["rmse"],
        "test_mape":           metrics["mape"],
        "feature_importances": importances,
        "n_features":          len(feature_cols),
        "n_train":             len(X_train),
        "n_test":              len(X_test),
        "training_time_s":     round(time.time() - t_start, 1),
    }

    # ── Save model & metadata ─────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    log.info(f"  Model saved   → {MODEL_PATH}")

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Metadata saved → {META_PATH}")

    # ── Print & save report ───────────────────────────────────────────────────
    report = build_report(meta)
    print("\n" + report)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    log.info(f"  Report saved  → {REPORT_PATH}")

    log.info(f"\nDone in {meta['training_time_s']}s ✓")
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the e-commerce sales prediction model."
    )
    parser.add_argument(
        "--data", default=DATA_PATH,
        help=f"Path to the CSV file (default: {DATA_PATH})"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run GridSearchCV hyperparameter tuning on the best model."
    )
    parser.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="Suppress matplotlib residual plot (default: show if available)."
    )
    parser.set_defaults(plot=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(data_path=args.data, tune=args.tune, show_plot=args.plot)
