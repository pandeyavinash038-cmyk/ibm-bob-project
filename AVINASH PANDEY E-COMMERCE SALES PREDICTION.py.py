"""
train_model.py — Sales Revenue Prediction Model Training
=========================================================
Trains, tunes, evaluates, and persists the best regression model for
predicting order-level sales revenue on the Iranian e-commerce dataset.

Candidate models
----------------
  1. GradientBoostingRegressor  — primary; usually best on tabular data
  2. RandomForestRegressor       — strong ensemble baseline
  3. Ridge                       — fast linear baseline via Pipeline

Selection criterion: highest mean 5-fold cross-validated R² on training data.

CLI Usage (run from project root)
----------------------------------
  # Basic — train and save
  python ecommerce_sales/train_model.py

  # Point to a custom CSV
  python ecommerce_sales/train_model.py --data path/to/custom.csv

  # Also run GridSearchCV hyperparameter tuning (slower, more accurate)
  python ecommerce_sales/train_model.py --tune

  # Tune + skip matplotlib residual plot
  python ecommerce_sales/train_model.py --tune --no-plot

Outputs (written to ecommerce_sales/models/)
--------------------------------------------
  sales_model.joblib       — serialised best estimator (joblib)
  feature_columns.json     — ordered feature column list for inference
  model_meta.json          — all metrics, CV scores, feature importances
  training_report.txt      — human-readable training summary
"""

# ── Standard library ───────────────────────────────────────────────────────────
import os
import sys
import json
import time
import argparse
import logging
import warnings

# ── Third-party ────────────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data",   "clean_final_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "sales_model.joblib")
COLS_PATH   = os.path.join(MODELS_DIR, "feature_columns.json")
META_PATH   = os.path.join(MODELS_DIR, "model_meta.json")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.txt")

# Required columns — script aborts if any are missing
REQUIRED_COLS = [
    "OrderDate", "SignupDate",
    "Quantity", "Discount", "UnitPrice", "Sales",
    "PaymentMethod", "Status", "Age",
    "City", "CustomerSegment", "Category",
]

# Columns to drop before building the feature matrix
DROP_COLS = [
    "Sales", "OrderValue",       # regression target / exact duplicate
    "OrderID", "CustomerID",     # identifier columns
    "ProductID",                 # high-cardinality ID
    "OrderDate", "SignupDate",   # raw dates (already extracted into numerics)
    "ProductName",               # high-cardinality free text
]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Load the CSV, parse date columns, and validate required columns exist.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with OrderDate and SignupDate parsed as datetime.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    ValueError
        If any required columns are missing from the CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    log.info(f"Loading data from: {path}")
    df = pd.read_csv(path, parse_dates=["OrderDate", "SignupDate"])

    # ── Validate required columns ──────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # ── Basic data quality checks ──────────────────────────────────────────────
    null_counts = df[REQUIRED_COLS].isnull().sum()
    if null_counts.any():
        log.warning("Null values found in required columns:")
        for col, n in null_counts[null_counts > 0].items():
            log.warning(f"    {col}: {n} nulls ({n/len(df)*100:.1f}%)")

    log.info(f"  Rows × Cols    : {len(df):,} × {len(df.columns)}")
    log.info(f"  Date range     : {df['OrderDate'].min().date()} → {df['OrderDate'].max().date()}")
    log.info(f"  Sales range    : ${df['Sales'].min():.2f} – ${df['Sales'].max():.2f}")
    log.info(f"  Sales mean/std : ${df['Sales'].mean():.2f} ± ${df['Sales'].std():.2f}")
    log.info(f"  Status counts  :\n{df['Status'].value_counts().to_string()}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features from the raw DataFrame.

    New columns added
    -----------------
    Temporal
        order_year, order_month, order_dayofweek, order_quarter,
        order_weekofyear, is_weekend

    Customer
        tenure_days  — days between SignupDate and OrderDate (clipped ≥ 0)

    Price / discount
        effective_price  — UnitPrice × (1 − Discount/100)
        revenue          — effective_price × Quantity
        is_discounted    — 1 if Discount > 0, else 0

    Categorical (one-hot)
        Category, PaymentMethod, Status, CustomerSegment, City

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by load_data().

    Returns
    -------
    pd.DataFrame
        New DataFrame with all original + engineered columns.
    """
    log.info("Engineering features …")
    df = df.copy()

    # ── Temporal features ──────────────────────────────────────────────────────
    df["order_year"]       = df["OrderDate"].dt.year
    df["order_month"]      = df["OrderDate"].dt.month
    df["order_dayofweek"]  = df["OrderDate"].dt.dayofweek   # 0 = Monday … 6 = Sunday
    df["order_quarter"]    = df["OrderDate"].dt.quarter
    df["order_weekofyear"] = df["OrderDate"].dt.isocalendar().week.astype(int)
    df["is_weekend"]       = df["order_dayofweek"].isin([5, 6]).astype(int)

    # ── Customer tenure ────────────────────────────────────────────────────────
    df["tenure_days"] = (
        (df["OrderDate"] - df["SignupDate"]).dt.days.clip(lower=0)
    )

    # ── Price / discount features ──────────────────────────────────────────────
    df["effective_price"] = df["UnitPrice"] * (1.0 - df["Discount"] / 100.0)
    df["revenue"]         = df["effective_price"] * df["Quantity"]
    df["is_discounted"]   = (df["Discount"] > 0).astype(int)

    # ── One-hot encode categorical columns ────────────────────────────────────
    cat_cols = ["Category", "PaymentMethod", "Status", "CustomerSegment", "City"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    log.info(f"  Shape after encoding : {df.shape}")
    log.info(f"  Engineered features  : order_year, order_month, order_dayofweek, "
             f"order_quarter, order_weekofyear, is_weekend, tenure_days, "
             f"effective_price, revenue, is_discounted + {len(cat_cols)} one-hot groups")
    return df


def build_feature_matrix(df_eng: pd.DataFrame):
    """
    Split the engineered DataFrame into feature matrix X and target vector y.

    Drops all ID columns, raw date columns, and the regression target (Sales).

    Parameters
    ----------
    df_eng : pd.DataFrame
        Output of engineer_features().

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Regression target — Sales revenue ($).
    """
    cols_to_drop = [c for c in DROP_COLS if c in df_eng.columns]
    X = df_eng.drop(columns=cols_to_drop)
    y = df_eng["Sales"]
    log.info(f"  Feature matrix       : {X.shape[0]:,} rows × {X.shape[1]} features")
    log.info(f"  Target (Sales) stats : mean=${y.mean():.2f}  std=${y.std():.2f}  "
             f"min=${y.min():.2f}  max=${y.max():.2f}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_candidate_models() -> dict:
    """
    Return the three candidate regression models.

    GradientBoostingRegressor
        Best on tabular data; handles non-linearities well.
    RandomForestRegressor
        Strong bagging ensemble; good variance reduction.
    Ridge (Pipeline)
        Fast linear baseline; StandardScaler normalises inputs first.

    Returns
    -------
    dict[str, estimator]
        Mapping of model name → unfitted sklearn estimator.
    """
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ]),
    }


def get_tuning_grids() -> dict:
    """
    Hyperparameter search grids for GridSearchCV.

    Used only when --tune flag is passed.

    Returns
    -------
    dict[str, dict]
        Mapping of model name → parameter grid.
    """
    return {
        "GradientBoosting": {
            "n_estimators":    [200, 300, 400],
            "learning_rate":   [0.05, 0.08, 0.10],
            "max_depth":       [4, 5, 6],
            "subsample":       [0.7, 0.8, 1.0],
        },
        "RandomForest": {
            "n_estimators":      [100, 200, 300],
            "max_depth":         [8, 12, None],
            "min_samples_split": [2, 5, 10],
        },
        "Ridge": {
            "ridge__alpha": [0.1, 1.0, 10.0, 100.0],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_all(
    candidates: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> dict:
    """
    Run k-fold cross-validation on every candidate model and rank them.

    Scoring metric: R² (coefficient of determination).

    Parameters
    ----------
    candidates : dict
        {name: estimator} as returned by get_candidate_models().
    X_train, y_train
        Training split.
    cv : int
        Number of folds (default 5).

    Returns
    -------
    dict
        {model_name: {cv_r2_mean, cv_r2_std, cv_r2_min, cv_r2_max, cv_time_s}}
    """
    log.info(f"Running {cv}-fold cross-validation on {len(candidates)} models …")
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for name, model in candidates.items():
        t0     = time.time()
        scores = cross_val_score(model, X_train, y_train,
                                 cv=kf, scoring="r2", n_jobs=-1)
        elapsed = round(time.time() - t0, 2)
        results[name] = {
            "cv_r2_mean": float(scores.mean()),
            "cv_r2_std":  float(scores.std()),
            "cv_r2_min":  float(scores.min()),
            "cv_r2_max":  float(scores.max()),
            "cv_time_s":  elapsed,
        }
        log.info(
            f"  {name:<22s}  R² = {scores.mean():.4f} ± {scores.std():.4f}"
            f"  [min={scores.min():.4f}, max={scores.max():.4f}]  ({elapsed}s)"
        )

    # ── Rank by mean CV R² ────────────────────────────────────────────────────
    ranked = sorted(results.items(), key=lambda x: -x[1]["cv_r2_mean"])
    log.info("")
    log.info("  CV ranking:")
    for rank, (name, res) in enumerate(ranked, 1):
        log.info(f"    #{rank}  {name:<22s}  mean R² = {res['cv_r2_mean']:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HYPERPARAMETER TUNING (optional)
# ══════════════════════════════════════════════════════════════════════════════

def tune_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3,
):
    """
    Run GridSearchCV on the best model to find optimal hyperparameters.

    Parameters
    ----------
    name : str
        Model name key (must exist in get_tuning_grids()).
    model
        Unfitted estimator.
    X_train, y_train
        Training split.
    cv : int
        Number of folds for GridSearch (default 3 — faster than 5-fold).

    Returns
    -------
    best_estimator_
        Fitted estimator with the best found hyperparameters.
    """
    grids = get_tuning_grids()
    if name not in grids:
        log.warning(f"  No tuning grid defined for {name} — skipping tuning.")
        return model

    param_grid = grids[name]
    n_combos   = 1
    for v in param_grid.values():
        n_combos *= len(v)

    log.info(f"  Tuning {name} — {n_combos} parameter combinations × {cv} folds …")
    t0 = time.time()

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        model, param_grid,
        cv=kf, scoring="r2",
        n_jobs=-1, verbose=0, refit=True,
    )
    gs.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 1)

    log.info(f"  Best params  : {gs.best_params_}")
    log.info(f"  Best CV R²   : {gs.best_score_:.4f}  ({elapsed}s)")
    return gs.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Compute a full suite of regression metrics.

    Parameters
    ----------
    y_true : array-like   Actual Sales values.
    y_pred : array-like   Predicted Sales values.

    Returns
    -------
    dict with keys: r2, mae, rmse, mape
    """
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape}


def get_feature_importances(model, feature_cols: list) -> dict:
    """
    Extract and rank feature importances from tree-based models.

    For Pipeline models the inner estimator is accessed via model[-1].
    Ridge does not expose feature_importances_ — returns empty dict.

    Returns
    -------
    dict
        Top-20 {feature_name: importance_score} sorted descending.
    """
    base = model[-1] if hasattr(model, "__len__") else model
    if not hasattr(base, "feature_importances_"):
        return {}
    imp    = base.feature_importances_
    ranked = sorted(zip(feature_cols, imp.tolist()), key=lambda x: -x[1])
    return dict(ranked[:20])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RESIDUAL PLOT (optional, matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    """
    Show a residual scatter plot: predicted vs actual, and residual histogram.

    Silently skipped if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("  matplotlib not installed — skipping residual plot.")
        return

    residuals = np.array(y_true) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=13, fontweight="bold")

    # ── Predicted vs Actual ────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_pred, y_true, alpha=0.4, s=12, color="#3b82d4")
    lims = [min(y_pred.min(), float(y_true.min())),
            max(y_pred.max(), float(y_true.max()))]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Predicted Sales ($)")
    ax.set_ylabel("Actual Sales ($)")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=9)

    # ── Residual Histogram ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(residuals, bins=50, color="#3b82d4", edgecolor="white", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Residual (Actual − Predicted)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Residual Distribution  (mean={residuals.mean():.2f})")

    plt.tight_layout()
    plt.show()
    log.info("  Residual plot displayed.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRAINING REPORT
# ══════════════════════════════════════════════════════════════════════════════

def build_report(meta: dict) -> str:
    """
    Build a formatted human-readable training report string.

    Parameters
    ----------
    meta : dict
        The metadata dict assembled during the training run.

    Returns
    -------
    str
        Multi-line report suitable for printing and saving.
    """
    W = 64
    sep  = "=" * W
    thin = "─" * W

    lines = [
        sep,
        "  E-COMMERCE SALES PREDICTION — TRAINING REPORT",
        sep,
        f"  Model selected   : {meta['model_name']}",
        f"  Hypertuned       : {'Yes' if meta['tuned'] else 'No'}",
        f"  Training samples : {meta['n_train']:,}",
        f"  Test samples     : {meta['n_test']:,}",
        f"  Total features   : {meta['n_features']}",
        f"  Training time    : {meta['training_time_s']}s",
        "",
        thin,
        "  TEST-SET METRICS",
        thin,
        f"  R²   (coefficient of determination) : {meta['test_r2']:.4f}",
        f"  MAE  (mean absolute error)          : ${meta['test_mae']:.2f}",
        f"  RMSE (root mean squared error)      : ${meta['test_rmse']:.2f}",
        f"  MAPE (mean abs percentage error)    : {meta['test_mape']:.2f}%",
        "",
        thin,
        "  CROSS-VALIDATION RESULTS (5-fold R²)",
        thin,
    ]

    for name, res in meta["cv_results"].items():
        marker = "  ◀ SELECTED" if name == meta["model_name"] else ""
        lines.append(
            f"  {name:<22s}  {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}"
            f"  [{res['cv_r2_min']:.4f} – {res['cv_r2_max']:.4f}]"
            f"  {res['cv_time_s']}s{marker}"
        )

    fi = meta.get("feature_importances", {})
    if fi:
        lines += [
            "",
            thin,
            "  TOP-10 FEATURE IMPORTANCES",
            thin,
        ]
        for i, (feat, score) in enumerate(list(fi.items())[:10], 1):
            bar = "█" * int(score * 300)
            lines.append(f"  {i:2}. {feat:<32s}  {score:.4f}  {bar}")

    lines += ["", sep]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(data_path: str, tune: bool = False, show_plot: bool = True) -> dict:
    """
    Execute the full training pipeline end-to-end.

    Steps
    -----
    1.  Load & validate data
    2.  Engineer features
    3.  Build feature matrix (X, y)
    4.  Train / test split  (80/20, random_state=42)
    5.  5-fold cross-validate all candidate models
    6.  Pick the best model by mean CV R²
    7.  Optionally tune with GridSearchCV
    8.  Fit on full training split
    9.  Evaluate on held-out test split
    10. Extract feature importances
    11. Save model + metadata + report
    12. Optionally show residual plot

    Parameters
    ----------
    data_path : str
        Path to the CSV dataset.
    tune : bool
        If True, run GridSearchCV on the best model before final fit.
    show_plot : bool
        If True (and matplotlib is installed), show a residual plot.

    Returns
    -------
    dict
        Full metadata dict (identical to what is saved in model_meta.json).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    t_start = time.time()

    # ── 1. Load & validate ─────────────────────────────────────────────────────
    df_raw = load_data(data_path)

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    df_eng = engineer_features(df_raw)

    # ── 3. Feature matrix ──────────────────────────────────────────────────────
    X, y = build_feature_matrix(df_eng)

    feature_cols = list(X.columns)
    with open(COLS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info(f"  Feature columns saved → {COLS_PATH}")

    # ── 4. Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    log.info(f"  Train split  : {len(X_train):,} rows")
    log.info(f"  Test split   : {len(X_test):,}  rows")

    # ── 5. Cross-validate all candidates ──────────────────────────────────────
    candidates = get_candidate_models()
    cv_results = cross_validate_all(candidates, X_train, y_train, cv=5)

    # ── 6. Select best model ───────────────────────────────────────────────────
    best_name  = max(cv_results, key=lambda k: cv_results[k]["cv_r2_mean"])
    best_model = candidates[best_name]
    log.info(f"\nBest model   : {best_name}")
    log.info(f"Best CV R²   : {cv_results[best_name]['cv_r2_mean']:.4f}")

    # ── 7. Optional GridSearchCV tuning ───────────────────────────────────────
    if tune:
        log.info(f"\nHyperparameter tuning — {best_name} …")
        best_model = tune_model(best_name, best_model, X_train, y_train, cv=3)

    # ── 8. Final fit on full training split ────────────────────────────────────
    log.info("\nFitting best model on training split …")
    best_model.fit(X_train, y_train)

    # ── 9. Evaluate on held-out test split ─────────────────────────────────────
    y_pred  = best_model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    log.info(
        f"  R²={metrics['r2']:.4f}  "
        f"MAE=${metrics['mae']:.2f}  "
        f"RMSE=${metrics['rmse']:.2f}  "
        f"MAPE={metrics['mape']:.2f}%"
    )

    # ── 10. Feature importances ────────────────────────────────────────────────
    importances = get_feature_importances(best_model, feature_cols)
    if importances:
        log.info("  Top-5 features:")
        for feat, score in list(importances.items())[:5]:
            log.info(f"    {feat:<30s}  {score:.4f}")

    # ── 11. Save model + metadata ──────────────────────────────────────────────
    t_end = time.time()

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
        "n_train":             int(len(X_train)),
        "n_test":              int(len(X_test)),
        "training_time_s":     round(t_end - t_start, 1),
    }

    # model binary
    joblib.dump(best_model, MODEL_PATH)
    log.info(f"  Model saved    → {MODEL_PATH}")

    # JSON metadata
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Metadata saved → {META_PATH}")

    # text report
    report = build_report(meta)
    print("\n" + report)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    log.info(f"  Report saved   → {REPORT_PATH}")

    log.info(f"\n✓ Done in {meta['training_time_s']}s")

    # ── 12. Optional residual plot ─────────────────────────────────────────────
    if show_plot:
        plot_residuals(y_test, y_pred, best_name)

    return meta


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the E-Commerce Sales Prediction model.\n\n"
            "Examples:\n"
            "  python ecommerce_sales/train_model.py\n"
            "  python ecommerce_sales/train_model.py --tune\n"
            "  python ecommerce_sales/train_model.py --data custom.csv --tune --no-plot"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        metavar="CSV_PATH",
        help=f"Path to the input CSV file (default: {DATA_PATH})",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run GridSearchCV hyperparameter tuning on the best model (slower).",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Suppress the matplotlib residual plot (useful for headless environments).",
    )
    parser.set_defaults(plot=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(data_path=args.data, tune=args.tune, show_plot=args.plot)
