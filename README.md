# 🛒 E-Commerce Sales & Order Failure Prediction

> **Full-stack machine learning application** built on an Iranian e-commerce dataset.
> Predicts order-level sales revenue **and** classifies whether an order will fail
> (be Returned or Cancelled) — all served through a REST API and a Streamlit UI.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Dataset](#-dataset)
6. [Prerequisites](#-prerequisites)
7. [Installation](#-installation)
8. [Running the Project](#-running-the-project)
   - [Step 1 — Train the Sales Model](#step-1--train-the-sales-model)
   - [Step 2 — Train the Failure Model](#step-2--train-the-failure-model)
   - [Step 3 — Start the Backend API](#step-3--start-the-backend-api)
   - [Step 4 — Launch the Frontend UI](#step-4--launch-the-frontend-ui)
9. [API Reference](#-api-reference)
10. [Streamlit UI Pages](#-streamlit-ui-pages)
11. [ML Pipelines](#-ml-pipelines)
12. [Configuration](#-configuration)
13. [Troubleshooting](#-troubleshooting)
14. [Project Credits](#-project-credits)

---

## 🎯 Project Overview

This project solves two complementary machine learning problems on a real-world
Iranian e-commerce dataset:

| Problem | Type | Algorithm | Key Metric |
|---|---|---|---|
| Predict order sales revenue ($) | Regression | Gradient Boosting / Random Forest / Ridge | R² > 0.98 |
| Predict order failure (Returned/Cancelled) | Binary Classification | Gradient Boosting / Random Forest / Logistic Regression | F1 Score |

Both models are served through a **Flask REST API** and a **Streamlit** multi-page
dashboard that lets users explore data, make predictions, and inspect model performance.

---

## ✨ Features

### Machine Learning
- ✅ **Sales Revenue Prediction** — regression with 5-fold CV and optional GridSearchCV tuning
- ✅ **Order Failure Prediction** — binary classifier with stratified 5-fold CV
- ✅ **Automatic model selection** — best model picked by CV R² (regression) or CV F1 (classification)
- ✅ **Feature engineering** — date parts, customer tenure, price features, one-hot encoding
- ✅ **Feature importances** — top-20 extracted from tree-based models
- ✅ **Serialised artifacts** — `.joblib` model + `.json` metadata + `.txt` training report

### Backend (Flask)
- ✅ **15 REST endpoints** — predictions, model metrics, dataset analytics
- ✅ **CORS enabled** — Streamlit UI can call the API from any port
- ✅ **Input validation** — required fields, type checks, range checks on all POST endpoints
- ✅ **Lazy loading** — models and dataset loaded once on first request, cached in memory
- ✅ **Structured JSON errors** — every error returns `{"error": "<message>"}`

### Frontend (Streamlit)
- ✅ **7 pages** in a single `ui.py` — sidebar radio navigation
- ✅ **Live API status badge** — 🟢 Online / 🔴 Offline shown in every sidebar
- ✅ **Plotly interactive charts** — line, bar, donut, gauge, heatmap, area
- ✅ **Failure probability gauge** — Indicator chart with colour-coded risk zones
- ✅ **Confusion matrix heatmap** — for the failure classifier
- ✅ **Two-tab Model Metrics page** — sales regressor and failure classifier side-by-side

---

## 🛠 Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.9+ |
| Data | pandas, numpy | 2.x, 1.25+ |
| ML | scikit-learn, joblib | 1.4+, 1.3+ |
| Backend | Flask, flask-cors | 3.x, 4.x |
| Backend (async alt.) | FastAPI, uvicorn, pydantic | 0.110+, 0.29+, 2.x |
| Frontend | Streamlit | 1.35+ |
| Charts | Plotly | 5.20+ |
| HTTP client | requests | 2.31+ |

---

## 📁 Project Structure

```
ecommerce_sales/
│
├── data/
│   └── clean_final_data.csv               ← Iranian e-commerce orders (10,000 rows)
│
├── models/                                ← auto-created by training scripts
│   ├── sales_model.joblib                 ← serialised best sales regressor
│   ├── feature_columns.json               ← ordered feature list (sales inference)
│   ├── model_meta.json                    ← R², MAE, RMSE, MAPE, CV scores, importances
│   ├── training_report.txt                ← human-readable sales training summary
│   │
│   ├── failure_model.joblib               ← serialised best failure classifier
│   ├── failure_feature_columns.json       ← ordered feature list (failure inference)
│   ├── failure_model_meta.json            ← accuracy, F1, ROC-AUC, confusion matrix
│   └── failure_training_report.txt        ← human-readable failure training summary
│
├── backend/
│   ├── __init__.py
│   ├── data_loader.py                     ← CSV loading, feature engineering, get_stats()
│   ├── api.py                             ← FastAPI app (async, 15 endpoints)
│   ├── app.py                             ← Flask app  (sync,  15 endpoints)
│   └── train.py                           ← sales training module (importable)
│
├── frontend/
│   ├── __init__.py
│   ├── ui.py                              ← single-file Streamlit app (7 pages)
│   └── pages/                             ← multi-page Streamlit alternative
│       ├── 1_📊_Dashboard.py
│       ├── 2_🔮_Predict.py
│       ├── 3_📈_Insights.py
│       ├── 4_🤖_Model_Metrics.py
│       ├── 5_⚠️_Failure_Predict.py
│       └── 6_📉_Failure_Analytics.py
│
├── train_model.py                         ← standalone sales training CLI
├── train_failure.py                       ← standalone failure training CLI
├── requirements.txt                       ← all Python dependencies
└── README.md                              ← this file
```

---

## 📦 Dataset

**File:** `ecommerce_sales/data/clean_final_data.csv`

| Column | Type | Description |
|---|---|---|
| `OrderID` | int | Unique order identifier |
| `CustomerID` | int | Customer identifier |
| `OrderDate` | date | Date the order was placed |
| `ProductID` | int | Product identifier |
| `ProductName` | str | Product name |
| `Category` | str | `Electronics` / `Accessories` / `Home Office` / `Stationery` |
| `Quantity` | float | Units ordered |
| `UnitPrice` | float | Price per unit ($) |
| `Discount` | float | Discount percentage (0, 5, 10, 20, 30, 50) |
| `Sales` | float | ⭐ **Regression target** — actual revenue ($) |
| `Status` | str | ⭐ **Classification source** — `Completed` / `Returned` / `Cancelled` |
| `PaymentMethod` | str | `Gateway` / `Wallet` / `CardToCard` |
| `Age` | float | Customer age |
| `City` | str | Customer city (Tehran, Isfahan, Mashhad, …) |
| `SignupDate` | date | Customer registration date |
| `CustomerSegment` | str | `Regular` / `New` / `VIP` |

> **Failure label:** `order_failed = 1` when `Status ∈ {Returned, Cancelled}`, else `0`.

---

## ⚙️ Prerequisites

- **Python 3.9 or higher**
- **pip** package manager
- Two terminal windows (one for the API, one for the UI)

Verify your Python version:

```bash
python --version
# Python 3.x.x  (must be 3.9+)
```

---

## 💾 Installation

**Clone or navigate to the project directory, then run:**

```bash
pip install -r ecommerce_sales/requirements.txt
```

This installs all required libraries in one command:
`pandas`, `numpy`, `scikit-learn`, `joblib`, `flask`, `flask-cors`,
`fastapi`, `uvicorn`, `pydantic`, `streamlit`, `plotly`, `requests`.

---

## 🚀 Running the Project

> **Important:** Steps 1 and 2 (model training) must be completed **before** starting
> the backend API. The API returns `503` if a model file is missing.

---

### Step 1 — Train the Sales Model

```bash
python ecommerce_sales/train_model.py
```

**What happens:**
1. Loads `data/clean_final_data.csv` and validates required columns
2. Engineers 10 new features (date parts, tenure, price, flags)
3. One-hot encodes Category, PaymentMethod, Status, CustomerSegment, City
4. Splits 80% train / 20% test (random_state=42)
5. Runs 5-fold cross-validation on 3 models — GradientBoosting, RandomForest, Ridge
6. Selects the best model by mean CV R²
7. Fits on full training split, evaluates on test split
8. Saves 4 artifacts to `models/`

**Expected console output:**
```
10:00:01  INFO      Loading data from: .../clean_final_data.csv
10:00:01  INFO        Rows × Cols    : 10,000 × 17
10:00:02  INFO      Engineering features …
10:00:03  INFO      Running 5-fold cross-validation on 3 models …
10:00:03  INFO        GradientBoosting        R² = 0.9821 ± 0.0031  (12.4s)
10:00:04  INFO        RandomForest            R² = 0.9790 ± 0.0044   (6.1s)
10:00:05  INFO        Ridge                   R² = 0.8912 ± 0.0089   (0.3s)
10:00:05  INFO      Best model  : GradientBoosting
10:00:17  INFO        R²=0.9834  MAE=$2.41  RMSE=$4.87  MAPE=3.21%
10:00:17  INFO        Model saved → .../models/sales_model.joblib
```

**Optional flags:**

```bash
# Run GridSearchCV hyperparameter tuning (slower, more accurate)
python ecommerce_sales/train_model.py --tune

# Use a custom CSV file
python ecommerce_sales/train_model.py --data path/to/custom.csv

# Tune + no residual plot (for headless / CI environments)
python ecommerce_sales/train_model.py --tune --no-plot
```

**Output files (`models/`):**

| File | Contents |
|---|---|
| `sales_model.joblib` | Serialised best estimator |
| `feature_columns.json` | Ordered feature column list used during training |
| `model_meta.json` | All metrics, CV scores, top-20 feature importances |
| `training_report.txt` | Human-readable training summary |

---

### Step 2 — Train the Failure Model

```bash
python ecommerce_sales/train_failure.py
```

**What happens:**
1. Loads the same CSV dataset
2. Creates binary label: `order_failed = 1` if `Status ≠ Completed`
3. Engineers features (same as sales model, **excluding Status** to avoid leakage)
4. Adds `order_value` and `high_value_order` features
5. Splits 80% / 20% with **stratified sampling** (preserves failure rate in both splits)
6. Runs **stratified** 5-fold CV on 3 classifiers — GradientBoosting, RandomForest, LogisticRegression
7. Selects best model by mean CV **F1 score**
8. Evaluates with accuracy, precision, recall, F1, ROC-AUC, confusion matrix
9. Saves 4 artifacts to `models/`

**Expected console output:**
```
10:00:01  INFO      Loading data from: .../clean_final_data.csv
10:00:01  INFO        Failure rate: 28.5%
10:00:02  INFO      Engineering features …
10:00:03  INFO      Running 5-fold stratified CV on 3 models …
10:00:04  INFO        GradientBoosting        F1 = 0.8712 ± 0.0082  (15.2s)
10:00:05  INFO        RandomForest            F1 = 0.8640 ± 0.0091   (9.3s)
10:00:06  INFO        LogisticRegression      F1 = 0.7821 ± 0.0104   (1.1s)
10:00:06  INFO      Best model: GradientBoosting
```

**Optional flags:**

```bash
# Run GridSearchCV hyperparameter tuning
python ecommerce_sales/train_failure.py --tune

# Use a custom CSV file
python ecommerce_sales/train_failure.py --data path/to/custom.csv
```

**Output files (`models/`):**

| File | Contents |
|---|---|
| `failure_model.joblib` | Serialised best classifier |
| `failure_feature_columns.json` | Ordered feature column list |
| `failure_model_meta.json` | Accuracy, F1, ROC-AUC, confusion matrix, CV scores, importances |
| `failure_training_report.txt` | Human-readable training summary |

---

### Step 3 — Start the Backend API

> Open **Terminal 1** and keep it running while you use the UI.

**Option A — Flask (recommended, simpler):**

```bash
python ecommerce_sales/backend/app.py
```

API starts at: **http://localhost:5000**

**Option B — Flask via CLI:**

```bash
flask --app ecommerce_sales/backend/app.py run --port 5000 --debug
```

**Option C — FastAPI (async alternative):**

```bash
uvicorn ecommerce_sales.backend.api:app --reload --port 8000
```

FastAPI starts at: **http://localhost:8000**
Auto-generated docs: **http://localhost:8000/docs**

> **Note:** The Streamlit `ui.py` connects to `http://localhost:5000` (Flask) by default.
> To use FastAPI, edit `API_BASE = "http://localhost:8000"` at the top of `frontend/ui.py`.

**Verify the API is running:**

```bash
curl http://localhost:5000/health
# {"service":"E-Commerce Sales & Order Failure Prediction API","status":"ok"}
```

**Registered routes on startup (Flask):**
```
GET   /health
POST  /predict
GET   /model/metrics
POST  /predict-failure
GET   /failure/metrics
GET   /failure/analytics
GET   /data/stats
GET   /data/monthly-revenue
GET   /data/category-sales
GET   /data/top-products
GET   /data/city-sales
GET   /data/payment-distribution
GET   /data/segment-revenue
GET   /data/status-breakdown
GET   /data/discount-analysis
```

---

### Step 4 — Launch the Frontend UI

> Open **Terminal 2**.

**Option A — Single-file UI (recommended):**

```bash
streamlit run ecommerce_sales/frontend/ui.py
```

**Option B — Multi-page app (auto-discovers `pages/` folder):**

```bash
streamlit run ecommerce_sales/frontend/app.py
```

The app opens automatically at: **http://localhost:8501**

---

## 🌐 API Reference

Base URL: `http://localhost:5000`

All responses are JSON. Errors return `{"error": "<message>"}`.

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — confirms API is running |

---

### Sales Prediction

#### `POST /predict`

Predict the sales revenue for a new order.

**Request body:**

```json
{
  "category":         "Electronics",
  "unit_price":       18.0,
  "quantity":         2,
  "discount":         5.0,
  "payment_method":   "Gateway",
  "status":           "Completed",
  "customer_segment": "Regular",
  "city":             "Tehran",
  "customer_age":     35,
  "order_month":      6,
  "order_year":       2025,
  "order_dayofweek":  1,
  "order_quarter":    2,
  "order_weekofyear": 24,
  "tenure_days":      365
}
```

**Required fields:** `category`, `unit_price`, `quantity`, `city`, `payment_method`

**Response `200`:**

```json
{
  "predicted_sales":   34.20,
  "effective_price":   17.10,
  "gross_order_value": 34.20,
  "model_used":        "GradientBoosting",
  "note":              "Predicted value is estimated post-discount revenue."
}
```

---

#### `GET /model/metrics`

Returns the sales model's evaluation metrics, CV scores, and top-20 feature importances.

```json
{
  "model_name":          "GradientBoosting",
  "tuned":               false,
  "test_r2":             0.9834,
  "test_mae":            2.41,
  "test_rmse":           4.87,
  "test_mape":           3.21,
  "n_train":             8000,
  "n_test":              2000,
  "n_features":          38,
  "training_time_s":     18.4,
  "cv_results":          { "GradientBoosting": { ... }, ... },
  "feature_importances": { "revenue": 0.412, "effective_price": 0.381, ... }
}
```

---

### Order Failure Prediction

#### `POST /predict-failure`

Predict whether a new order will be Returned or Cancelled.

> ⚠️ Do NOT include a `status` field — it is the prediction target.

**Request body:**

```json
{
  "category":         "Electronics",
  "unit_price":       62.0,
  "quantity":         3,
  "discount":         20.0,
  "payment_method":   "CardToCard",
  "customer_segment": "New",
  "city":             "Tehran",
  "customer_age":     24,
  "order_month":      11,
  "order_year":       2025,
  "order_dayofweek":  1,
  "order_quarter":    4,
  "order_weekofyear": 45,
  "tenure_days":      30
}
```

**Response `200`:**

```json
{
  "order_failed":        1,
  "failure_probability": 0.7241,
  "risk_level":          "High",
  "model_used":          "GradientBoosting",
  "note":                "1 = likely Returned/Cancelled; 0 = likely Completed."
}
```

**Risk level thresholds:**

| Risk | Probability range |
|---|---|
| 🟢 Low | < 30% |
| 🟡 Medium | 30% – 60% |
| 🔴 High | > 60% |

---

#### `GET /failure/metrics`

Returns the failure classifier's classification metrics, confusion matrix, and CV scores.

```json
{
  "model_name":    "GradientBoosting",
  "test_accuracy": 0.8721,
  "test_precision": 0.8543,
  "test_recall":   0.8214,
  "test_f1":       0.8375,
  "test_roc_auc":  0.9102,
  "confusion_matrix": [[TN, FP], [FN, TP]],
  "n_train": 8000,
  "n_test":  2000,
  ...
}
```

---

#### `GET /failure/analytics`

Returns failure rate breakdowns across all key dimensions.

```json
{
  "overall_failure_rate": 28.5,
  "by_category":  [ { "Category": "Electronics", "total": 2500, "failures": 700, "failure_rate": 28.0 }, ... ],
  "by_city":      [ ... ],
  "by_payment":   [ ... ],
  "by_segment":   [ ... ],
  "by_discount":  [ ... ],
  "monthly_failures": [ { "month": "2024-01", "total": 420, "failures": 120, "failure_rate": 28.57 }, ... ]
}
```

---

### Dataset Analytics

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/data/stats` | KPI summary — revenue, orders, avg value, top category/city |
| `GET` | `/data/monthly-revenue` | Monthly revenue time series |
| `GET` | `/data/category-sales` | Revenue per product category |
| `GET` | `/data/top-products?n=10` | Top N products by total revenue |
| `GET` | `/data/city-sales` | Revenue per city |
| `GET` | `/data/payment-distribution` | Order count per payment method |
| `GET` | `/data/segment-revenue` | Revenue per customer segment |
| `GET` | `/data/status-breakdown` | Order count + revenue per status |
| `GET` | `/data/discount-analysis` | Avg sales grouped by discount level |

---

## 🖥️ Streamlit UI Pages

Run: `streamlit run ecommerce_sales/frontend/ui.py`

| # | Page | What you'll see |
|---|---|---|
| 1 | 🏠 **Home** | 2-column model overview · live dataset KPIs · quick-start code block · navigation guide |
| 2 | 📊 **Dashboard** | 5 KPI metric cards · monthly revenue line chart · category bar + donut pie · order status bars |
| 3 | 🔮 **Predict Sales** | Order details form → predicted revenue result box + 4 metric cards + breakdown bar chart |
| 4 | ⚠️ **Failure Predict** | Order details form → failure probability gauge · 🟢/🟡/🔴 risk badge · order summary |
| 5 | 📈 **Insights** | Top-10 products · city revenue · payment pie · segment bars · discount avg bars |
| 6 | 📉 **Failure Analytics** | Failure rates by category, city, payment, segment, discount, monthly trend · confusion matrix |
| 7 | 🤖 **Model Metrics** | Two tabs: Sales (R²/MAE/RMSE/MAPE + CV bar + feature importances) · Failure (F1/ROC-AUC + confusion matrix + CV F1 + importances) |

---

## 🧠 ML Pipelines

### Sales Revenue — Regression

```
clean_final_data.csv
        │
        ▼  Feature Engineering
        ├── Date features      : order_year, order_month, order_dayofweek,
        │                        order_quarter, order_weekofyear, is_weekend
        ├── Customer tenure    : days between SignupDate and OrderDate (clipped ≥ 0)
        ├── Price features     : effective_price, revenue, is_discounted
        └── One-hot encoding   : Category, PaymentMethod, Status, CustomerSegment, City
        │
        ▼  80/20 train/test split (random_state=42)
        │
        ▼  5-fold KFold cross-validation (scoring = R²)
        ├── GradientBoostingRegressor  (n_estimators=300, lr=0.08, max_depth=5)
        ├── RandomForestRegressor      (n_estimators=200, max_depth=12)
        └── Ridge                      (StandardScaler → Ridge α=1.0)
        │
        ▼  Best model (highest mean CV R²)
        │
        ▼  Optional GridSearchCV tuning  (--tune flag)
        │
        ▼  Final fit on training split
        │
        ▼  Evaluate on test split: R², MAE, RMSE, MAPE
        │
        ▼  Save artifacts → models/
```

---

### Order Failure — Classification

```
clean_final_data.csv
        │
        ▼  Label creation
        │   order_failed = 1  if Status ∈ {Returned, Cancelled}
        │   order_failed = 0  if Status = Completed
        │
        ▼  Feature Engineering (same as above, EXCLUDING Status)
        │   + order_value       : effective_price × quantity
        │   + high_value_order  : 1 if order_value > median (~$50)
        │
        ▼  80/20 stratified train/test split (preserves failure rate)
        │
        ▼  Stratified 5-fold cross-validation (scoring = F1)
        ├── GradientBoostingClassifier   (n_estimators=300, lr=0.08, max_depth=4)
        ├── RandomForestClassifier       (n_estimators=200, class_weight="balanced")
        └── LogisticRegression           (StandardScaler → LR, class_weight="balanced")
        │
        ▼  Best model (highest mean CV F1)
        │
        ▼  Optional GridSearchCV tuning  (--tune flag)
        │
        ▼  Final fit on training split
        │
        ▼  Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
        │
        ▼  Save artifacts → models/
```

---

## ⚙️ Configuration

### Change the API port

**Flask (default 5000):**
```bash
flask --app ecommerce_sales/backend/app.py run --port 5001
```
Then update `API_BASE` in `frontend/ui.py`:
```python
API_BASE = "http://localhost:5001"
```

**FastAPI (default 8000):**
```bash
uvicorn ecommerce_sales.backend.api:app --reload --port 8001
```

### Change the Streamlit port

```bash
streamlit run ecommerce_sales/frontend/ui.py --server.port 8502
```

### Use a custom dataset

```bash
python ecommerce_sales/train_model.py   --data path/to/your_data.csv
python ecommerce_sales/train_failure.py --data path/to/your_data.csv
```

> The CSV must contain all required columns listed in the [Dataset](#-dataset) section.

---

## 🔧 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: flask` | Dependencies not installed | `pip install -r ecommerce_sales/requirements.txt` |
| `ModuleNotFoundError: streamlit` | Streamlit not installed | `pip install streamlit` |
| API returns `503 Sales model not found` | Sales model not trained | `python ecommerce_sales/train_model.py` |
| API returns `503 Failure model not found` | Failure model not trained | `python ecommerce_sales/train_failure.py` |
| Streamlit shows `🔴 API Offline` | Flask backend not running | `python ecommerce_sales/backend/app.py` |
| `Address already in use` on port 5000 | Port occupied | Use `--port 5001` and update `API_BASE` in `ui.py` |
| `Address already in use` on port 8501 | Port occupied | `streamlit run ... --server.port 8502` |
| `ValueError: Missing required columns` | CSV missing columns | Check column names match the [Dataset](#-dataset) table |
| Charts show no data | API returned empty response | Check Flask terminal for errors; verify CSV is in `data/` |
| `predict_proba` not available | Ridge model selected | Ridge doesn't support probabilities; failure probability defaults to 0.5 |
| Slow training | Large dataset / many estimators | Use `--no-plot` to skip matplotlib; reduce `n_estimators` in `train_model.py` |

---

## 📋 Full Dependency List

```
# Core data & ML
pandas>=2.0.0          # DataFrame operations, CSV loading, time-series resampling
numpy>=1.25.0          # Numerical arrays, math operations
scikit-learn>=1.4.0    # All ML models, CV, metrics, Pipeline, StandardScaler
joblib>=1.3.0          # Model serialisation (dump/load)

# Backend — Flask (sync)
flask>=3.0.0           # REST API framework
flask-cors>=4.0.0      # CORS headers for browser/Streamlit clients

# Backend — FastAPI (async alternative)
fastapi>=0.110.0       # Async REST framework with auto Swagger docs
uvicorn[standard]>=0.29.0  # ASGI server for FastAPI
pydantic>=2.0.0        # Schema validation for FastAPI models
python-multipart>=0.0.9    # FastAPI form parsing

# Frontend
streamlit>=1.35.0      # Multi-page UI framework
plotly>=5.20.0         # Interactive charts (line, bar, pie, gauge, heatmap)
requests>=2.31.0       # HTTP calls from Streamlit to Flask/FastAPI
```

Install all at once:

```bash
pip install -r ecommerce_sales/requirements.txt
```

---

## 👤 Project Credits

- **Dataset:** Iranian e-commerce order data
- **Author:** Avinash Pandey
- **Stack:** Python · scikit-learn · Flask · FastAPI · Streamlit · Plotly
