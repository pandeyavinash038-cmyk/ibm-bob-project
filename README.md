# 🛒 E-Commerce Sales Prediction

portA full-stack machine learning application that predicts **order-level sales revenue**
for an Iranian e-commerce dataset, with a Flask REST backend and a Streamlit UI frontend.

---

## 📁 Project Structure

```
ecommerce_sales/
│
├── data/
│   └── clean_final_data.csv          ← raw dataset (orders, products, customers)
│
├── models/                           ← auto-created after training
│   ├── sales_model.joblib            ← serialised best estimator
│   ├── feature_columns.json          ← ordered feature list for inference
│   ├── model_meta.json               ← metrics, CV scores, importances
│   └── training_report.txt           ← human-readable training summary
│
├── backend/
│   ├── __init__.py
│   ├── data_loader.py                ← CSV loading, feature engineering, get_stats()
│   └── app.py                        ← Flask REST API (12 endpoints)
│
├── frontend/
│   ├── __init__.py
│   └── ui.py                         ← Single-file Streamlit app (5 pages)
│
├── train_model.py                    ← Standalone ML training script
├── requirements.txt                  ← All Python dependencies
└── README.md
```

---

## ⚙️ Prerequisites

- Python **3.9 or higher**
- `pip` package manager

---

## 🚀 Quick Start (4 Steps)

### Step 1 — Install dependencies

```bash
pip install -r ecommerce_sales/requirements.txt
```

### Step 2 — Train the model

```bash
python ecommerce_sales/train_model.py
```

This will:
- Load and engineer features from `data/clean_final_data.csv`
- Run **5-fold cross-validation** on 3 candidate models
- Pick the best model (usually Gradient Boosting)
- Save the model and metrics to `ecommerce_sales/models/`
- Print a full training report to the console

Optional flags:
```bash
# Also run GridSearchCV hyperparameter tuning (slower but more accurate)
python ecommerce_sales/train_model.py --tune

# Point to a custom CSV file
python ecommerce_sales/train_model.py --data path/to/your_data.csv
```

Expected output:
```
09:00:01  INFO      Loading data from: .../clean_final_data.csv
09:00:01  INFO        Loaded 10,000 rows × 17 columns
09:00:02  INFO      Engineering features ...
09:00:03  INFO      Running 5-fold cross-validation on 3 models ...
09:00:03  INFO        GradientBoosting    R² = 0.9821 ± 0.0031  (12.4s)
09:00:04  INFO        RandomForest        R² = 0.9790 ± 0.0044  (6.1s)
09:00:04  INFO        Ridge               R² = 0.8912 ± 0.0089  (0.3s)
09:00:04  INFO      Best model: GradientBoosting
09:00:16  INFO        R²=0.9834  MAE=$2.41  RMSE=$4.87  MAPE=3.21%
09:00:16  INFO        Model saved → .../models/sales_model.joblib
```

### Step 3 — Start the Flask backend

Open **Terminal 1** and run:

```bash
python ecommerce_sales/backend/app.py
```

The API starts at: **http://localhost:5000**

Verify it's running:
```bash
curl http://localhost:5000/health
# {"status": "ok", "service": "E-Commerce Sales Prediction API"}
```

### Step 4 — Launch the Streamlit UI

Open **Terminal 2** and run:

```bash
streamlit run ecommerce_sales/frontend/ui.py
```

The app opens at: **http://localhost:8501**

---

## 📦 Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `OrderID` | int | Unique order identifier |
| `CustomerID` | int | Customer identifier |
| `OrderDate` | date | Date the order was placed |
| `ProductID` | int | Product identifier |
| `Quantity` | float | Units ordered |
| `Discount` | float | Discount percentage applied (0–50%) |
| `PaymentMethod` | str | `Gateway` / `Wallet` / `CardToCard` |
| `Status` | str | `Completed` / `Returned` / `Cancelled` |
| `Age` | float | Customer age |
| `City` | str | Customer city (Tehran, Isfahan, …) |
| `SignupDate` | date | Date customer registered |
| `CustomerSegment` | str | `Regular` / `New` / `VIP` |
| `ProductName` | str | Product name |
| `Category` | str | `Electronics` / `Accessories` / `Home Office` / `Stationery` |
| `UnitPrice` | float | Price per unit ($) |
| `Sales` | float | **Target** — actual sales revenue ($) |
| `OrderValue` | float | Duplicate of Sales (dropped during training) |

---

## 🧠 ML Pipeline

```
clean_final_data.csv
        │
        ▼  Feature Engineering (data_loader.py)
        ├── Date parts      : order_year, order_month, order_dayofweek,
        │                     order_quarter, order_weekofyear
        ├── Customer tenure : days between SignupDate and OrderDate
        ├── Price features  : effective_price, revenue, is_discounted
        ├── Flag features   : is_weekend
        └── One-hot encoding: Category, PaymentMethod, Status,
                              CustomerSegment, City
        │
        ▼  Model Comparison (train_model.py) — 5-fold CV
        ├── GradientBoostingRegressor  (n_estimators=300, lr=0.08)
        ├── RandomForestRegressor      (n_estimators=200, max_depth=12)
        └── Ridge                      (Pipeline: StandardScaler + Ridge)
        │
        ▼  Best model selected by highest mean CV R²
        │
        ▼  Saved to models/
           ├── sales_model.joblib
           ├── feature_columns.json
           ├── model_meta.json
           └── training_report.txt
```

---

## 🌐 Flask API Endpoints

Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Predict sales for a new order |
| `GET` | `/model/metrics` | R², MAE, RMSE, MAPE, CV scores, feature importances |
| `GET` | `/data/stats` | KPI summary (revenue, orders, top category/city) |
| `GET` | `/data/monthly-revenue` | Monthly revenue time series |
| `GET` | `/data/category-sales` | Revenue per product category |
| `GET` | `/data/top-products?n=10` | Top N products by total revenue |
| `GET` | `/data/city-sales` | Revenue per city |
| `GET` | `/data/payment-distribution` | Order count per payment method |
| `GET` | `/data/segment-revenue` | Revenue per customer segment |
| `GET` | `/data/status-breakdown` | Order count + revenue per status |
| `GET` | `/data/discount-analysis` | Avg sales grouped by discount level |

### `/predict` — Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "category":          "Electronics",
    "unit_price":        18.0,
    "quantity":          2,
    "discount":          5.0,
    "payment_method":    "Gateway",
    "status":            "Completed",
    "customer_segment":  "Regular",
    "city":              "Tehran",
    "customer_age":      35,
    "order_month":       6,
    "order_year":        2025,
    "order_dayofweek":   1,
    "order_quarter":     2,
    "order_weekofyear":  24,
    "tenure_days":       365
  }'
```

### `/predict` — Example Response

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

## 🖥️ Streamlit UI Pages

All pages are in a single file [`frontend/ui.py`](frontend/ui.py), navigated via the sidebar.

| Page | What you'll see |
|------|----------------|
| 🏠 **Home** | Project overview, quick-start guide, live dataset snapshot |
| 📊 **Dashboard** | 5 KPI cards · Monthly revenue trend line · Category bar + donut · Order status charts |
| 🔮 **Predict Sales** | Order form (product, customer, date) → predicted sales card + breakdown bar |
| 📈 **Insights** | Top-10 products · City revenue · Payment method pie · Segment comparison · Discount analysis |
| 🤖 **Model Metrics** | R²/MAE/RMSE/MAPE · CV comparison bar · Top feature importances |

The sidebar shows a live **🟢 API Online / 🔴 API Offline** status indicator.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: flask` | Run `pip install -r ecommerce_sales/requirements.txt` |
| `503 Model not found` | Run `python ecommerce_sales/train_model.py` first |
| `❌ API Offline` badge in UI | Start Flask: `python ecommerce_sales/backend/app.py` |
| `Port 5000 already in use` | Change port: `flask --app ecommerce_sales/backend/app.py run --port 5001` and update `API_BASE` in `ui.py` |
| Streamlit not found | Run `pip install streamlit` |

---

## 📋 Requirements Summary

```
pandas        — data loading & feature engineering
numpy         — numerical operations
scikit-learn  — ML models, cross-validation, metrics
joblib        — model serialisation
flask         — REST API backend
flask-cors    — cross-origin requests from browser
streamlit     — frontend UI framework
plotly        — interactive charts
requests      — HTTP calls from Streamlit to Flask
```

Install all at once:
```bash
pip install -r ecommerce_sales/requirements.txt
```
