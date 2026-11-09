"""
src/config.py
=============
Central configuration for RTO Shield AI - Model 1.

All paths, constants, and feature lists live here so every other
script imports them from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (one level up from this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_CSV        = PROJECT_ROOT / "data" / "raw"  / "DataCoSupplyChainDataset.csv"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
OUTPUTS_DIR    = PROJECT_ROOT / "outputs"

# Ensure output directories exist at import time
for _d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Chronological split fractions (must sum to 1.0)
# Applied at ORDER level (by Order Id) to prevent cross-order leakage.
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70   # earliest 70 % of orders → training set
VAL_FRAC   = 0.15   # next    15 %             → validation set (tuning & threshold)
TEST_FRAC  = 0.15   # last    15 %             → test set (final reporting only)

assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9, "Split fractions must sum to 1."

# ---------------------------------------------------------------------------
# Target & proxy columns
# ---------------------------------------------------------------------------
TARGET_COL    = "Late_delivery_risk"   # 1 = late / at-risk, 0 = on-time
DATE_COL      = "order_date"           # column name in orders.parquet
DATE_COL_RAW  = "order date (DateOrders)"  # column name in raw CSV
ORDER_ID_COL  = "Order Id"
PAYMENT_COL   = "Type"                 # COD proxy: Type == "CASH"
COD_VALUE     = "CASH"

# ---------------------------------------------------------------------------
# Columns that must NEVER be used as model features
# (post-dispatch outcomes or customer PII)
# ---------------------------------------------------------------------------
FORBIDDEN_FEATURES = [
    # Post-dispatch outcome leakage
    "Days for shipping (real)",
    "Days_for_shipping_real",
    "Delivery Status",
    "Order Status",
    "shipping date (DateOrders)",
    "shipping_date",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
    # Evaluation-only secondary target — must never be a model input
    "aux_target_cancel_fraud",
    # Customer PII
    "Customer Fname",
    "Customer Lname",
    "Customer Email",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    # Other non-feature columns
    "Product Image",
    "Product Description",
    # History gating column (dropped from parquet, listed for safety)
    "outcome_known_date",
]

# ---------------------------------------------------------------------------
# Feature lists — ORDER-LEVEL columns from data/processed/orders.parquet
#
# Rules:
#   • Known BEFORE dispatch (no actual shipping outcome)
#   • Not in FORBIDDEN_FEATURES
#   • Not an identifier (Order Id, Customer Id) or date (order_date)
#   • Not a target (Late_delivery_risk) or eval-only target (aux_target_cancel_fraud)
#   • NaN-heavy cols (customer_prev_cash_share, days_since_prev_order) kept;
#     imputed downstream (median / 0 fill in features.py).
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES: list[str] = [
    # Payment / order type
    "Type",
    # Logistics
    "Shipping_Mode",
    "Market",
    "Order_Region",
    "Order_Country",
    # Customer
    "Customer_Segment",
    "Department_Name",
    # Product catalogue
    "main_item_category",
]

NUMERIC_FEATURES: list[str] = [
    # Scheduled shipping window (pre-dispatch, known at order time)
    "Days_for_shipment_scheduled",
    # Order value / quantity / discount
    "order_value",
    "n_items",
    "avg_discount_rate",
    "value_per_item",
    # Binary / flag features (treated as numeric for scaling)
    "is_cash",
    "is_weekend",
    "is_night_order",
    "high_discount_flag",
    "n_distinct_categories",
    # Temporal
    "order_month",
    "order_dow",
    "order_hour",
    # Leakage-safe history features
    "customer_prev_orders",
    "customer_prev_late_rate",
    "country_prev_orders",
    "country_prev_late_rate",
    "customer_prev_cash_share",   # NaN → imputed with 0 (no prior cash history)
    "days_since_prev_order",      # NaN → imputed with median
]

# ---------------------------------------------------------------------------
# Baseline model feature subset (shipping mode + scheduled days only)
# Used by the Baseline Logistic Regression in 02_train.py
# ---------------------------------------------------------------------------
BASELINE_CAT_FEATURES: list[str] = ["Shipping_Mode"]
BASELINE_NUM_FEATURES: list[str] = ["Days_for_shipment_scheduled"]
