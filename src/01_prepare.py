"""
src/01_prepare.py
=================
Step 2 – Build the ORDER-LEVEL feature table for RTO Shield AI - Model 1.

Collapses the raw line-item CSV into one row per Order Id, engineers
leakage-safe history features, and writes data/processed/orders.parquet.

Usage:
    python src/01_prepare.py

Outputs:
    data/processed/orders.parquet   – the order-level feature table
    (stdout)                        – summary statistics and QA checks

LEAKAGE CONTRACT
----------------
- outcome_known_date  : earliest timestamp at which the delivery outcome is
                        observable in the data.  Used ONLY for history-window
                        gating; it is DROPPED before saving the parquet.
- aux_target_cancel_fraud : secondary EVALUATION-ONLY target.  It is kept in
                        the parquet for downstream evaluation notebooks but
                        MUST NOT appear in any model feature list.  The column
                        is documented here and enforced in the leakage test.
- FORBIDDEN_FEATURES  : as defined in config.py.  None of these appear in the
                        final feature set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable when called from the project root
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    RAW_CSV,
    PROCESSED_DIR,
    TRAIN_FRAC,
    TARGET_COL,
    FORBIDDEN_FEATURES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"

# Smoothing hyper-parameter for country-level Bayesian rate estimate
COUNTRY_K = 20          # strength of prior (number of pseudo-observations)

# Column name aliases used throughout
COL_ORDER_DATE    = "order date (DateOrders)"
COL_SHIP_DATE     = "shipping date (DateOrders)"
COL_DAYS_REAL     = "Days for shipping (real)"
COL_DAYS_SCHED    = "Days for shipment (scheduled)"
COL_ORDER_ID      = "Order Id"
COL_CUSTOMER_ID   = "Customer Id"
COL_ORDER_COUNTRY = "Order Country"
COL_LATE          = "Late_delivery_risk"
COL_ORDER_STATUS  = "Order Status"
COL_TYPE          = "Type"
COL_CATEGORY      = "Category Name"
COL_ITEM_PRICE    = "Order Item Product Price"
COL_ITEM_TOTAL    = "Order Item Total"
COL_ITEM_QTY      = "Order Item Quantity"
COL_ITEM_DISC     = "Order Item Discount Rate"

# Secondary evaluation target  ← NEVER use as a model feature
AUX_TARGET_COL = "aux_target_cancel_fraud"


# ===========================================================================
# 1. Load raw data
# ===========================================================================

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


# ===========================================================================
# 2. Parse dates
# ===========================================================================

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[COL_ORDER_DATE] = pd.to_datetime(df[COL_ORDER_DATE], errors="coerce")
    df[COL_SHIP_DATE]  = pd.to_datetime(df[COL_SHIP_DATE],  errors="coerce")
    return df


# ===========================================================================
# 3. Date-rule check: does (ship_date - order_date).days == real_days?
# ===========================================================================

def check_date_rule(df: pd.DataFrame) -> str:
    """
    Empirically verify whether shipping_date = order_date + real_shipping_days.

    Returns 'shipping_date' or 'order_date + real_days' — the rule chosen.
    Prints a detailed explanation.
    """
    diff_days = (df[COL_SHIP_DATE] - df[COL_ORDER_DATE]).dt.days
    pct_match = (diff_days == df[COL_DAYS_REAL]).mean() * 100

    print(f"\n[date-rule check]")
    print(f"  (shipping_date - order_date).days == Days_for_shipping_(real)")
    print(f"  Match rate: {pct_match:.2f}% of rows")

    if pct_match > 90.0:
        rule = "shipping_date"
        print(
            f"  [OK] {pct_match:.1f}% > 90% → shipping_date already encodes when the\n"
            f"     outcome became observable.  Using outcome_known_date = shipping_date.\n"
            f"     (This date marks the moment the carrier delivered/failed delivery,\n"
            f"      so it is the earliest we could label an order as late or on-time.)"
        )
    else:
        rule = "order_date_plus_real"
        print(
            f"  [WARN] {pct_match:.1f}% ≤ 90% → shipping_date does not reliably equal\n"
            f"     order_date + real_days.  Using outcome_known_date = order_date + real_days\n"
            f"     to stay conservative (outcome visible only after actual transit)."
        )

    return rule


# ===========================================================================
# 4. Collapse to order level
# ===========================================================================

def build_order_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per Order Id.

    Aggregation rules:
    - first()  for scalar order-level fields (all items share the same value)
    - sum()    for monetary / quantity columns (true order totals)
    - mean()   for discount rate (average across line items)
    - Main category = Category Name of the highest-priced item (argmax price)
    - n_distinct_categories = nunique(Category Name)
    - aux_target_cancel_fraud = any item has Order Status in {CANCELED, SUSPECTED_FRAUD}
    """

    # -- auxiliary secondary target (computed before groupby) ----------------
    cancel_fraud_statuses = {"CANCELED", "SUSPECTED_FRAUD"}
    df = df.copy()
    df["_cancel_fraud"] = df[COL_ORDER_STATUS].isin(cancel_fraud_statuses).astype(int)

    # -- main item category: Category Name of the most expensive line item ---
    df["_max_price_rank"] = df.groupby(COL_ORDER_ID)[COL_ITEM_PRICE].rank(
        method="first", ascending=False
    )

    # For each order keep only the top-priced row to extract its category name
    top_item = (
        df[df["_max_price_rank"] == 1]
        .groupby(COL_ORDER_ID)[COL_CATEGORY]
        .first()
        .rename("main_item_category")
    )

    # -- n_distinct_categories per order -------------------------------------
    n_cats = (
        df.groupby(COL_ORDER_ID)[COL_CATEGORY]
        .nunique()
        .rename("n_distinct_categories")
    )

    # -- core aggregation ----------------------------------------------------
    agg = df.groupby(COL_ORDER_ID, sort=False).agg(
        # Payment / type
        Type                        =(COL_TYPE,         "first"),
        Shipping_Mode               =("Shipping Mode",  "first"),
        Days_for_shipment_scheduled =(COL_DAYS_SCHED,   "first"),
        Market                      =("Market",         "first"),
        Order_Region                =("Order Region",   "first"),
        Department_Name             =("Department Name","first"),
        Customer_Segment            =("Customer Segment","first"),
        Customer_Id                 =(COL_CUSTOMER_ID,  "first"),
        Order_Country               =(COL_ORDER_COUNTRY,"first"),
        Late_delivery_risk          =(COL_LATE,         "first"),
        order_date                  =(COL_ORDER_DATE,   "first"),
        shipping_date               =(COL_SHIP_DATE,    "first"),
        Days_for_shipping_real      =(COL_DAYS_REAL,    "first"),
        # Monetary / quantity aggregates
        order_value                 =(COL_ITEM_TOTAL,   "sum"),
        n_items                     =(COL_ITEM_QTY,     "sum"),
        avg_discount_rate           =(COL_ITEM_DISC,    "mean"),
        # Secondary target aggregate
        aux_target_cancel_fraud     =("_cancel_fraud",  "max"),
    ).reset_index()

    # -- attach main_item_category and n_distinct_categories -----------------
    agg = agg.merge(top_item, on=COL_ORDER_ID, how="left")
    agg = agg.merge(n_cats,   on=COL_ORDER_ID, how="left")

    return agg


# ===========================================================================
# 5. Order-level feature engineering
# ===========================================================================

def engineer_features(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()

    # Payment type
    orders["is_cash"] = (orders["Type"] == "CASH").astype(np.int8)

    # Temporal features (all derived from order_date, known at order time)
    orders["order_month"]   = orders["order_date"].dt.month.astype(np.int8)
    orders["order_dow"]     = orders["order_date"].dt.day_of_week.astype(np.int8)  # Mon=0
    orders["order_hour"]    = orders["order_date"].dt.hour.astype(np.int8)
    orders["is_weekend"]    = (orders["order_dow"] >= 5).astype(np.int8)
    orders["is_night_order"]= (
        (orders["order_hour"] >= 22) | (orders["order_hour"] < 6)
    ).astype(np.int8)

    # Value-per-item (guard against 0-item edge case)
    orders["value_per_item"] = np.where(
        orders["n_items"] > 0,
        orders["order_value"] / orders["n_items"],
        0.0,
    )

    # Discount flag
    orders["high_discount_flag"] = (orders["avg_discount_rate"] > 0.15).astype(np.int8)

    return orders


# ===========================================================================
# 6. outcome_known_date — determines history window boundary
# ===========================================================================

def add_outcome_known_date(orders: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Compute when each order's outcome became visible (used ONLY for history
    gating — this column is dropped before saving).

    rule = 'shipping_date'        → outcome_known_date = shipping_date
    rule = 'order_date_plus_real' → outcome_known_date = order_date + real_days
    """
    orders = orders.copy()
    if rule == "shipping_date":
        orders["outcome_known_date"] = orders["shipping_date"]
    else:
        orders["outcome_known_date"] = orders["order_date"] + pd.to_timedelta(
            orders["Days_for_shipping_real"], unit="D"
        )
    return orders


# ===========================================================================
# 7. Leakage-safe history features
# ===========================================================================
#
# For each order O placed at time T:
#   "past" orders = orders whose outcome_known_date < T
#   (i.e., the outcome was visible before O was placed)
#
# We use np.searchsorted on sorted cumulative arrays for O(n log n) speed.
# ===========================================================================

def add_history_features(orders: pd.DataFrame, train_late_mean: float) -> pd.DataFrame:
    """
    Add customer-level and country-level history features.

    Parameters
    ----------
    orders : DataFrame sorted by order_date (ascending)
    train_late_mean : float
        Late-delivery rate computed on the first TRAIN_FRAC of orders only.
        Used as the prior / fallback for unseen customers/countries.

    Returns the modified DataFrame with new columns:
        customer_prev_orders, customer_prev_late_rate,
        country_prev_orders,  country_prev_late_rate,
        customer_prev_cash_share, days_since_prev_order
    """
    orders = orders.copy()

    # Sort order_date so searchsorted works correctly
    orders = orders.sort_values("order_date").reset_index(drop=True)

    # Convenience arrays
    t_order   = orders["order_date"].values.astype("datetime64[ns]").astype(np.int64)
    t_outcome = orders["outcome_known_date"].values.astype("datetime64[ns]").astype(np.int64)
    late_arr  = orders["Late_delivery_risk"].values.astype(np.float64)
    cash_arr  = orders["is_cash"].values.astype(np.float64)
    cust_arr  = orders["Customer_Id"].values
    ctry_arr  = orders["Order_Country"].values

    n = len(orders)

    # Pre-allocate output arrays
    cust_prev_orders   = np.zeros(n, dtype=np.int32)
    cust_prev_late_sum = np.zeros(n, dtype=np.float64)
    cust_prev_cash_sum = np.zeros(n, dtype=np.float64)
    days_since_prev    = np.full(n, np.nan)

    ctry_prev_orders   = np.zeros(n, dtype=np.int32)
    ctry_prev_late_sum = np.zeros(n, dtype=np.float64)

    # ---- Customer-level history ----------------------------------------
    # Group indices by customer; iterate over each customer's orders once.
    # outcome_known_date is sorted along with order_date (sorted globally),
    # but within a customer the relationship matters for the sliding window.
    # We build cumulative arrays per customer sorted by order_date and use
    # searchsorted to find how many prior outcomes are visible.

    cust_index: dict[object, list[int]] = {}
    for i, cid in enumerate(cust_arr):
        cust_index.setdefault(cid, []).append(i)

    for cid, idxs in cust_index.items():
        idxs_arr = np.array(idxs)                # positions in global array
        # sort by order_date within customer
        order_times   = t_order[idxs_arr]
        outcome_times = t_outcome[idxs_arr]
        late_vals     = late_arr[idxs_arr]
        cash_vals     = cash_arr[idxs_arr]

        sort_order  = np.argsort(order_times)
        idxs_sorted = idxs_arr[sort_order]
        ord_times_s = order_times[sort_order]
        out_times_s = outcome_times[sort_order]
        late_vals_s = late_vals[sort_order]
        cash_vals_s = cash_vals[sort_order]

        # Cumulative sums on outcome_known_date-sorted events
        # For each order at position j in sorted list, we want:
        #   count of prior orders whose outcome_known_date < current order_date
        # outcome_known_date and order_date may not be in the same sort order,
        # so we build two sorted arrays and searchsort the boundary.

        # Sort by outcome_known_date to form the "observable" sequence
        out_sort   = np.argsort(out_times_s)
        out_times_sorted = out_times_s[out_sort]
        late_cum   = np.cumsum(late_vals_s[out_sort])
        cash_cum   = np.cumsum(cash_vals_s[out_sort])
        # Prepend a 0 for the "nothing visible" state
        late_cum_ext = np.concatenate([[0.0], late_cum])
        cash_cum_ext = np.concatenate([[0.0], cash_cum])

        for j, global_idx in enumerate(idxs_sorted):
            cur_order_time = ord_times_s[j]

            # Number of past outcomes with outcome_known_date < current order_date
            pos = np.searchsorted(out_times_sorted, cur_order_time, side="left")
            count = int(pos)

            cust_prev_orders[global_idx]   = count
            cust_prev_late_sum[global_idx] = late_cum_ext[count]
            cust_prev_cash_sum[global_idx] = cash_cum_ext[count]

            # Days since previous order (use order_date, not outcome)
            if j > 0:
                prev_order_time = ord_times_s[j - 1]
                delta_ns = cur_order_time - prev_order_time
                days_since_prev[global_idx] = delta_ns / 1e9 / 86400.0

    # ---- Country-level history -----------------------------------------
    ctry_index: dict[object, list[int]] = {}
    for i, ctry in enumerate(ctry_arr):
        ctry_index.setdefault(ctry, []).append(i)

    for ctry, idxs in ctry_index.items():
        idxs_arr = np.array(idxs)
        order_times   = t_order[idxs_arr]
        outcome_times = t_outcome[idxs_arr]
        late_vals     = late_arr[idxs_arr]

        sort_order  = np.argsort(order_times)
        idxs_sorted = idxs_arr[sort_order]
        ord_times_s = order_times[sort_order]
        out_times_s = outcome_times[sort_order]
        late_vals_s = late_vals[sort_order]

        out_sort   = np.argsort(out_times_s)
        out_times_sorted = out_times_s[out_sort]
        late_cum   = np.cumsum(late_vals_s[out_sort])
        late_cum_ext = np.concatenate([[0.0], late_cum])

        for j, global_idx in enumerate(idxs_sorted):
            cur_order_time = ord_times_s[j]
            pos  = np.searchsorted(out_times_sorted, cur_order_time, side="left")
            count = int(pos)

            ctry_prev_orders[global_idx]   = count
            ctry_prev_late_sum[global_idx] = late_cum_ext[count]

    # ---- Assemble feature columns ----------------------------------------

    # Customer-level late rate (fill no-history rows with train_late_mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        cust_late_rate = np.where(
            cust_prev_orders > 0,
            cust_prev_late_sum / cust_prev_orders,
            train_late_mean,
        )
        cust_cash_share = np.where(
            cust_prev_orders > 0,
            cust_prev_cash_sum / cust_prev_orders,
            np.nan,
        )

    # Country-level late rate with Bayesian smoothing
    # smoothed = (late_sum + k * prior_mean) / (n + k)
    ctry_late_rate_smooth = (
        ctry_prev_late_sum + COUNTRY_K * train_late_mean
    ) / (ctry_prev_orders + COUNTRY_K)

    orders["customer_prev_orders"]     = cust_prev_orders
    orders["customer_prev_late_rate"]  = cust_late_rate
    orders["country_prev_orders"]      = ctry_prev_orders
    orders["country_prev_late_rate"]   = ctry_late_rate_smooth
    orders["customer_prev_cash_share"] = cust_cash_share
    orders["days_since_prev_order"]    = days_since_prev

    return orders


# ===========================================================================
# 8. Main pipeline
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  RTO Shield AI – Step 2: Build Order-Level Feature Table")
    print("=" * 70)

    # -- 8.1 Load ----------------------------------------------------------
    print("\n[1/7] Loading raw CSV …")
    raw = load_raw()
    print(f"      Raw rows: {len(raw):,}  |  columns: {raw.shape[1]}")

    # -- 8.2 Parse dates ---------------------------------------------------
    print("\n[2/7] Parsing order and shipping dates …")
    raw = parse_dates(raw)

    # -- 8.3 Date-rule check -----------------------------------------------
    print("\n[3/7] Running date-rule check …")
    rule = check_date_rule(raw)

    # -- 8.4 Collapse to order level ---------------------------------------
    print("\n[4/7] Collapsing to order level …")
    orders = build_order_level(raw)
    print(f"      Order-level rows: {len(orders):,}")

    # -- 8.5 Feature engineering -------------------------------------------
    print("\n[5/7] Engineering features …")
    orders = engineer_features(orders)

    # -- 8.6 Leakage-safe history features ---------------------------------
    print("\n[6/7] Computing leakage-safe history features …")

    # outcome_known_date (not saved to parquet)
    orders = add_outcome_known_date(orders, rule)

    # Sort globally by order_date before computing history
    orders = orders.sort_values("order_date").reset_index(drop=True)

    # Train mean = late rate of the first TRAIN_FRAC orders only
    n_train = int(np.floor(len(orders) * TRAIN_FRAC))
    train_late_mean = float(orders.iloc[:n_train]["Late_delivery_risk"].mean())
    print(f"      Train-split late mean (prior): {train_late_mean:.4f}")

    orders = add_history_features(orders, train_late_mean)

    # Drop internal columns that must not appear in the parquet
    drop_cols = [
        "outcome_known_date",   # history-gating only
        "shipping_date",        # post-dispatch leakage
        "Days_for_shipping_real",# post-dispatch leakage
    ]
    orders = orders.drop(columns=[c for c in drop_cols if c in orders.columns])

    # -- 8.7 Save ----------------------------------------------------------
    print("\n[7/7] Saving orders.parquet …")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    orders.to_parquet(ORDER_PARQUET, index=False)
    print(f"      Saved → {ORDER_PARQUET}")
    print(f"      Shape : {orders.shape[0]:,} rows × {orders.shape[1]} columns")

    # -- Summary statistics ------------------------------------------------
    _print_summary(orders, raw, rule, train_late_mean)


# ===========================================================================
# 9. Summary printout
# ===========================================================================

def _print_summary(
    orders: pd.DataFrame,
    raw: pd.DataFrame,
    rule: str,
    train_late_mean: float,
) -> None:
    sep = "─" * 70

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    # Counts
    n_orders    = len(orders)
    n_customers = orders["Customer_Id"].nunique()
    n_countries = orders["Order_Country"].nunique()
    date_min    = orders["order_date"].min()
    date_max    = orders["order_date"].max()

    print(f"\n  Orders       : {n_orders:,}")
    print(f"  Customers    : {n_customers:,}")
    print(f"  Countries    : {n_countries:,}")
    print(f"  Date range   : {date_min.date()} → {date_max.date()}")
    print(f"  Span         : {(date_max - date_min).days} days")

    # Positive rate (overall late rate)
    pos_rate = orders["Late_delivery_risk"].mean()
    print(f"\n  Overall late-delivery rate  : {pos_rate:.4f}  ({pos_rate*100:.2f}%)")
    print(f"  Train-split prior (k=20)    : {train_late_mean:.4f}")

    # Late rate by Shipping Mode
    print(f"\n  {sep}")
    print(f"  Late rate by Shipping Mode")
    print(f"  {sep}")
    ship_grp = (
        orders.groupby("Shipping_Mode")["Late_delivery_risk"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "late_rate", "count": "orders"})
        .sort_values("late_rate", ascending=False)
    )
    for mode, row in ship_grp.iterrows():
        bar = "█" * int(row["late_rate"] * 30)
        print(f"  {mode:<20}  {row['late_rate']:.4f}  ({row['orders']:>6,} orders)  {bar}")

    # Late rate by Payment Type
    print(f"\n  {sep}")
    print(f"  Late rate by Payment Type (Type)")
    print(f"  {sep}")
    pay_grp = (
        orders.groupby("Type")["Late_delivery_risk"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "late_rate", "count": "orders"})
        .sort_values("late_rate", ascending=False)
    )
    for ptype, row in pay_grp.iterrows():
        bar = "█" * int(row["late_rate"] * 30)
        print(f"  {str(ptype):<20}  {row['late_rate']:.4f}  ({row['orders']:>6,} orders)  {bar}")

    # Date rule recap
    print(f"\n  {sep}")
    print(f"  Date-Rule Result")
    print(f"  {sep}")
    if rule == "shipping_date":
        print("  [OK] Rule used : outcome_known_date = shipping_date")
        print("     Reason   : ≥90% of rows have (ship_date - order_date).days == real_days,")
        print("                so shipping_date faithfully marks when the outcome is known.")
    else:
        print("  [WARN] Rule used : outcome_known_date = order_date + Days_for_shipping_(real)")
        print("     Reason   : <90% rows satisfy the equality; conservative window used.")

    print(f"\n{'=' * 70}\n")

    # Feature column audit (no forbidden columns)
    feature_cols = [
        c for c in orders.columns
        if c not in {
            COL_ORDER_ID, "order_date",
            "Late_delivery_risk", AUX_TARGET_COL,
            "Customer_Id",
        }
    ]
    forbidden_in_features = [
        c for c in feature_cols
        if any(f.lower() in c.lower() for f in FORBIDDEN_FEATURES)
    ]
    if forbidden_in_features:
        print(f"  [FAIL] FORBIDDEN columns found in feature set: {forbidden_in_features}")
    else:
        print("  [OK] No FORBIDDEN_FEATURES detected in the feature columns.")
    print()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()
