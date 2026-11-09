"""
tests/test_pipeline.py
======================
4 pytest tests covering core pipeline correctness:
  1. Chronological split has no date overlap
  2. No forbidden feature appears in the trained model's feature names
  3. score_order() returns a valid output dict
  4. History features contain no future-information leakage
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    FORBIDDEN_FEATURES,
    TRAIN_FRAC,
    VAL_FRAC,
    MODELS_DIR,
    TARGET_COL,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "orders.parquet"
DATE_COL = "order_date"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def orders_df():
    """Load the processed orders parquet once for all tests."""
    df = pd.read_parquet(PARQUET_PATH)
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def splits(orders_df):
    """Return train / val / test DataFrames using the same 70/15/15 split."""
    n = len(orders_df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train = orders_df.iloc[:train_end]
    val = orders_df.iloc[train_end:val_end]
    test = orders_df.iloc[val_end:]
    return train, val, test


# ---------------------------------------------------------------------------
# Test 1: Chronological split has no date overlap
# ---------------------------------------------------------------------------

def test_chrono_split_no_overlap(splits):
    """
    max(train.order_date) < min(val.order_date) and
    max(val.order_date) < min(test.order_date).
    Guarantees the temporal split is strictly non-overlapping.
    """
    train, val, test = splits

    train_max = pd.Timestamp(train[DATE_COL].max())
    val_min   = pd.Timestamp(val[DATE_COL].min())
    val_max   = pd.Timestamp(val[DATE_COL].max())
    test_min  = pd.Timestamp(test[DATE_COL].min())

    assert train_max < val_min, (
        f"Date overlap between TRAIN and VAL: "
        f"train_max={train_max}, val_min={val_min}"
    )
    assert val_max < test_min, (
        f"Date overlap between VAL and TEST: "
        f"val_max={val_max}, test_min={test_min}"
    )
    print(f"\n  Train: {train[DATE_COL].min()} → {train_max}  ({len(train):,} rows)")
    print(f"  Val:   {val_min} → {val_max}  ({len(val):,} rows)")
    print(f"  Test:  {test_min} → {test[DATE_COL].max()}  ({len(test):,} rows)")
    print("  ✅ No date overlap confirmed.")


# ---------------------------------------------------------------------------
# Test 2: No forbidden feature appears in the trained preprocessor
# ---------------------------------------------------------------------------

def test_no_forbidden_features_in_model():
    """
    Load the fitted preprocessor and assert that none of the FORBIDDEN_FEATURES
    (post-dispatch leakage or PII columns) appear in the model's feature names.
    Also explicitly checks for 'aux_target_cancel_fraud'.
    """
    import joblib

    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    assert preprocessor_path.exists(), f"Preprocessor not found at {preprocessor_path}"

    preprocessor = joblib.load(preprocessor_path)
    feature_names = list(preprocessor.get_feature_names_out())

    # Normalise to lower-case for robust substring matching
    feature_names_lower = [f.lower() for f in feature_names]

    extra_forbidden = ["aux_target_cancel_fraud", "shipping_date", "outcome_known_date",
                       "days_for_shipping_real"]
    all_forbidden = list(FORBIDDEN_FEATURES) + extra_forbidden

    violations = []
    for bad in all_forbidden:
        bad_lower = bad.lower().replace(" ", "_")
        for fn_lower in feature_names_lower:
            if bad_lower in fn_lower:
                violations.append((bad, fn_lower))

    if violations:
        for orig, matched in violations:
            print(f"  ❌ Forbidden '{orig}' matched feature '{matched}'")

    assert not violations, (
        f"Forbidden features found in model: {violations}"
    )
    print(f"\n  Checked {len(feature_names)} encoded features against "
          f"{len(all_forbidden)} forbidden names. ✅ None found.")


# ---------------------------------------------------------------------------
# Test 3: score_order() returns a valid output dict
# ---------------------------------------------------------------------------

def test_score_order_returns_valid_output(orders_df):
    """
    Import predict.score_order and call it with a real order from the test set.
    Assert:
      - 0.0 <= risk_score <= 1.0
      - tier in {Low, Medium, High}
      - flagged is bool
      - top_reasons is a list of exactly 3 strings
    """
    from predict import score_order

    # Use the last (test) rows; pick one real order
    n = len(orders_df)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    test_df = orders_df.iloc[val_end:].reset_index(drop=True)

    sample = test_df.iloc[0].to_dict()

    result = score_order(sample)

    assert isinstance(result, dict), "score_order must return a dict"
    assert "risk_score" in result, "Missing 'risk_score' key"
    assert "tier" in result, "Missing 'tier' key"
    assert "flagged" in result, "Missing 'flagged' key"
    assert "top_reasons" in result, "Missing 'top_reasons' key"

    assert 0.0 <= result["risk_score"] <= 1.0, (
        f"risk_score={result['risk_score']} is outside [0, 1]"
    )
    assert result["tier"] in {"Low", "Medium", "High"}, (
        f"tier='{result['tier']}' is not one of Low/Medium/High"
    )
    assert isinstance(result["flagged"], bool), (
        f"flagged must be bool, got {type(result['flagged'])}"
    )
    assert isinstance(result["top_reasons"], list), "top_reasons must be a list"
    assert len(result["top_reasons"]) == 3, (
        f"Expected 3 reasons, got {len(result['top_reasons'])}"
    )

    print(f"\n  Tested order: risk_score={result['risk_score']:.4f}, "
          f"tier={result['tier']}, flagged={result['flagged']}")
    print(f"  Reasons: {result['top_reasons']}")
    print("  ✅ score_order output is valid.")


# ---------------------------------------------------------------------------
# Test 4: History features contain no future information
# ---------------------------------------------------------------------------

def test_history_features_no_future_leakage(orders_df):
    """
    For a 500-row random sample, verify that customer_prev_orders does not
    exceed the count of prior orders placed strictly before the current order.

    In 01_prepare.py, customer_prev_orders counts orders whose delivery outcome
    was observable before the current order (outcome_known_date < order_date).
    Therefore, stored_prev <= total_prior_orders_placed is a strict invariant:
    if stored_prev > expected_max_prior, future orders were leaked.
    """
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(orders_df), size=500, replace=False)
    sample = orders_df.iloc[sample_idx].reset_index(drop=True)

    # Build a lookup: customer_id → sorted array of order_date timestamps
    cust_col = "Customer_Id" if "Customer_Id" in orders_df.columns else None
    if cust_col is None:
        # Try alternate casing
        for col in orders_df.columns:
            if col.lower() == "customer_id":
                cust_col = col
                break

    if cust_col is None:
        pytest.skip("Customer ID column not found in parquet — skipping leakage test.")

    # Build series of dates per customer
    all_dates_by_cust = orders_df.groupby(cust_col)[DATE_COL].apply(list).to_dict()

    leakage_violations = []   # stored > expected_max → actual future leakage
    in_transit_cases = []     # stored < expected_max → prior orders still in transit (correctly excluded)

    for _, row in sample.iterrows():
        cid = row[cust_col]
        order_dt = pd.Timestamp(row[DATE_COL])
        stored_prev = int(row["customer_prev_orders"])

        all_dates = [pd.Timestamp(d) for d in all_dates_by_cust.get(cid, [])]
        # Upper bound: count of all orders placed strictly before current order's timestamp
        expected_max_prior = sum(1 for d in all_dates if d < order_dt)

        if stored_prev > expected_max_prior:
            # ACTUAL leakage: history count includes future orders placed after order_dt
            leakage_violations.append({
                "customer_id": cid,
                "order_date": order_dt,
                "stored": stored_prev,
                "expected_max": expected_max_prior,
            })
        elif stored_prev < expected_max_prior:
            # Safe and expected: prior order was placed before order_dt, but its
            # delivery outcome was not yet known when this order was placed (in-transit).
            in_transit_cases.append((cid, stored_prev, expected_max_prior))

    if leakage_violations:
        for v in leakage_violations[:5]:
            print(f"  ❌ LEAKAGE {v}")
    if in_transit_cases:
        print(f"  ℹ️  {len(in_transit_cases)} orders have prior orders still in transit "
              f"at placement time (stored < total_prior) — correctly excluded from history.")

    assert not leakage_violations, (
        f"{len(leakage_violations)}/500 orders have stored > expected_max_prior "
        f"(future leakage detected!): {leakage_violations}"
    )

    print(f"\n  Checked 500 random orders for customer_prev_orders leakage.")
    print(f"  Leakage violations (stored > expected_max_prior): 0")
    print("  ✅ No future leakage detected in history features.")
