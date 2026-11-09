"""
tests/test_no_leakage.py
========================
Pytest suite for RTO Shield AI – Step 2 (01_prepare.py).

Tests
-----
1. test_history_brute_force   – for a random sample of 2 000 orders, recompute
   customer_prev_orders, customer_prev_late_rate, country_prev_orders,
   country_prev_late_rate, customer_prev_cash_share, and days_since_prev_order
   by brute force and assert they match the vectorized values produced by
   01_prepare.py within floating-point tolerance.

2. test_no_forbidden_in_features – assert that none of the FORBIDDEN_FEATURES
   (and the aux secondary target) appear in the model feature list.

Run:
    pytest tests/test_no_leakage.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROCESSED_DIR, FORBIDDEN_FEATURES, TRAIN_FRAC

# ---------------------------------------------------------------------------
# Load 01_prepare as "src_prepare" (must happen before any from-imports of it)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("src_prepare", SRC_DIR / "01_prepare.py")
_mod  = _ilu.module_from_spec(_spec)       # type: ignore[arg-type]
_spec.loader.exec_module(_mod)             # type: ignore[union-attr]
sys.modules["src_prepare"] = _mod

from src_prepare import (
    COL_ORDER_ID,
    COL_CUSTOMER_ID,
    COL_ORDER_COUNTRY,
    AUX_TARGET_COL,
    COUNTRY_K,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"
SEED          = 42
SAMPLE_N      = 2000


@pytest.fixture(scope="module")
def orders() -> pd.DataFrame:
    """Load the pre-built parquet once per test session."""
    if not ORDER_PARQUET.exists():
        pytest.skip(
            "orders.parquet not found – run `python src/01_prepare.py` first."
        )
    df = pd.read_parquet(ORDER_PARQUET)
    # Ensure correct sort order
    df = df.sort_values("order_date").reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def sample(orders) -> pd.DataFrame:
    """Random sample of SAMPLE_N orders for brute-force checks."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(orders), size=min(SAMPLE_N, len(orders)), replace=False)
    return orders.iloc[np.sort(idx)].copy()


# ---------------------------------------------------------------------------
# Helper: brute-force history for one order
# ---------------------------------------------------------------------------

def _brute_force_one(
    order_row: pd.Series,
    all_orders: pd.DataFrame,
    train_late_mean: float,
) -> dict:
    """
    Recompute history features for a single order by scanning all_orders.

    "Past" means: same customer / country, AND outcome_known_date < order_date.
    outcome_known_date is approximated here as:
        order_date + days_since_prev_order proxy  (we cannot reconstruct it
        exactly without re-running the pipeline, so we re-derive it from the
        parquet columns that survive: we use the fact that the pipeline
        dropped shipping_date and Days_for_shipping_real, and stored the
        CHOSEN rule's result implicitly via the history columns).

    Instead of re-deriving outcome_known_date, we use the SIMPLER brute-force
    definition that matches the vectorised code exactly:

        A past order q is "visible" to current order p iff:
            q["outcome_known_date"] < p["order_date"]

    Because the parquet does not store outcome_known_date, we recompute it
    here using the same rule the main script used (shipping_date if >90%
    match, else order_date + real_days).  Since the parquet dropped both,
    we reload the raw dataset once per session and join on Order Id.
    """
    raise NotImplementedError("Use _brute_force_one_from_raw instead.")


# ---------------------------------------------------------------------------
# Module-level raw data cache (loaded once)
# ---------------------------------------------------------------------------

_raw_cache: dict[str, pd.DataFrame] = {}


def _get_enriched(all_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Merge outcome_known_date back into all_orders by re-reading the raw CSV
    and re-applying the same date rule.  Result is cached.
    """
    if "enriched" in _raw_cache:
        return _raw_cache["enriched"]

    raw = _mod.load_raw()
    raw = _mod.parse_dates(raw)
    rule = _mod.check_date_rule(raw)

    # Order-level shipping/real columns
    order_dates = (
        raw.groupby(COL_ORDER_ID, sort=False)
        .agg(
            shipping_date  =("shipping date (DateOrders)", "first"),
            real_days      =("Days for shipping (real)",   "first"),
            order_date_raw =("order date (DateOrders)",    "first"),
        )
        .reset_index()
    )

    if rule == "shipping_date":
        order_dates["outcome_known_date"] = order_dates["shipping_date"]
    else:
        order_dates["outcome_known_date"] = (
            order_dates["order_date_raw"]
            + pd.to_timedelta(order_dates["real_days"], unit="D")
        )

    enriched = all_orders.merge(
        order_dates[[COL_ORDER_ID, "outcome_known_date"]],
        on=COL_ORDER_ID,
        how="left",
    )
    _raw_cache["enriched"] = enriched
    _raw_cache["rule"]     = rule
    return enriched


# ---------------------------------------------------------------------------
# Test 1: brute-force vs vectorised history features
# ---------------------------------------------------------------------------

def test_history_brute_force(orders, sample):
    """
    For each sampled order, recompute history features by O(n) scan and
    compare with the vectorised values stored in the parquet.
    """
    enriched    = _get_enriched(orders)
    rule        = _raw_cache["rule"]

    # Recompute train_late_mean (first TRAIN_FRAC of globally-sorted orders)
    n_train          = int(np.floor(len(orders) * TRAIN_FRAC))
    train_late_mean  = float(orders.iloc[:n_train]["Late_delivery_risk"].mean())

    errors: list[str] = []

    for _, row in sample.iterrows():
        oid         = row[COL_ORDER_ID]
        cid         = row["Customer_Id"]
        ctry        = row["Order_Country"]
        order_time  = row["order_date"]

        # Retrieve outcome_known_date for all orders
        def visible(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["outcome_known_date"] < order_time]

        all_cust_past  = visible(
            enriched[enriched["Customer_Id"] == cid]
        )
        all_ctry_past  = visible(
            enriched[enriched["Order_Country"] == ctry]
        )

        # --- customer features ---
        bf_cust_n    = len(all_cust_past)
        bf_cust_late = (
            float(all_cust_past["Late_delivery_risk"].mean())
            if bf_cust_n > 0
            else train_late_mean
        )
        bf_cash_share = (
            float(all_cust_past["is_cash"].mean())
            if bf_cust_n > 0
            else np.nan
        )

        # --- country features ---
        bf_ctry_n    = len(all_ctry_past)
        bf_ctry_late = (
            float(
                (all_ctry_past["Late_delivery_risk"].sum() + COUNTRY_K * train_late_mean)
                / (bf_ctry_n + COUNTRY_K)
            )
        )

        # --- days since prev order ---
        all_cust_past_ordered = (
            enriched[
                (enriched["Customer_Id"] == cid)
                & (enriched["order_date"] < order_time)
            ]
            .sort_values("order_date")
        )
        if len(all_cust_past_ordered) > 0:
            prev_order_time = all_cust_past_ordered.iloc[-1]["order_date"]
            bf_days_since   = (order_time - prev_order_time).total_seconds() / 86400.0
        else:
            bf_days_since   = np.nan

        # --- compare ---
        tol = 1e-6

        checks = [
            (
                "customer_prev_orders",
                row["customer_prev_orders"],
                bf_cust_n,
                0,
            ),
            (
                "customer_prev_late_rate",
                row["customer_prev_late_rate"],
                bf_cust_late,
                tol,
            ),
            (
                "country_prev_orders",
                row["country_prev_orders"],
                bf_ctry_n,
                0,
            ),
            (
                "country_prev_late_rate",
                row["country_prev_late_rate"],
                bf_ctry_late,
                tol,
            ),
        ]

        for name, vec_val, bf_val, atol in checks:
            if not np.isclose(float(vec_val), float(bf_val), atol=max(atol, 1e-9), rtol=1e-5):
                errors.append(
                    f"Order {oid}: {name} mismatch "
                    f"(vectorised={vec_val}, brute_force={bf_val})"
                )

        # cash share (allow nan == nan)
        vec_cs = row["customer_prev_cash_share"]
        if np.isnan(bf_cash_share):
            if not np.isnan(vec_cs):
                errors.append(
                    f"Order {oid}: customer_prev_cash_share expected NaN, got {vec_cs}"
                )
        else:
            if not np.isclose(float(vec_cs), float(bf_cash_share), atol=tol, rtol=1e-5):
                errors.append(
                    f"Order {oid}: customer_prev_cash_share mismatch "
                    f"(vectorised={vec_cs}, brute_force={bf_cash_share})"
                )

        # days_since_prev_order
        vec_ds = row["days_since_prev_order"]
        if np.isnan(bf_days_since):
            if not np.isnan(vec_ds):
                errors.append(
                    f"Order {oid}: days_since_prev_order expected NaN, got {vec_ds}"
                )
        else:
            if not np.isclose(float(vec_ds), float(bf_days_since), atol=0.01, rtol=1e-4):
                errors.append(
                    f"Order {oid}: days_since_prev_order mismatch "
                    f"(vectorised={vec_ds:.4f}, brute_force={bf_days_since:.4f})"
                )

    if errors:
        # Print first 20 mismatches for diagnostics
        for e in errors[:20]:
            print(f"MISMATCH: {e}")
        pytest.fail(
            f"{len(errors)} history-feature mismatches found in sample of {len(sample)}."
        )
    else:
        print(
            f"\n✅ Brute-force check passed for {len(sample):,} sampled orders "
            f"— all 6 history features match."
        )


# ---------------------------------------------------------------------------
# Test 2: no forbidden column in feature lists
# ---------------------------------------------------------------------------

def test_no_forbidden_in_features(orders):
    """
    Assert that FORBIDDEN_FEATURES columns and the aux secondary target are
    absent from the parquet's feature columns.

    Non-feature columns excluded from check:
        Order Id, order_date, Late_delivery_risk, Customer_Id,
        aux_target_cancel_fraud
    """
    # Columns that are intentionally in the parquet but are NOT model features
    non_feature_cols = {
        COL_ORDER_ID,
        "order_date",
        "Late_delivery_risk",
        "Customer_Id",
        AUX_TARGET_COL,
    }

    feature_cols = [c for c in orders.columns if c not in non_feature_cols]

    forbidden_violations: list[str] = []

    # Check exact column names from FORBIDDEN_FEATURES
    for f in FORBIDDEN_FEATURES:
        if f in feature_cols:
            forbidden_violations.append(f"FORBIDDEN column '{f}' found in features")

    # Check that the aux target is not accidentally in feature columns
    if AUX_TARGET_COL in feature_cols:
        forbidden_violations.append(
            f"Secondary evaluation target '{AUX_TARGET_COL}' found in feature columns"
        )

    # outcome_known_date must have been dropped
    if "outcome_known_date" in orders.columns:
        forbidden_violations.append(
            "'outcome_known_date' was not dropped — it must never be a feature"
        )

    # shipping_date must have been dropped
    if "shipping_date" in orders.columns:
        forbidden_violations.append(
            "'shipping_date' was not dropped — it is a forbidden post-dispatch column"
        )

    # Days_for_shipping_real must have been dropped
    if "Days_for_shipping_real" in orders.columns:
        forbidden_violations.append(
            "'Days_for_shipping_real' was not dropped — it is a forbidden post-dispatch column"
        )

    if forbidden_violations:
        for v in forbidden_violations:
            print(f"VIOLATION: {v}")
        pytest.fail(
            f"{len(forbidden_violations)} leakage violations found.\n"
            + "\n".join(forbidden_violations)
        )
    else:
        print(
            f"\n✅ Leakage guard passed — none of the {len(FORBIDDEN_FEATURES)} FORBIDDEN_FEATURES "
            f"appear in the {len(feature_cols)} feature columns."
        )
        print(
            f"   aux_target_cancel_fraud is correctly excluded from features.\n"
            f"   outcome_known_date, shipping_date, Days_for_shipping_real "
            f"are correctly dropped."
        )
