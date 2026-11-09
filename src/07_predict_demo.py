"""
src/07_predict_demo.py
======================
Demo script for RTO Shield AI – Model 1 inference.

Selects 5 real test-set orders (2 High, 1 Medium, 2 Low risk), scores
each one with predict.score_order(), and prints a rich per-order report.

Also writes outputs/demo_scores.csv.

Usage
-----
  python src/07_predict_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR,
    TARGET_COL, TRAIN_FRAC, VAL_FRAC,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)
import predict   # loads lazily on first score call

ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"
DATE_COL      = "order_date"

# Hardcoded test-set indices identified from pre-flight check
# (2 High, 1 Medium, 2 Low — real orders with known actual outcomes)
DEMO_INDICES = [9846, 4, 4818, 5840, 9612]  # test-set positional indices
DEMO_LABELS  = ["High", "High", "Medium", "Low", "Low"]   # expected tiers


# ===========================================================================
# Helpers
# ===========================================================================

def _sep(char="─", w=74): return char * w


def _load_test_df() -> pd.DataFrame:
    orders = (
        pd.read_parquet(ORDER_PARQUET)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    n       = len(orders)
    te_st   = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    return orders.iloc[te_st:].copy().reset_index(drop=True)


def _print_order_card(
    rank:     int,
    result:   dict,
    actual:   int,
    order_id: int,
    mode:     str,
    ptype:    str,
    region:   str,
    sched:    int,
) -> None:
    """Print a formatted score card for one order."""
    flag  = "FLAGGED" if result["flagged"] else "OK"
    act_s = "Late" if actual == 1 else "On-time"
    correct = (
        "Correct" if
        (result["flagged"] and actual == 1) or (not result["flagged"] and actual == 0)
        else "Wrong"
    )

    print(f"\n  {'─'*72}")
    print(f"  Order #{rank}  │  Order Id: {order_id}  │  Tier: {result['tier']}  │  {flag}")
    print(f"  {'─'*72}")
    print(f"  Risk Score : {result['risk_score']:.4f}")
    print(f"  Actual     : {act_s}  ({correct})")
    print(f"  Shipping   : {mode}  │  Payment: {ptype}  │  Region: {region}  │  Sched days: {sched}")
    print(f"  Top Reasons:")
    for i, reason in enumerate(result["top_reasons"], 1):
        print(f"    {i}. {reason}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 74)
    print("  RTO Shield AI – Prediction Demo  (5 real test-set orders)")
    print("=" * 74)

    # ── Load test set ─────────────────────────────────────────────────────────
    print("\n  Loading test orders …")
    test_df = _load_test_df()
    print(f"  Test set: {len(test_df):,} orders")

    # ── Score 5 selected orders ───────────────────────────────────────────────
    print(f"\n  Scoring 5 demo orders …  (model loads on first call)")

    results    = []
    csv_rows   = []
    order_dicts = []

    for rank, (idx, expected_tier) in enumerate(zip(DEMO_INDICES, DEMO_LABELS), 1):
        row = test_df.iloc[idx]

        # Build order dict — only feature columns (no target / forbidden cols)
        order_dict = row.to_dict()

        result = predict.score_order(order_dict)

        actual   = int(row[TARGET_COL])
        order_id = int(row["Order Id"])
        mode     = str(row["Shipping_Mode"])
        ptype    = str(row["Type"])
        region   = str(row["Order_Region"])
        sched    = int(row["Days_for_shipment_scheduled"])

        _print_order_card(rank, result, actual, order_id, mode, ptype, region, sched)

        results.append(result)
        order_dicts.append(order_dict)
        csv_rows.append({
            "rank":         rank,
            "order_id":     order_id,
            "risk_score":   result["risk_score"],
            "tier":         result["tier"],
            "flagged":      result["flagged"],
            "actual":       "Late" if actual == 1 else "On-time",
            "shipping_mode":mode,
            "payment_type": ptype,
            "region":       region,
            "sched_days":   sched,
            "reason_1":     result["top_reasons"][0] if len(result["top_reasons"]) > 0 else "",
            "reason_2":     result["top_reasons"][1] if len(result["top_reasons"]) > 1 else "",
            "reason_3":     result["top_reasons"][2] if len(result["top_reasons"]) > 2 else "",
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n  {'='*74}")
    print(f"  Summary Table")
    print(f"  {'='*74}")
    print(f"  {'#':>2}  {'OrderId':>8}  {'Score':>7}  {'Tier':>7}  "
          f"{'Flag':>7}  {'Actual':>8}  {'Shipping Mode':<18}")
    print(f"  {_sep('-', 70)}")
    for r, row in enumerate(csv_rows, 1):
        flag_s = "YES" if row["flagged"] else "no"
        print(
            f"  {r:>2}  {row['order_id']:>8}  {row['risk_score']:>7.4f}  "
            f"{row['tier']:>7}  {flag_s:>7}  {row['actual']:>8}  "
            f"{row['shipping_mode']:<18}"
        )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = OUTPUTS_DIR / "demo_scores.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\n  Demo scores saved → {out_csv}")

    # ── Also save a sample_order.json (the first order, for CLI demo) ─────────
    sample_path = MODELS_DIR.parent / "sample_order.json"
    # Serialise — convert numpy types to Python natives
    sample_dict = {}
    for k, v in order_dicts[0].items():
        if isinstance(v, (np.integer,)):
            sample_dict[k] = int(v)
        elif isinstance(v, (np.floating,)):
            sample_dict[k] = float(v)
        elif pd.isna(v) if not isinstance(v, str) else False:
            sample_dict[k] = None
        else:
            sample_dict[k] = v
    sample_path.write_text(json.dumps(sample_dict, indent=2, default=str))
    print(f"  Sample order JSON saved → {sample_path}")

    # ── CLI demo hint ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*74}")
    print(f"  CLI usage:")
    print(f"    python src/predict.py --json sample_order.json --pretty")
    print(f"\n  FastAPI wrapper (future):")
    print(f"    from fastapi import FastAPI")
    print(f"    from predict import score_order")
    print(f"    app = FastAPI()")
    print(f"    @app.post('/score')")
    print(f"    def score(order: dict) -> dict: return score_order(order)")

    print("\n" + "=" * 74)
    print("  Demo complete.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()

