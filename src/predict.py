"""
src/predict.py
==============
Production-ready scoring module for RTO Shield AI - Model 1.

Public API
----------
score_order(order_dict) -> dict
    Score a single order supplied as a plain Python dict (or JSON-parsed dict).
    Returns:
        {
          "order_id":    int | str | None,
          "risk_score":  float,          # calibrated probability [0, 1]
          "tier":        "Low" | "Medium" | "High",
          "flagged":     bool,           # True if risk_score >= flag_threshold
          "top_reasons": [str, str, str] # 3 plain-English SHAP reason codes
        }

score_batch(df) -> pd.DataFrame
    Score a DataFrame of orders.  Columns: all fields returned by score_order
    plus "order_id" (taken from "Order Id" if present).

CLI
---
  python src/predict.py --json sample_order.json
  python src/predict.py --json sample_order.json --pretty

Design notes
------------
- Loads model/preprocessor/calibrator/thresholds lazily on first call (cached).
- Uses SHAP TreeExplainer for per-order reason codes (cached explainer).
- JSON-in / JSON-out interface: a FastAPI wrapper needs only 2 lines.
- FORBIDDEN columns (shipping_date, outcome columns, PII) are ignored if
  present in the input dict — they are never fed to the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing  import Any

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Make src/ importable when called directly
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config   import (
    MODELS_DIR, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    TARGET_COL, FORBIDDEN_FEATURES,
)
from features import _select_feature_cols
from reasons  import get_reason_codes

# ---------------------------------------------------------------------------
# Module-level cache (populated on first call)
# ---------------------------------------------------------------------------
_CACHE: dict[str, Any] = {}

_THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
_MODEL_PATH      = MODELS_DIR / "final_model.joblib"
_PREP_PATH       = MODELS_DIR / "preprocessor.joblib"
_CAL_PATH        = MODELS_DIR / "final_calibrator.joblib"


# ===========================================================================
# Lazy loader
# ===========================================================================

def _load_artifacts() -> None:
    """Load all artifacts into module cache on first call."""
    if _CACHE:
        return

    _CACHE["model"] = joblib.load(_MODEL_PATH)
    _CACHE["prep"]  = joblib.load(_PREP_PATH)
    _CACHE["cal"]   = joblib.load(_CAL_PATH)

    thr_data = json.loads(_THRESHOLDS_PATH.read_text())
    _CACHE["flag_threshold"] = float(
        thr_data["thresholds"].get("final_tuned_max_f1",
        thr_data["thresholds"].get("max_f1", 0.35))
    )
    _CACHE["low_cut"]  = float(thr_data["tier_cutoffs"]["low_max"])
    _CACHE["high_cut"] = float(thr_data["tier_cutoffs"]["high_min"])
    _CACHE["feat_names"] = list(_CACHE["prep"].get_feature_names_out())

    # SHAP explainer (expensive — created once)
    _CACHE["explainer"] = shap.TreeExplainer(_CACHE["model"])


# ===========================================================================
# Internal helpers
# ===========================================================================

def _tier(score: float) -> str:
    low_cut  = _CACHE["low_cut"]
    high_cut = _CACHE["high_cut"]
    if score < low_cut:
        return "Low"
    if score < high_cut:
        return "Medium"
    return "High"


def _dict_to_df(order_dict: dict) -> pd.DataFrame:
    """
    Convert a single order dict to a one-row DataFrame.
    Strips any FORBIDDEN columns silently.
    Missing feature columns are filled with NaN (the preprocessor's
    SimpleImputer handles them).
    """
    # Remove forbidden fields (post-dispatch leakage / PII)
    clean = {k: v for k, v in order_dict.items()
             if k not in FORBIDDEN_FEATURES}
    df = pd.DataFrame([clean])

    # Ensure all expected feature columns exist (fill missing with NaN)
    for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    return df


def _score_X(X: np.ndarray) -> np.ndarray:
    """Return calibrated probabilities for a transformed feature matrix."""
    raw   = _CACHE["model"].predict_proba(X)[:, 1]
    return _CACHE["cal"].predict(raw)


def _shap_reasons(X_row: np.ndarray, raw_dict: dict | None = None) -> list[str]:
    """
    Compute SHAP values for one transformed row and return 3 reason codes.
    """
    sv_row   = _CACHE["explainer"].shap_values(X_row)
    # LightGBM may return list[array] for binary classification
    if isinstance(sv_row, list):
        sv_row = sv_row[1]
    shap_flat = sv_row[0]
    feat_vals = X_row[0]
    return get_reason_codes(shap_flat, _CACHE["feat_names"], feat_vals, k=3, raw_dict=raw_dict)


# ===========================================================================
# Public API
# ===========================================================================

def score_order(order_dict: dict) -> dict:
    """
    Score a single order.

    Parameters
    ----------
    order_dict : dict
        Keys are the ORDER-LEVEL feature column names (as in orders.parquet).
        Post-dispatch leakage columns and PII are silently ignored.
        Missing feature columns default to NaN → imputed by preprocessor.

    Returns
    -------
    dict with keys:
        order_id    : value of "Order Id" key if present, else None
        risk_score  : float in [0, 1]  — calibrated late-delivery probability
        tier        : "Low" | "Medium" | "High"
        flagged     : bool — True if risk_score ≥ flag_threshold
        top_reasons : list[str] — 3 plain-English SHAP-derived reasons
    """
    _load_artifacts()

    order_id = order_dict.get("Order Id", None)
    df       = _dict_to_df(order_dict)
    X_raw    = _select_feature_cols(df, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    X        = _CACHE["prep"].transform(X_raw)

    score   = float(_score_X(X)[0])
    tier_   = _tier(score)
    flagged = score >= _CACHE["flag_threshold"]
    reasons = _shap_reasons(X, raw_dict=order_dict)

    return {
        "order_id":    order_id,
        "risk_score":  round(score, 6),
        "tier":        tier_,
        "flagged":     flagged,
        "top_reasons": reasons,
    }


def score_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of orders.

    Parameters
    ----------
    df : pd.DataFrame
        Each row is one order.  Same column expectations as score_order().

    Returns
    -------
    pd.DataFrame with columns:
        order_id, risk_score, tier, flagged, reason_1, reason_2, reason_3
    """
    _load_artifacts()

    order_ids = df.get("Order Id", pd.Series([None] * len(df))).tolist()

    # Ensure all feature cols exist
    df_clean = df.copy()
    for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    X_raw = _select_feature_cols(df_clean, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    X     = _CACHE["prep"].transform(X_raw)

    scores  = _score_X(X)
    tiers   = [_tier(float(s)) for s in scores]
    flagged = [bool(s >= _CACHE["flag_threshold"]) for s in scores]

    # SHAP reasons (row by row — TreeExplainer is fast for LightGBM)
    sv_all = _CACHE["explainer"].shap_values(X)
    if isinstance(sv_all, list):
        sv_all = sv_all[1]

    rows = []
    for i in range(len(df)):
        row_dict = df.iloc[i].to_dict()
        reasons = get_reason_codes(
            sv_all[i], _CACHE["feat_names"], X[i], k=3, raw_dict=row_dict
        )
        rows.append({
            "order_id":  order_ids[i],
            "risk_score": round(float(scores[i]), 6),
            "tier":       tiers[i],
            "flagged":    flagged[i],
            "reason_1":   reasons[0] if len(reasons) > 0 else "",
            "reason_2":   reasons[1] if len(reasons) > 1 else "",
            "reason_3":   reasons[2] if len(reasons) > 2 else "",
        })

    return pd.DataFrame(rows)


# ===========================================================================
# FastAPI-ready wrapper (drop-in example)
# ===========================================================================
# To expose as an API endpoint:
#
#   from fastapi import FastAPI
#   from predict import score_order
#
#   app = FastAPI()
#
#   @app.post("/score")
#   def score(order: dict) -> dict:
#       return score_order(order)
#
# ===========================================================================


# ===========================================================================
# CLI
# ===========================================================================

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="RTO Shield AI – score a single order from a JSON file."
    )
    parser.add_argument(
        "--json", required=True,
        help="Path to a JSON file containing one order dict.",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON response.",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    order_dict = json.loads(json_path.read_text())
    result     = score_order(order_dict)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    _cli()

