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
          "tier_description": str,       # relative-language text from thresholds.json
          "flagged":     bool,           # True if risk_score >= flag_threshold (= High tier)
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

import threading

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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
_LOAD_LOCK = threading.Lock()   # the dashboard warms the model in a background thread

_THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
_MODEL_PATH      = MODELS_DIR / "final_model.joblib"
_PREP_PATH       = MODELS_DIR / "preprocessor.joblib"
_CAL_PATH        = MODELS_DIR / "final_calibrator.joblib"
if not _CAL_PATH.exists():
    _CAL_PATH = MODELS_DIR / "calibrator.joblib"


def format_score_clipped(score: float) -> str:
    """Format risk score as a percentage clipped at <1% and >99%."""
    if score < 0.01:
        return "<1%"
    elif score > 0.99:
        return ">99%"
    else:
        return f"{score * 100:.1f}%"


# ===========================================================================
# Lazy loader
# ===========================================================================

def _load_artifacts() -> None:
    """Load all artifacts into module cache on first call (thread-safe; ``import shap`` is lazy: ~2 s)."""
    if _CACHE.get("explainer") is not None:
        return
    with _LOAD_LOCK:
        if _CACHE.get("explainer") is not None:
            return
        import shap   # lazy: only needed for explanations, and slow to import

        local: dict[str, Any] = {}
        local["model"] = joblib.load(_MODEL_PATH)
        local["prep"]  = joblib.load(_PREP_PATH)
        local["cal"]   = joblib.load(_CAL_PATH)

        thr_data = json.loads(_THRESHOLDS_PATH.read_text(encoding="utf-8"))
        local["low_cut"]  = float(thr_data["tier_cutoffs"]["low_max"])
        local["high_cut"] = float(thr_data["tier_cutoffs"]["high_min"])
        # Flagged defined strictly as High tier (score >= high_min)
        local["flag_threshold"] = local["high_cut"]
        # Relative-language tier text (with real held-out numbers) lives in thresholds.json
        local["tier_desc"] = {k: v["description"] for k, v in thr_data.get("tier_descriptions", {}).items()}
        local["feat_names"] = list(local["prep"].get_feature_names_out())

        # SHAP explainer (expensive — created once); set last so readers never see a half-filled cache
        local["explainer"] = shap.TreeExplainer(local["model"])
        _CACHE.update(local)


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


def _raw_X(X: np.ndarray) -> np.ndarray:
    """Return RAW (uncalibrated) model scores - used for ranking metrics."""
    return _CACHE["model"].predict_proba(X)[:, 1]


def _transform(X_raw: pd.DataFrame) -> np.ndarray:
    """
    Apply the fitted preprocessor.  Orders without a country / region / product category get the placeholder
    category "missing", which the one-hot encoder (handle_unknown="ignore") maps to all zeros BY DESIGN, so
    sklearn's "Found unknown categories" notice is expected and suppressed (only that message).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories")
        return _CACHE["prep"].transform(X_raw)


def _shap_values(X: np.ndarray) -> np.ndarray:
    """SHAP values (rows x encoded features) in log-odds of the positive class, via the modern Explanation API."""
    sv = np.asarray(_CACHE["explainer"](X).values)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def _shap_reasons(X_row: np.ndarray, raw_dict: dict | None = None) -> list[str]:
    """
    Compute SHAP values for one transformed row and return 3 reason codes.
    """
    shap_flat = _shap_values(X_row)[0]
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
    X        = _transform(X_raw)

    score   = float(_score_X(X)[0])
    tier_   = _tier(score)
    flagged = score >= _CACHE["flag_threshold"]
    reasons = _shap_reasons(X, raw_dict=order_dict)

    return {
        "order_id":      order_id,
        "risk_score":    round(score, 6),
        "display_score": format_score_clipped(score),
        "tier":          tier_,
        "tier_description": _CACHE["tier_desc"].get(tier_, ""),
        "flagged":       flagged,
        "top_reasons":   reasons,
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
        order_id, risk_score, raw_score, display_score, tier, tier_description, flagged,
        reason_1, reason_2, reason_3
    """
    _load_artifacts()

    order_ids = df.get("Order Id", pd.Series([None] * len(df))).tolist()

    # Ensure all feature cols exist
    df_clean = df.copy()
    for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    X_raw = _select_feature_cols(df_clean, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    X     = _transform(X_raw)

    raw_scores = _raw_X(X)
    scores  = _CACHE["cal"].predict(raw_scores)
    tiers   = [_tier(float(s)) for s in scores]
    flagged = [bool(s >= _CACHE["flag_threshold"]) for s in scores]

    # SHAP reasons (row by row — TreeExplainer is fast for LightGBM)
    sv_all = _shap_values(X)

    rows = []
    for i in range(len(df)):
        row_dict = df.iloc[i].to_dict()
        reasons = get_reason_codes(
            sv_all[i], _CACHE["feat_names"], X[i], k=3, raw_dict=row_dict
        )
        s_val = float(scores[i])
        rows.append({
            "order_id":      order_ids[i],
            "risk_score":    round(s_val, 6),
            "raw_score":     round(float(raw_scores[i]), 6),
            "display_score": format_score_clipped(s_val),
            "tier":          tiers[i],
            "tier_description": _CACHE["tier_desc"].get(tiers[i], ""),
            "flagged":       flagged[i],
            "reason_1":      reasons[0] if len(reasons) > 0 else "",
            "reason_2":      reasons[1] if len(reasons) > 1 else "",
            "reason_3":      reasons[2] if len(reasons) > 2 else "",
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Batch evaluation (labelled batches)
# ===========================================================================

SMALL_SAMPLE_N = 200   # fewer labelled rows than this -> "small sample" warning
MIN_TIER_N = 30        # tiers with fewer orders get no calibration gap


def _boot_ci(y: np.ndarray, fn, arrays: list[np.ndarray], n_boot: int, seed: int):
    """95% percentile bootstrap CI of fn(y_b, *arrays_b); skips one-class resamples."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        yb = y[b]
        if yb.min() == yb.max():
            continue
        vals.append(fn(yb, *[a[b] for a in arrays]))
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate_batch(
    y_true,
    raw_scores,
    cal_scores,
    tiers,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Evaluate a LABELLED batch scored by score_batch().

    Ranking metrics (ROC-AUC, PR-AUC) use the RAW scores; Brier and the tier
    calibration table use the calibrated scores.

    * n < SMALL_SAMPLE_N (200): ``small_sample`` is True and a warning is
      attached; read the 95% bootstrap intervals, not the point estimates.
    * Tiers with fewer than MIN_TIER_N (30) orders get ``gap = None`` and
      ``suppressed = True`` (a gap on so few orders is just noise).
    """
    y = np.asarray(y_true).astype(int)
    raw = np.asarray(raw_scores, dtype=float)
    cal = np.asarray(cal_scores, dtype=float)
    tiers = np.asarray(tiers)
    n = int(len(y))
    boots = n_boot if n <= 20000 else 200

    out: dict = {"n": n, "prevalence": float(y.mean()) if n else None,
                 "small_sample": n < SMALL_SAMPLE_N, "warning": None,
                 "roc_auc": None, "roc_auc_ci": (None, None),
                 "pr_auc": None, "pr_auc_ci": (None, None),
                 "brier": None, "brier_ci": (None, None), "tier_calibration": []}
    if n < SMALL_SAMPLE_N:
        out["warning"] = (f"Small sample: only {n} labelled rows (< {SMALL_SAMPLE_N}). Metrics are noisy - "
                          "read the 95% bootstrap intervals, not the point estimates.")
    if n == 0:
        return out

    out["brier"] = float(np.mean((cal - y) ** 2))
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, raw))
        out["pr_auc"] = float(average_precision_score(y, raw))
        out["roc_auc_ci"] = _boot_ci(y, roc_auc_score, [raw], boots, seed)
        out["pr_auc_ci"] = _boot_ci(y, average_precision_score, [raw], boots, seed)
        out["brier_ci"] = _boot_ci(y, lambda yb, cb: float(np.mean((cb - yb) ** 2)), [cal], boots, seed)

    for t in ["Low", "Medium", "High"]:
        m = tiers == t
        k = int(m.sum())
        suppressed = k < MIN_TIER_N
        mean_p = float(cal[m].mean()) if k else None
        act = float(y[m].mean()) if k else None
        out["tier_calibration"].append({
            "tier": t, "n": k, "share": k / n, "mean_predicted": mean_p, "actual_late_rate": act,
            "gap": None if (suppressed or k == 0) else mean_p - act,
            "suppressed": suppressed,
        })
    return out


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

