"""
src/03_calibrate_threshold.py
==============================
Step 4 – Calibration, Threshold Selection, Risk Tiers, Lift Analysis.

Picks the best tree model by validation ROC-AUC (from metrics CSV), then:

  1. Isotonic calibration fitted on VAL, evaluated on TEST.
     Outputs: outputs/calibration.png, models/calibrator.joblib

  2. Three threshold policies chosen on VAL, applied once to TEST:
       (a) Max-F1
       (b) Min-threshold with recall ≥ 0.80
       (c) Cost-minimising (cost_FN / cost_FP), with ratio sweep 1:1 → 5:1

  3. Risk tiers Low / Medium / High on TEST (cutoffs from VAL).
     Counts, actual late rate, share of all late orders per tier.

  4. Decile lift table + cumulative gain chart on TEST.
     Outputs: outputs/lift_gain.png, outputs/decile_table.csv

  5. Full-test metrics vs CASH-only subset.

  6. Thresholds and tier cutoffs → models/thresholds.json.

Usage
-----
  python src/03_calibrate_threshold.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.isotonic    import IsotonicRegression
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    TARGET_COL, ORDER_ID_COL, SEED,
    TRAIN_FRAC, VAL_FRAC,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES,
)
from features import build_feature_matrix
from utils   import compute_metrics, expected_calibration_error

# ---------------------------------------------------------------------------
# Cost configuration  (cost_FN : cost_FP)
# A missed late order (FN) costs more than a false alert (FP) in RTO context.
# ---------------------------------------------------------------------------
COST_CONFIG = {
    "cost_FN": 2,   # default: 2× more expensive to miss a late order
    "cost_FP": 1,
}
COST_RATIOS_SWEEP = [
    (1, 1),
    (2, 1),
    (3, 1),
    (5, 1),
]

DATE_COL      = "order_date"
ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"


# ===========================================================================
# Helpers
# ===========================================================================

def _sep(char="─", w=72): return char * w


def _chrono_split(df: pd.DataFrame):
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    n         = len(df)
    train_end = int(np.floor(n * TRAIN_FRAC))
    val_end   = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def _build_full_matrices(train_df, val_df, test_df):
    """Return (X_tr, X_val, X_te, y_tr, y_val, y_te) using saved preprocessor."""
    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
    from features import _select_feature_cols
    X_tr  = prep.transform(_select_feature_cols(train_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    X_val = prep.transform(_select_feature_cols(val_df,   CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    X_te  = prep.transform(_select_feature_cols(test_df,  CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    return (
        X_tr,  X_val,  X_te,
        train_df[TARGET_COL].values,
        val_df[TARGET_COL].values,
        test_df[TARGET_COL].values,
    )


# ===========================================================================
# 1. Pick best model
# ===========================================================================

def pick_best_model(metrics_csv: Path) -> tuple[str, float]:
    """
    Choose between XGBoost and LightGBM by validation ROC-AUC.
    Returns (model_name, val_roc_auc).
    """
    m = pd.read_csv(metrics_csv)
    candidates = m[
        m["model"].isin(["xgboost", "lightgbm"]) & (m["split"] == "val")
    ].set_index("model")["roc_auc"]

    best_name  = candidates.idxmax()
    best_score = candidates.max()
    other_name = [n for n in ["xgboost", "lightgbm"] if n != best_name][0]

    print(f"\n  {'Model':<12}  Val ROC-AUC")
    print(f"  {_sep('-', 28)}")
    for n in ["xgboost", "lightgbm"]:
        star = " ← WINNER" if n == best_name else ""
        print(f"  {n:<12}  {candidates[n]:.4f}{star}")
    print(f"\n  → Using [{best_name}] (val ROC-AUC={best_score:.4f} vs "
          f"{other_name} {candidates[other_name]:.4f})\n")

    return best_name, best_score


# ===========================================================================
# 2. Isotonic calibration
# ===========================================================================

def calibrate(model, X_val, y_val, X_te, y_te) -> tuple[IsotonicRegression, np.ndarray, np.ndarray]:
    """
    Fit IsotonicRegression on (val_proba, y_val).
    Returns (calibrator, raw_test_proba, calibrated_test_proba).
    """
    raw_val  = model.predict_proba(X_val)[:, 1]
    raw_te   = model.predict_proba(X_te)[:, 1]

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_val, y_val)

    cal_te = ir.predict(raw_te)
    return ir, raw_te, cal_te


def print_calibration_metrics(y_te, raw_proba, cal_proba) -> None:
    brier_raw = float(np.mean((raw_proba - y_te) ** 2))
    brier_cal = float(np.mean((cal_proba - y_te) ** 2))
    ece_raw   = expected_calibration_error(y_te, raw_proba)
    ece_cal   = expected_calibration_error(y_te, cal_proba)

    print(f"\n  {'Metric':<12}  {'Raw':>8}  {'Calibrated':>12}  {'Δ':>8}")
    print(f"  {_sep('-', 46)}")
    print(f"  {'Brier↓':<12}  {brier_raw:>8.4f}  {brier_cal:>12.4f}  {brier_cal-brier_raw:>+8.4f}")
    print(f"  {'ECE↓':<12}  {ece_raw:>8.4f}  {ece_cal:>12.4f}  {ece_cal-ece_raw:>+8.4f}")


def save_calibration_plot(y_te, raw_proba, cal_proba, out_path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, proba, label, color in [
        (axes[0], raw_proba, "Raw",        "#2196F3"),
        (axes[1], cal_proba, "Calibrated", "#4CAF50"),
    ]:
        # Reliability curve
        frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="uniform")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=6, label=label)
        ax.set_xlabel("Mean predicted probability", fontsize=11)
        ax.set_ylabel("Fraction of positives",      fontsize=11)
        ax.set_title(f"Reliability diagram – {label} ({model_name})", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        # Inset histogram
        ax_hist = ax.inset_axes([0.60, 0.07, 0.36, 0.28])
        ax_hist.hist(proba, bins=20, color=color, alpha=0.7, edgecolor="white", linewidth=0.4)
        ax_hist.set_xticks([0, 0.5, 1])
        ax_hist.tick_params(labelsize=7)
        ax_hist.set_title("Score dist.", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Calibration plot saved → {out_path}")


# ===========================================================================
# 3. Threshold selection (on VAL, applied to TEST)
# ===========================================================================

def _f1_at(y_true, proba, thr):
    yp = (proba >= thr).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _recall_at(y_true, proba, thr):
    yp = (proba >= thr).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _cost_at(y_true, proba, thr, cost_fn, cost_fp):
    yp = (proba >= thr).astype(int)
    fn = int(((yp == 0) & (y_true == 1)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum())
    return cost_fn * fn + cost_fp * fp


def select_thresholds(
    y_val:      np.ndarray,
    val_proba:  np.ndarray,   # calibrated val probabilities
    cost_fn:    int = 2,
    cost_fp:    int = 1,
    recall_min: float = 0.80,
) -> dict:
    """
    Choose three thresholds on VAL probabilities.
    Returns dict with keys: 'max_f1', 'recall_80', 'cost_default'.
    """
    thresholds = np.linspace(0.01, 0.99, 990)

    # (a) Max F1
    f1s   = [_f1_at(y_val, val_proba, t)[2] for t in thresholds]
    thr_f1 = float(thresholds[np.argmax(f1s)])

    # (b) Smallest threshold with recall >= recall_min
    thr_r80 = None
    for t in thresholds:
        if _recall_at(y_val, val_proba, t) >= recall_min:
            thr_r80 = float(t)
            break
    if thr_r80 is None:
        thr_r80 = thresholds[0]  # fallback: lowest threshold

    # (c) Cost-minimising (default ratio)
    costs    = [_cost_at(y_val, val_proba, t, cost_fn, cost_fp) for t in thresholds]
    thr_cost = float(thresholds[np.argmin(costs)])

    return {
        "max_f1":       round(thr_f1,  4),
        "recall_80":    round(thr_r80, 4),
        "cost_default": round(thr_cost,4),
    }


def apply_thresholds_on_test(
    y_te:       np.ndarray,
    te_proba:   np.ndarray,  # calibrated test probabilities
    y_val:      np.ndarray,
    val_proba:  np.ndarray,  # calibrated val probabilities
) -> dict:
    """
    Choose all thresholds on VAL, apply each once to TEST.
    Returns dict of results.
    """
    thresholds = np.linspace(0.01, 0.99, 990)

    # --- (a) Max-F1 on VAL ---
    f1s    = [_f1_at(y_val, val_proba, t)[2] for t in thresholds]
    thr_f1 = float(thresholds[np.argmax(f1s)])

    # --- (b) Smallest threshold with recall >= 0.80 on VAL ---
    thr_r80 = thresholds[0]
    for t in thresholds:
        if _recall_at(y_val, val_proba, t) >= 0.80:
            thr_r80 = float(t)
            break

    # --- (c) Cost sweep on VAL ---
    cost_results = {}
    for cost_fn, cost_fp in COST_RATIOS_SWEEP:
        costs    = [_cost_at(y_val, val_proba, t, cost_fn, cost_fp) for t in thresholds]
        thr_cost = float(thresholds[np.argmin(costs)])
        cost_results[(cost_fn, cost_fp)] = thr_cost

    all_thresholds = {
        "max_f1":    thr_f1,
        "recall_80": thr_r80,
        **{f"cost_{fn}_{fp}": thr for (fn, fp), thr in cost_results.items()},
    }

    return all_thresholds


def print_threshold_table(
    y_te:      np.ndarray,
    te_proba:  np.ndarray,
    y_val:     np.ndarray,
    val_proba: np.ndarray,
) -> dict:
    """Print threshold comparison table on TEST, return threshold dict."""
    thresholds = np.linspace(0.01, 0.99, 990)

    # (a) Max-F1
    f1s    = [_f1_at(y_val, val_proba, t)[2] for t in thresholds]
    thr_f1 = float(thresholds[np.argmax(f1s)])
    val_f1_at_opt = max(f1s)

    # (b) Min-threshold recall >= 0.80
    thr_r80 = thresholds[0]
    for t in thresholds:
        if _recall_at(y_val, val_proba, t) >= 0.80:
            thr_r80 = float(t)
            break

    # (c) Cost sweep
    cost_rows = []
    for cost_fn, cost_fp in COST_RATIOS_SWEEP:
        costs    = [_cost_at(y_val, val_proba, t, cost_fn, cost_fp) for t in thresholds]
        thr_cost = float(thresholds[np.argmin(costs)])
        pr, rc, f1 = _f1_at(y_te, te_proba, thr_cost)
        cost_rows.append({
            "ratio": f"FN:FP={cost_fn}:{cost_fp}",
            "threshold": thr_cost,
            "precision": pr, "recall": rc, "f1": f1,
        })

    # Print
    print(f"\n  {'Policy':<26}  {'Thr':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {_sep('-', 65)}")

    p, r, f1 = _f1_at(y_te, te_proba, thr_f1)
    print(f"  {'(a) Max-F1 (val)':<26}  {thr_f1:>6.4f}  {p:>10.4f}  {r:>8.4f}  {f1:>8.4f}")

    p, r, f1 = _f1_at(y_te, te_proba, thr_r80)
    print(f"  {'(b) Recall≥0.80 (val)':<26}  {thr_r80:>6.4f}  {p:>10.4f}  {r:>8.4f}  {f1:>8.4f}")

    for row in cost_rows:
        label = f"(c) Cost {row['ratio']}"
        print(
            f"  {label:<26}  {row['threshold']:>6.4f}  "
            f"{row['precision']:>10.4f}  {row['recall']:>8.4f}  {row['f1']:>8.4f}"
        )

    # Build export dict
    thr_dict = {
        "max_f1":    round(thr_f1,  4),
        "recall_80": round(thr_r80, 4),
    }
    for row in cost_rows:
        key = "cost_" + row["ratio"].replace("FN:FP=", "").replace(":", "_")
        thr_dict[key] = round(row["threshold"], 4)

    return thr_dict


# ===========================================================================
# 4. Risk tiers (cutoffs from VAL)
# ===========================================================================

def define_tiers(val_proba: np.ndarray) -> tuple[float, float]:
    """
    Define Low / Medium / High tier cutoffs from VAL score distribution.

    Strategy: use the 33rd and 67th percentiles of the CALIBRATED val scores
    as the cutoffs.  This yields roughly equal-sized buckets on the validation
    set while being derived purely from VAL (no test look-ahead).

    Justification: percentile-based cuts are robust to score distribution
    shifts and don't require a specific threshold search, making them stable
    across retraining runs.
    """
    low_cutoff  = float(np.percentile(val_proba, 33))
    high_cutoff = float(np.percentile(val_proba, 67))
    return low_cutoff, high_cutoff


def print_tier_analysis(
    y_te:         np.ndarray,
    te_proba:     np.ndarray,
    low_cut:      float,
    high_cut:     float,
) -> None:
    tiers = np.where(
        te_proba < low_cut, "Low",
        np.where(te_proba < high_cut, "Medium", "High")
    )
    total_late = y_te.sum()

    print(f"\n  Tier cutoffs (from VAL p33={low_cut:.4f}, p67={high_cut:.4f})")
    print(f"    Low    : score < {low_cut:.4f}")
    print(f"    Medium : {low_cut:.4f} ≤ score < {high_cut:.4f}")
    print(f"    High   : score ≥ {high_cut:.4f}")

    print(f"\n  {'Tier':<8}  {'Orders':>8}  {'Share%':>7}  {'Late rate':>10}  {'Late captured%':>15}")
    print(f"  {_sep('-', 54)}")
    for tier in ["Low", "Medium", "High"]:
        mask      = tiers == tier
        n         = mask.sum()
        late_n    = y_te[mask].sum()
        late_rate = late_n / n if n > 0 else 0.0
        captured  = late_n / total_late if total_late > 0 else 0.0
        pct_share = n / len(y_te) * 100
        print(
            f"  {tier:<8}  {n:>8,}  {pct_share:>6.1f}%  "
            f"{late_rate:>10.4f}  {captured*100:>14.1f}%"
        )


# ===========================================================================
# 5. Decile lift table + cumulative gain chart
# ===========================================================================

def compute_decile_table(y_te: np.ndarray, te_proba: np.ndarray) -> pd.DataFrame:
    """
    Sort by descending score, bin into 10 deciles, compute lift per decile.
    """
    df = pd.DataFrame({"score": te_proba, "label": y_te})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    n         = len(df)
    base_rate = y_te.mean()
    decile_size = n // 10

    rows = []
    cum_late  = 0
    cum_total = 0

    for d in range(10):
        lo = d * decile_size
        hi = lo + decile_size if d < 9 else n
        chunk = df.iloc[lo:hi]

        late_in_decile = int(chunk["label"].sum())
        n_decile       = len(chunk)
        cum_late      += late_in_decile
        cum_total     += n_decile

        late_rate = late_in_decile / n_decile if n_decile > 0 else 0.0
        lift      = late_rate / base_rate if base_rate > 0 else 0.0
        cum_rate  = cum_late / cum_total if cum_total > 0 else 0.0
        cum_lift  = cum_rate / base_rate if base_rate > 0 else 0.0
        cum_gain  = cum_late / y_te.sum() * 100 if y_te.sum() > 0 else 0.0

        rows.append({
            "decile":           d + 1,
            "n":                n_decile,
            "late_in_decile":   late_in_decile,
            "late_rate":        round(late_rate, 4),
            "lift":             round(lift, 4),
            "cum_late":         cum_late,
            "cum_lift":         round(cum_lift, 4),
            "cum_gain_pct":     round(cum_gain, 2),
        })

    return pd.DataFrame(rows)


def print_decile_table(dt: pd.DataFrame) -> None:
    hdr = (
        f"  {'Decile':>6}  {'N':>6}  {'Late':>6}  "
        f"{'LateRate':>9}  {'Lift':>6}  {'CumLift':>8}  {'CumGain%':>9}"
    )
    print(hdr)
    print(f"  {_sep('-', 65)}")
    for _, row in dt.iterrows():
        print(
            f"  {int(row['decile']):>6}  {int(row['n']):>6,}  {int(row['late_in_decile']):>6,}  "
            f"{row['late_rate']:>9.4f}  {row['lift']:>6.3f}  "
            f"{row['cum_lift']:>8.3f}  {row['cum_gain_pct']:>8.1f}%"
        )


def save_lift_gain_plot(dt: pd.DataFrame, y_te: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

    # -- Left: Lift by decile ------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, 10))   # type: ignore
    bars = ax1.bar(dt["decile"], dt["lift"], color=colors, edgecolor="white", linewidth=0.6)
    ax1.axhline(1.0, color="grey", ls="--", lw=1.2, label="Baseline (lift=1)")
    ax1.set_xlabel("Score decile (1 = highest risk)", fontsize=11)
    ax1.set_ylabel("Lift", fontsize=11)
    ax1.set_title("Decile Lift Chart – TEST set", fontsize=12)
    ax1.set_xticks(dt["decile"])
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, dt["lift"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # -- Right: Cumulative gain ----------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    xs  = [0] + list(dt["decile"] / 10 * 100)
    ys  = [0] + list(dt["cum_gain_pct"])
    ax2.plot(xs, ys, "o-", color="#1976D2", lw=2.2, ms=6, label="Model cumulative gain")
    ax2.plot([0, 100], [0, 100], "k--", lw=1.2, alpha=0.5, label="Random baseline")
    ax2.fill_between(xs, ys, [0] + list(np.linspace(0, 100, 10)),
                     alpha=0.12, color="#1976D2")

    # Perfect model line
    pos_rate = y_te.mean()
    perfect_x = [0, pos_rate * 100, 100]
    perfect_y = [0, 100, 100]
    ax2.plot(perfect_x, perfect_y, "g--", lw=1.2, alpha=0.6, label="Perfect model")

    ax2.set_xlabel("% of orders reviewed (highest risk first)", fontsize=11)
    ax2.set_ylabel("% of late orders captured",                 fontsize=11)
    ax2.set_title("Cumulative Gain Chart – TEST set",           fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 102)
    ax2.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Lift/gain chart saved → {out_path}")


# ===========================================================================
# 6. CASH-only subset metrics
# ===========================================================================

def print_cash_vs_full(
    y_te:       np.ndarray,
    te_proba:   np.ndarray,
    test_df:    pd.DataFrame,
    threshold:  float,
    label:      str,
) -> None:
    """Compare full-test metrics vs CASH (COD proxy) subset at given threshold."""
    is_cash = (test_df["Type"] == "CASH").values

    rows = []
    for name, mask in [("Full TEST", np.ones(len(y_te), dtype=bool)), ("CASH only", is_cash)]:
        y_sub    = y_te[mask]
        prob_sub = te_proba[mask]
        m = compute_metrics(y_sub, prob_sub, threshold=threshold)
        rows.append({"subset": name, "n": int(mask.sum()), **m})

    df = pd.DataFrame(rows)

    print(f"\n  Threshold = {threshold:.4f}  ({label})")
    print(f"  {'Subset':<12}  {'N':>6}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>8}  {'ROC-AUC':>8}  {'PR-AUC':>8}  {'Brier↓':>8}")
    print(f"  {_sep('-', 76)}")
    for _, row in df.iterrows():
        print(
            f"  {row['subset']:<12}  {row['n']:>6,}  "
            f"{row['precision']:>10.4f}  {row['recall']:>8.4f}  "
            f"{row['f1']:>8.4f}  {row['roc_auc']:>8.4f}  "
            f"{row['pr_auc']:>8.4f}  {row['brier']:>8.4f}"
        )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 72)
    print("  RTO Shield AI – Step 4: Calibration, Thresholds, Tiers & Lift")
    print("=" * 72)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/7] Loading orders.parquet and splitting …")
    orders = pd.read_parquet(ORDER_PARQUET)
    train_df, val_df, test_df = _chrono_split(orders)
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # ── Build feature matrices ────────────────────────────────────────────────
    print("\n[2/7] Building feature matrices (using saved preprocessor) …")
    _, X_val, X_te, _, y_val, y_te = _build_full_matrices(train_df, val_df, test_df)

    # ── Pick best model ───────────────────────────────────────────────────────
    print("\n[3/7] Selecting best tree model by validation ROC-AUC …")
    metrics_csv = OUTPUTS_DIR / "metrics_model_comparison.csv"
    best_name, best_val_roc = pick_best_model(metrics_csv)
    model = joblib.load(MODELS_DIR / f"{best_name}.joblib")

    # ── Calibrate ────────────────────────────────────────────────────────────
    print(f"\n[4/7] Isotonic calibration on VAL, evaluated on TEST …")
    ir, raw_te, cal_te = calibrate(model, X_val, y_val, X_te, y_te)
    cal_val = ir.predict(model.predict_proba(X_val)[:, 1])

    print(f"\n  {'=' * 50}")
    print(f"  Calibration quality comparison (TEST set)")
    print(f"  {'=' * 50}")
    print_calibration_metrics(y_te, raw_te, cal_te)

    save_calibration_plot(
        y_te, raw_te, cal_te,
        OUTPUTS_DIR / "calibration.png",
        best_name,
    )
    joblib.dump(ir, MODELS_DIR / "calibrator.joblib")
    print(f"  Calibrator saved → {MODELS_DIR}/calibrator.joblib")

    # ── Threshold selection ───────────────────────────────────────────────────
    print(f"\n[5/7] Threshold selection (chosen on VAL, applied once to TEST) …")
    print(f"\n  {'=' * 70}")
    print(f"  Threshold policies on TEST set (calibrated probabilities)")
    print(f"  {'=' * 70}")
    thr_dict = print_threshold_table(y_te, cal_te, y_val, cal_val)

    # ── Risk tiers ────────────────────────────────────────────────────────────
    print(f"\n[6/7] Risk tier analysis (cutoffs from VAL) …")
    print(f"\n  {'=' * 70}")
    print(f"  Risk Tiers on TEST set")
    print(f"  {'=' * 70}")
    low_cut, high_cut = define_tiers(cal_val)
    print_tier_analysis(y_te, cal_te, low_cut, high_cut)

    # ── Decile lift ───────────────────────────────────────────────────────────
    print(f"\n[7/7] Decile lift table and cumulative gain chart …")
    print(f"\n  {'=' * 70}")
    print(f"  Decile Lift Table – TEST set (calibrated scores, descending)")
    print(f"  {'=' * 70}")
    dt = compute_decile_table(y_te, cal_te)
    print_decile_table(dt)

    dt.to_csv(OUTPUTS_DIR / "decile_table.csv", index=False)
    print(f"\n  Decile table saved → {OUTPUTS_DIR}/decile_table.csv")

    save_lift_gain_plot(dt, y_te, OUTPUTS_DIR / "lift_gain.png")

    # ── CASH-only subset ──────────────────────────────────────────────────────
    print(f"\n  {'=' * 70}")
    print(f"  CASH (COD proxy) subset vs Full TEST – calibrated probabilities")
    print(f"  {'=' * 70}")
    for policy_label, thr_key in [
        ("Max-F1 threshold",       "max_f1"),
        ("Recall≥0.80 threshold",  "recall_80"),
        ("Cost 2:1 threshold",     "cost_2_1"),
    ]:
        thr_val = thr_dict.get(thr_key, 0.5)
        print_cash_vs_full(y_te, cal_te, test_df, thr_val, policy_label)

    # ── Save thresholds + tier cutoffs ────────────────────────────────────────
    export = {
        "best_model":        best_name,
        "best_val_roc_auc":  round(best_val_roc, 6),
        "thresholds":        {k: round(v, 4) for k, v in thr_dict.items()},
        "tier_cutoffs": {
            "low_max":   round(low_cut,  4),
            "high_min":  round(high_cut, 4),
            "note": (
                "Low: score < low_max  |  "
                "Medium: low_max ≤ score < high_min  |  "
                "High: score ≥ high_min  |  "
                "Cutoffs = p33 and p67 of calibrated VAL scores"
            ),
        },
        "cost_config": COST_CONFIG,
    }
    thr_json = MODELS_DIR / "thresholds.json"
    thr_json.write_text(json.dumps(export, indent=2))
    print(f"\n  Thresholds + tier cutoffs saved → {thr_json}")

    print("\n" + "=" * 72)
    print("  Calibration & threshold analysis complete.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

