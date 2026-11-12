"""
src/03_calibrate_threshold.py
==============================
Step 4 - Calibration review, cluster-aware thresholds and risk tiers for the
FINAL tuned LightGBM model.

Must run AFTER 05_tune_multiseed.py (which produces final_model.joblib and
final_calibrator.joblib).  Pipeline order: 01, 02, 05, 03, 04, 06, ...

What it does
------------
  1. Loads the FINAL model + isotonic calibrator (no re-fitting).
  2. Max-F1 threshold and cost-sweep thresholds chosen on VALIDATION.
  3. Risk tiers (Low / Medium / High): cutoffs are midpoints of low-density
     gaps between score clusters, chosen on VALIDATION (see tiering.py) and
     verified on TEST.  Flagged = High tier.
  4. Precision >= 0.80 threshold: smallest low-density gap midpoint whose
     VALIDATION precision >= 0.80.
  5. Stability table on TEST for threshold +/-0.01 and +/-0.02; the script
     FAILS LOUDLY if the flagged share moves by more than 5 points.
  6. Decile lift table + gain chart on TEST (final model).
  7. Writes models/thresholds.json (cutoffs + relative-language tier text).

Usage:  python src/03_calibrate_threshold.py
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

from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    TARGET_COL, TRAIN_FRAC, VAL_FRAC,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)
from features import _select_feature_cols  # noqa: E402
from utils import compute_metrics, expected_calibration_error  # noqa: E402
import tiering  # noqa: E402

COST_RATIOS_SWEEP = [(1, 1), (2, 1), (3, 1), (5, 1)]
COST_CONFIG = {"cost_FN": 2, "cost_FP": 1}
DATE_COL = "order_date"
ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"

TIER_ACTIONS = {
    "Low": "Standard dispatch with routine monitoring.",
    "Medium": "Monitor the carrier pickup SLA.",
    "High": "Operational flag: review routing and confirm proactively.",
}
TIER_LABELS = {"Low": "Lower risk", "Medium": "Elevated risk", "High": "Highest risk"}


def _sep(char="-", w=72):
    return char * w


# ===========================================================================
# Loading
# ===========================================================================
def load_everything():
    orders = pd.read_parquet(ORDER_PARQUET).sort_values(DATE_COL).reset_index(drop=True)
    n = len(orders)
    tr_end = int(np.floor(n * TRAIN_FRAC))
    val_end = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    val_df = orders.iloc[tr_end:val_end].copy()
    test_df = orders.iloc[val_end:].copy()

    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    cal = joblib.load(MODELS_DIR / "final_calibrator.joblib")

    def _scores(df):
        X = prep.transform(_select_feature_cols(df, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
        raw = model.predict_proba(X)[:, 1]
        return raw, cal.predict(raw)

    raw_val, cal_val = _scores(val_df)
    raw_te, cal_te = _scores(test_df)
    return (val_df, test_df, raw_val, cal_val, raw_te, cal_te,
            val_df[TARGET_COL].values, test_df[TARGET_COL].values)


# ===========================================================================
# Calibration review (final model)
# ===========================================================================
def print_calibration_metrics(y_te, raw_proba, cal_proba) -> None:
    b_raw = float(np.mean((raw_proba - y_te) ** 2))
    b_cal = float(np.mean((cal_proba - y_te) ** 2))
    e_raw = expected_calibration_error(y_te, raw_proba)
    e_cal = expected_calibration_error(y_te, cal_proba)
    print(f"\n  {'Metric':<10}  {'Raw':>8}  {'Calibrated':>11}")
    print(f"  {_sep('-', 34)}")
    print(f"  {'Brier':<10}  {b_raw:>8.4f}  {b_cal:>11.4f}")
    print(f"  {'ECE':<10}  {e_raw:>8.4f}  {e_cal:>11.4f}")


def save_calibration_plot(y_te, raw_proba, cal_proba, out_path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, proba, label, color in [
        (axes[0], raw_proba, "Raw", "#2196F3"),
        (axes[1], cal_proba, "Calibrated", "#4CAF50"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="uniform")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=6, label=label)
        ax.set_xlabel("Mean predicted probability", fontsize=11)
        ax.set_ylabel("Fraction of positives", fontsize=11)
        ax.set_title(f"Reliability diagram - {label} ({model_name})", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax_hist = ax.inset_axes([0.60, 0.07, 0.36, 0.28])
        ax_hist.hist(proba, bins=20, color=color, alpha=0.7, edgecolor="white", linewidth=0.4)
        ax_hist.set_xticks([0, 0.5, 1])
        ax_hist.tick_params(labelsize=7)
        ax_hist.set_title("Score dist.", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Calibration plot saved -> {out_path}")


# ===========================================================================
# Threshold helpers
# ===========================================================================
def _f1_at(y, p, thr):
    yp = (p >= thr).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)


def _cost_at(y, p, thr, cfn, cfp):
    yp = (p >= thr).astype(int)
    return cfn * int(((yp == 0) & (y == 1)).sum()) + cfp * int(((yp == 1) & (y == 0)).sum())


def select_other_thresholds(y_val, cal_val) -> dict:
    """Max-F1 and cost-minimising thresholds, chosen on VALIDATION."""
    grid = np.linspace(0.01, 0.99, 990)
    f1s = [_f1_at(y_val, cal_val, t)[2] for t in grid]
    out = {"max_f1": round(float(grid[int(np.argmax(f1s))]), 4)}
    for cfn, cfp in COST_RATIOS_SWEEP:
        costs = [_cost_at(y_val, cal_val, t, cfn, cfp) for t in grid]
        out[f"cost_{cfn}_{cfp}"] = round(float(grid[int(np.argmin(costs))]), 4)
    return out


# ===========================================================================
# Tiers
# ===========================================================================
def tier_assign(scores, low, high):
    return np.where(scores < low, "Low", np.where(scores < high, "Medium", "High"))


def tier_stats(y, scores, low, high) -> dict:
    tiers = tier_assign(scores, low, high)
    total_late = int(y.sum())
    out = {}
    for t in ["Low", "Medium", "High"]:
        m = tiers == t
        n = int(m.sum())
        late = int(y[m].sum())
        out[t] = {
            "orders": n,
            "share": n / len(y),
            "late_orders": late,
            "late_rate": late / n if n else 0.0,
            "late_captured": late / total_late if total_late else 0.0,
            "mean_score": float(scores[m].mean()) if n else 0.0,
        }
    return out


def tier_descriptions(test_stats: dict, base_rate: float) -> dict:
    desc = {}
    for t, st in test_stats.items():
        desc[t] = {
            "label": TIER_LABELS[t],
            "description": (f"{TIER_LABELS[t]}: {st['late_rate'] * 100:.0f}% of these orders were late "
                            f"in held-out test data (overall: {base_rate * 100:.0f}%)."),
            "action": TIER_ACTIONS[t],
        }
    return desc


# ===========================================================================
# Decile lift
# ===========================================================================
def compute_decile_table(y, p) -> pd.DataFrame:
    df = pd.DataFrame({"score": p, "label": y}).sort_values("score", ascending=False).reset_index(drop=True)
    n, base = len(df), y.mean()
    size = n // 10
    rows, cum_late, cum_total = [], 0, 0
    for d in range(10):
        lo = d * size
        hi = lo + size if d < 9 else n
        chunk = df.iloc[lo:hi]
        late = int(chunk["label"].sum())
        cum_late += late
        cum_total += len(chunk)
        rate = late / len(chunk)
        rows.append({
            "decile": d + 1, "n": len(chunk), "late_in_decile": late,
            "late_rate": round(rate, 4), "lift": round(rate / base, 4),
            "cum_late": cum_late, "cum_lift": round(cum_late / cum_total / base, 4),
            "cum_gain_pct": round(cum_late / y.sum() * 100, 2),
        })
    return pd.DataFrame(rows)


def save_lift_gain_plot(dt: pd.DataFrame, y, out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, 10))
    bars = ax1.bar(dt["decile"], dt["lift"], color=colors, edgecolor="white", linewidth=0.6)
    ax1.axhline(1.0, color="grey", ls="--", lw=1.2, label="Baseline (lift=1)")
    ax1.set_xlabel("Score decile (1 = highest risk)")
    ax1.set_ylabel("Lift")
    ax1.set_title("Decile Lift Chart - TEST set")
    ax1.set_xticks(dt["decile"])
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, dt["lift"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax2 = fig.add_subplot(gs[1])
    xs = [0] + list(dt["decile"] / 10 * 100)
    ys = [0] + list(dt["cum_gain_pct"])
    ax2.plot(xs, ys, "o-", color="#1976D2", lw=2.2, ms=6, label="Model cumulative gain")
    ax2.plot([0, 100], [0, 100], "k--", lw=1.2, alpha=0.5, label="Random baseline")
    ax2.plot([0, y.mean() * 100, 100], [0, 100, 100], "g--", lw=1.2, alpha=0.6, label="Perfect model")
    ax2.set_xlabel("% of orders reviewed (highest risk first)")
    ax2.set_ylabel("% of late orders captured")
    ax2.set_title("Cumulative Gain Chart - TEST set")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 102)
    ax2.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Lift/gain chart saved -> {out_path}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print("\n" + "=" * 72)
    print("  RTO Shield AI - Step 4: Cluster-aware thresholds & tiers (FINAL model)")
    print("=" * 72)

    (val_df, test_df, raw_val, cal_val, raw_te, cal_te, y_val, y_te) = load_everything()
    print(f"\n  val={len(val_df):,}  test={len(test_df):,}  "
          f"unique calibrated scores: val={len(np.unique(cal_val))}  test={len(np.unique(cal_te))}")

    print("\n[1/6] Calibration of the final model (TEST)")
    print_calibration_metrics(y_te, raw_te, cal_te)
    save_calibration_plot(y_te, raw_te, cal_te, OUTPUTS_DIR / "calibration.png", "final tuned LightGBM")

    print("\n[2/6] Candidate cutoffs (midpoints of low-density gaps, from VALIDATION)")
    low_cut, high_cut, cands = tiering.select_tier_cutoffs(cal_val)
    print(f"  Candidates: {cands}")
    print(f"  Tier cutoffs -> low_max={low_cut:.4f}  high_min={high_cut:.4f}")

    p80 = tiering.select_precision_threshold(cal_val, y_val, cands, 0.80)
    other = select_other_thresholds(y_val, cal_val)
    max_f1 = other["max_f1"]
    print(f"  Precision>=0.80 threshold (smallest qualifying gap midpoint): {p80:.4f}")
    print(f"  Max-F1 threshold (validation): {max_f1:.4f}")

    print("\n[3/6] Verify every cutoff sits in a gap on VALIDATION and TEST")
    checks = {}
    print(f"  {'cutoff':<18}{'value':>8}  {'split':>5}  {'val +/-0.01':>12}  {'test +/-0.01':>13}  "
          f"{'test +/-0.02':>13}  test gap")
    for name, c in [("tier low_max", low_cut), ("tier high_min", high_cut), ("precision_80", p80)]:
        cv = tiering.cutoff_check(cal_val, c)
        ct = tiering.cutoff_check(cal_te, c)
        checks[name] = {"validation": cv, "test": ct}
        print(f"  {name:<18}{c:>8.4f}  {ct['big_clusters_split']:>5}  {cv['share_within_0.01']:>11.2%}  "
              f"{ct['share_within_0.01']:>12.2%}  {ct['share_within_0.02']:>12.2%}  "
              f"{'in gap' if ct['in_gap'] else 'INSIDE CLUSTER'}")
        if ct["big_clusters_split"] or not ct["in_gap"]:
            raise tiering.CutoffError(
                f"Cutoff '{name}'={c} is not in a low-density gap on TEST scores "
                f"(inside_cluster={ct['inside_cluster']}, big_clusters_split={ct['big_clusters_split']}).")
        if ct["share_within_0.02"] > tiering.MAX_WINDOW_MASS:
            raise tiering.CutoffError(
                f"Cutoff '{name}'={c} has {ct['share_within_0.02']:.1%} of TEST orders within +/-0.02.")

    print("\n[4/6] Stability on TEST (threshold shifted by +/-0.01, +/-0.02)")
    pol_thr = {"High tier": high_cut, "Precision>=0.80": p80, "Max-F1": max_f1}
    stability, swings = [], {}
    for name, thr in pol_thr.items():
        rows, swing = tiering.stability_table(cal_te, y_te, thr, name)
        stability += rows
        swings[name] = swing
    sdf = pd.DataFrame(stability)
    print(f"  {'policy':<16}{'shift':>7}{'thr':>9}{'flagged%':>10}{'precision':>11}{'recall':>8}")
    for _, r in sdf.iterrows():
        print(f"  {r['policy']:<16}{r['shift']:>+7.2f}{r['threshold']:>9.4f}"
              f"{r['flagged_share_pct']:>10.2f}{r['precision']:>11.4f}{r['recall']:>8.4f}")
    for name, sw in swings.items():
        print(f"  swing[{name}] = {sw:.2f} pp (limit {tiering.MAX_FLAG_SWING_PP})")
    bad = {k: v for k, v in swings.items() if v > tiering.MAX_FLAG_SWING_PP}
    if bad:
        raise tiering.CutoffError(
            "UNSTABLE THRESHOLD(S): flagged share moves by more than "
            f"{tiering.MAX_FLAG_SWING_PP} points for a +/-0.02 shift: {bad}")

    print("\n[5/6] Risk tiers")
    val_stats = tier_stats(y_val, cal_val, low_cut, high_cut)
    te_stats = tier_stats(y_te, cal_te, low_cut, high_cut)
    base_rate = float(y_te.mean())
    desc = tier_descriptions(te_stats, base_rate)
    print(f"  {'Tier':<8}{'Orders':>8}{'Share':>8}{'Late rate':>11}{'Late captured':>15}   (TEST)")
    for t in ["Low", "Medium", "High"]:
        s = te_stats[t]
        print(f"  {t:<8}{s['orders']:>8,}{s['share']:>8.1%}{s['late_rate']:>11.1%}{s['late_captured']:>15.1%}")
        print(f"     {desc[t]['description']}")

    print("\n[6/6] Decile lift (TEST)")
    # Deciles evaluate RANKING, so they use raw scores (calibrated scores are heavily tied).
    dt = compute_decile_table(y_te, raw_te)
    dt.to_csv(OUTPUTS_DIR / "decile_table.csv", index=False)
    save_lift_gain_plot(dt, y_te, OUTPUTS_DIR / "lift_gain.png")

    export = {
        "best_model": "lightgbm_tuned",
        "winner_model": "tuned",
        "flag_definition": "Flagged = High tier (score >= tier_cutoffs.high_min)",
        "thresholds": {
            "flag_high_tier": round(high_cut, 4),
            "precision_80": round(p80, 4),
            "max_f1": round(max_f1, 4),
            "final_tuned_max_f1": round(max_f1, 4),
            **{k: v for k, v in other.items() if k.startswith("cost_")},
        },
        "tier_cutoffs": {
            "low_max": round(low_cut, 4),
            "high_min": round(high_cut, 4),
            "note": ("Low: score < low_max | Medium: low_max <= score < high_min | High: score >= high_min. "
                     "Cutoffs are midpoints of low-density gaps between calibrated-score clusters, "
                     "selected on validation scores only."),
        },
        "cutoff_selection": {
            "rule": {
                "cluster_link_gap": tiering.LINK_GAP,
                "density_window": tiering.WINDOW,
                "max_window_mass": tiering.MAX_WINDOW_MASS,
                "never_split_cluster_above_share": tiering.BIG_CLUSTER,
                "min_tier_share_validation": tiering.MIN_TIER_SHARE,
                "max_flag_swing_pp": tiering.MAX_FLAG_SWING_PP,
            },
            "candidates": cands,
            "checks": checks,
        },
        "tier_stats_validation": val_stats,
        "tier_stats_test": te_stats,
        "tier_descriptions": desc,
        "stability_test": stability,
        "stability_swing_pp": swings,
        "cost_config": COST_CONFIG,
        "base_rate_test": base_rate,
    }
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"\n  Saved -> {MODELS_DIR / 'thresholds.json'}")
    print("\n" + "=" * 72 + "\n  Done.\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    main()
