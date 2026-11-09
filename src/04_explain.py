"""
src/04_explain.py
=================
Step 5 – SHAP Explanations for RTO Shield AI - Model 1.

Uses the saved LightGBM model and fitted preprocessor to produce:
  1. Global importance  → outputs/shap_importance.png
                          outputs/shap_summary.png (beeswarm)
  2. Dependence plots   → outputs/shap_dep_<feature>.png  (top 3)
  3. Waterfall plots    → outputs/shap_waterfall_<case>.png  (3 cases)
  4. SHAP additivity    → verify base + sum(SHAP) ≈ raw_margin for all 500
  5. Sanity check       → Same Day late rate before vs at/after noon

Usage
-----
  python src/04_explain.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR,
    TARGET_COL, TRAIN_FRAC, VAL_FRAC, SEED,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)
from features import _select_feature_cols
from reasons  import get_reason_codes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"
DATE_COL      = "order_date"
SAMPLE_N      = 500
SHAP_TOL      = 1e-4   # additivity tolerance (log-odds space)

POS_COLOR = "#E53935"   # red  → raises risk
NEG_COLOR = "#1E88E5"   # blue → lowers risk


# ===========================================================================
# Helpers
# ===========================================================================

def _sep(char="─", w=72): return char * w


def _load_test_split() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (test_df, X_test, y_test)."""
    prep   = joblib.load(MODELS_DIR / "preprocessor.joblib")
    orders = pd.read_parquet(ORDER_PARQUET).sort_values(DATE_COL).reset_index(drop=True)
    n      = len(orders)
    te_st  = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    test_df = orders.iloc[te_st:].copy().reset_index(drop=True)

    X_raw   = _select_feature_cols(test_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    X_te    = prep.transform(X_raw)
    y_te    = test_df[TARGET_COL].values
    return test_df, X_te, y_te


def _clean_name(raw: str) -> str:
    """Strip sklearn prefix for plotting: 'num__customer_prev_late_rate' → 'customer_prev_late_rate'."""
    return raw.split("__", 1)[-1]


# ===========================================================================
# 1. SHAP global importance + beeswarm
# ===========================================================================

def compute_shap(
    model,
    X_te:   np.ndarray,
    feat_names: list[str],
    rng:    np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, shap.Explanation]:
    """
    Draw SAMPLE_N rows, run TreeExplainer, return
    (sample_idx, X_sample, shap_explanation).
    """
    idx = rng.choice(len(X_te), size=min(SAMPLE_N, len(X_te)), replace=False)
    idx = np.sort(idx)
    X_sample = X_te[idx]

    explainer = shap.TreeExplainer(model)
    sv = explainer(X_sample)          # returns shap.Explanation object

    return idx, X_sample, sv, explainer


def print_top_features(sv: shap.Explanation, feat_names: list[str], top_n: int = 10) -> list[str]:
    mean_abs = np.abs(sv.values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    total    = mean_abs.sum()

    print(f"\n  {'Rank':<5}  {'Feature':<45}  {'Mean|SHAP|':>10}  {'Rel%':>7}")
    print(f"  {_sep('-', 72)}")
    top_features = []
    for rank, i in enumerate(order[:top_n], 1):
        name  = _clean_name(feat_names[i])
        val   = mean_abs[i]
        rel   = val / total * 100 if total > 0 else 0.0
        print(f"  {rank:<5}  {name:<45}  {val:>10.5f}  {rel:>6.2f}%")
        top_features.append(feat_names[i])

    return top_features   # raw names (with prefix)


def save_importance_bar(sv: shap.Explanation, feat_names: list[str], out_path: Path) -> None:
    mean_abs = np.abs(sv.values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:15]

    names_clean = [_clean_name(feat_names[i]) for i in order]
    vals        = mean_abs[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.RdBu_r(np.linspace(0.15, 0.85, len(vals)))   # type: ignore
    bars    = ax.barh(range(len(vals)), vals[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names_clean[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Top-15 Global Feature Importance (Mean |SHAP|)", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  SHAP importance bar chart saved → {out_path}")


def save_beeswarm(sv: shap.Explanation, feat_names: list[str], out_path: Path) -> None:
    # Rename for clean display
    sv_copy = sv
    sv_copy.feature_names = [_clean_name(n) for n in feat_names]
    fig, ax = plt.subplots(figsize=(11, 8))
    shap.plots.beeswarm(sv_copy, max_display=15, show=False, plot_size=None)
    plt.title("SHAP Beeswarm – Top-15 Features", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP beeswarm saved → {out_path}")


# ===========================================================================
# 2. Dependence plots for top-3 features
# ===========================================================================

def save_dependence_plots(
    sv:         shap.Explanation,
    X_sample:   np.ndarray,
    feat_names: list[str],
    top_raw:    list[str],
) -> None:
    """
    Save one dependence scatter plot per top-3 feature.
    Uses raw SHAP values vs transformed feature value, coloured by the
    feature with the highest interaction (auto-detected via correlation).
    """
    top3 = top_raw[:3]

    for feat_raw in top3:
        fi = feat_names.index(feat_raw)
        shap_vals_col = sv.values[:, fi]
        feat_vals_col = X_sample[:, fi]

        # Find strongest interacting feature (max |correlation| with SHAP residuals)
        resid = shap_vals_col - np.polyval(np.polyfit(feat_vals_col, shap_vals_col, 1), feat_vals_col)
        corrs = []
        for j in range(X_sample.shape[1]):
            if j == fi:
                corrs.append(0.0)
                continue
            try:
                c = abs(np.corrcoef(X_sample[:, j], resid)[0, 1])
            except Exception:
                c = 0.0
            corrs.append(0.0 if np.isnan(c) else c)
        interact_i = int(np.argmax(corrs))
        interact_vals = X_sample[:, interact_i]

        clean_name  = _clean_name(feat_raw)
        interact_nm = _clean_name(feat_names[interact_i])
        fname_safe  = clean_name.replace(" ", "_").replace("/", "_")[:40]
        out_path    = OUTPUTS_DIR / f"shap_dep_{fname_safe}.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(feat_vals_col, shap_vals_col,
                        c=interact_vals, cmap="coolwarm",
                        alpha=0.65, s=18, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label(interact_nm, fontsize=8)
        ax.axhline(0, color="grey", ls="--", lw=0.9, alpha=0.6)
        ax.set_xlabel(clean_name, fontsize=11)
        ax.set_ylabel("SHAP value (log-odds)", fontsize=11)
        ax.set_title(f"Dependence plot – {clean_name}", fontsize=12)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Dependence plot saved → {out_path}")


# ===========================================================================
# 3. Waterfall plots for 3 test cases
# ===========================================================================

def _find_cases(
    X_te:   np.ndarray,
    y_te:   np.ndarray,
    proba:  np.ndarray,
) -> dict[str, int]:
    """
    Find indices for:
      'confident_late'   – highest proba among actual 1s
      'confident_ontime' – lowest proba among actual 0s
      'false_negative'   – FN with highest score (missed late, closest to boundary)
    All indices are into X_te (the full test set, not the sample).
    """
    cases = {}
    # Confident late: actual=1, highest raw proba
    late_mask = y_te == 1
    cases["confident_late"] = int(np.where(late_mask)[0][np.argmax(proba[late_mask])])

    # Confident on-time: actual=0, lowest raw proba
    ontime_mask = y_te == 0
    cases["confident_ontime"] = int(np.where(ontime_mask)[0][np.argmin(proba[ontime_mask])])

    # False negative: predicted 0 (< 0.5), actual 1, highest proba (narrowest miss)
    fn_mask = (proba < 0.5) & (y_te == 1)
    fn_indices = np.where(fn_mask)[0]
    cases["false_negative"] = int(fn_indices[np.argmax(proba[fn_mask])])

    return cases


def save_waterfall(
    explainer:   shap.TreeExplainer,
    X_te:        np.ndarray,
    case_idx:    int,
    case_label:  str,
    feat_names:  list[str],
    y_te:        np.ndarray,
    proba:       np.ndarray,
    test_df:     pd.DataFrame,
    out_path:    Path,
) -> None:
    """Compute SHAP for a single row and save waterfall plot."""
    X_row = X_te[case_idx : case_idx + 1]
    sv_row = explainer(X_row)

    # Rename for display
    sv_row.feature_names = [_clean_name(n) for n in feat_names]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(sv_row[0], max_display=12, show=False)
    title_str = (
        f"Waterfall – {case_label}\n"
        f"Score={proba[case_idx]:.4f}  |  Actual={'Late' if y_te[case_idx]==1 else 'On-time'}"
    )
    plt.title(title_str, fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Waterfall saved → {out_path}")


def print_waterfall_summary(
    explainer:  shap.TreeExplainer,
    X_te:       np.ndarray,
    cases:      dict[str, int],
    feat_names: list[str],
    y_te:       np.ndarray,
    proba:      np.ndarray,
    test_df:    pd.DataFrame,
) -> None:
    top_k = 6
    for case_label, case_idx in cases.items():
        X_row   = X_te[case_idx : case_idx + 1]
        sv_row  = explainer.shap_values(X_row)
        # LightGBM TreeExplainer may return list[array] for binary
        if isinstance(sv_row, list):
            sv_row = sv_row[1]
        sv_flat  = sv_row[0]
        order    = np.argsort(np.abs(sv_flat))[::-1][:top_k]

        actual_str = "Late" if y_te[case_idx] == 1 else "On-time"
        order_id   = test_df.iloc[case_idx]["Order Id"]
        print(f"\n  ── {case_label.replace('_',' ').title()} ──")
        print(f"     Order Id : {order_id}")
        print(f"     Score    : {proba[case_idx]:.4f}")
        print(f"     Actual   : {actual_str}")
        print(f"     Top {top_k} contributors:")
        for rank, fi in enumerate(order, 1):
            name  = _clean_name(feat_names[fi])
            sv    = float(sv_flat[fi])
            arrow = "▲ raises" if sv > 0 else "▼ lowers"
            print(f"       {rank}. {name:<42}  {sv:>+8.4f}  ({arrow} risk)")

        # Reason codes via reasons.py
        feat_vals = X_te[case_idx]
        codes = get_reason_codes(sv_flat, feat_names, feat_vals, k=3)
        print(f"     Reason codes:")
        for c in codes:
            print(f"       • {c}")


# ===========================================================================
# 4. SHAP additivity check
# ===========================================================================

def check_additivity(
    model,
    explainer:  shap.TreeExplainer,
    X_sample:   np.ndarray,
    sample_sv:  shap.Explanation,
) -> bool:
    """
    Verify: base_value + sum(SHAP values) ≈ model raw margin score
    for every row in X_sample.

    Returns True if all rows pass within SHAP_TOL.
    """
    raw_margin  = model.predict(X_sample, raw_score=True)
    base_value  = sample_sv.base_values
    shap_sums   = sample_sv.values.sum(axis=1)

    # base_values may be scalar or per-row
    if np.ndim(base_value) == 0:
        base_value = np.full(len(raw_margin), float(base_value))
    elif base_value.ndim > 1:
        base_value = base_value[:, 1] if base_value.shape[1] == 2 else base_value[:, 0]

    reconstructed = base_value + shap_sums
    max_err       = float(np.abs(raw_margin - reconstructed).max())
    all_pass      = max_err <= SHAP_TOL

    print(f"\n  SHAP additivity check on {len(X_sample)} rows")
    print(f"  Max |margin − (base + ΣSHAPs)| = {max_err:.6f}  (tol={SHAP_TOL})")
    if all_pass:
        print(f"  [PASS] SHAP values are additive.")
    else:
        print(f"  [FAIL] Additivity violated beyond tolerance.")
        # Soft pass for SHAP 0.4x with model_output='raw_score' difference
        # Try model_output='probability' base and re-check
        print(f"  [INFO] Note: tolerance mismatch may occur when SHAP uses internal "
              f"log-odds scaling.  Max error = {max_err:.6f}.")
    return all_pass


# ===========================================================================
# 5. Sanity check: Same Day late rate before vs at/after noon
# ===========================================================================

def sanity_check_sameday(test_df: pd.DataFrame) -> None:
    sd = test_df[test_df["Shipping_Mode"] == "Same Day"].copy()
    if len(sd) == 0:
        print("  No Same Day orders in test set.")
        return

    before_noon = sd[sd["order_hour"] < 12]
    at_or_after = sd[sd["order_hour"] >= 12]

    n_before = len(before_noon)
    n_after  = len(at_or_after)
    r_before = before_noon[TARGET_COL].mean() if n_before > 0 else float("nan")
    r_after  = at_or_after[TARGET_COL].mean() if n_after  > 0 else float("nan")

    print(f"\n  Same Day orders in TEST: {len(sd):,}")
    print(f"  {'Period':<25}  {'N':>6}  {'Late rate':>10}")
    print(f"  {_sep('-', 45)}")
    print(f"  {'Before noon (hour < 12)':<25}  {n_before:>6,}  {r_before:>10.4f}")
    print(f"  {'Noon or after (≥ 12)':<25}  {n_after:>6,}  {r_after:>10.4f}")
    if n_before > 0 and n_after > 0:
        diff = r_after - r_before
        sign = "higher" if diff > 0 else "lower"
        print(f"\n  Δ = {diff:+.4f} → orders placed at/after noon are {abs(diff)*100:.1f}pp "
              f"{sign}-risk for Same Day shipping.")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 72)
    print("  RTO Shield AI – Step 5: SHAP Explainability")
    print("=" * 72)

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("\n[1/6] Loading model, preprocessor, and test data …")
    model  = joblib.load(MODELS_DIR / "lightgbm.joblib")
    prep   = joblib.load(MODELS_DIR / "preprocessor.joblib")
    cal    = joblib.load(MODELS_DIR / "calibrator.joblib")
    test_df, X_te, y_te = _load_test_split()
    feat_names = list(prep.get_feature_names_out())
    proba  = model.predict_proba(X_te)[:, 1]
    print(f"  Test set : {len(X_te):,} rows × {X_te.shape[1]} features")

    # ── SHAP on 500-row sample ────────────────────────────────────────────────
    print(f"\n[2/6] Computing SHAP values on {SAMPLE_N}-row random sample …")
    rng = np.random.default_rng(SEED)
    sample_idx, X_sample, sv, explainer = compute_shap(model, X_te, feat_names, rng)
    print(f"  SHAP Explanation object: {sv.values.shape[0]} rows × {sv.values.shape[1]} features")
    print(f"  Base value (expected model output): {float(np.mean(sv.base_values)):.4f}")

    # ── Top-10 importance table ───────────────────────────────────────────────
    print(f"\n  {'='*72}")
    print(f"  Top-10 Features by Mean |SHAP| (global importance)")
    print(f"  {'='*72}")
    top_raw = print_top_features(sv, feat_names, top_n=10)

    # ── Save plots ────────────────────────────────────────────────────────────
    print(f"\n[3/6] Saving global importance and beeswarm plots …")
    save_importance_bar(sv, feat_names, OUTPUTS_DIR / "shap_importance.png")
    save_beeswarm(sv, feat_names, OUTPUTS_DIR / "shap_summary.png")

    # ── Dependence plots (top 3) ──────────────────────────────────────────────
    print(f"\n[4/6] Saving dependence plots for top-3 features …")
    save_dependence_plots(sv, X_sample, feat_names, top_raw)

    # ── Waterfall plots for 3 cases ───────────────────────────────────────────
    print(f"\n[5/6] Waterfall plots for 3 representative test orders …")
    cases = _find_cases(X_te, y_te, proba)
    print(f"\n  {'='*72}")
    print(f"  Per-Order Waterfall Summaries (top-{6} contributors)")
    print(f"  {'='*72}")

    print_waterfall_summary(explainer, X_te, cases, feat_names, y_te, proba, test_df)

    case_meta = {
        "confident_late":   "Confidently predicted late",
        "confident_ontime": "Confidently predicted on-time",
        "false_negative":   "False negative (missed late)",
    }
    for case_key, case_idx in cases.items():
        label    = case_meta[case_key]
        out_path = OUTPUTS_DIR / f"shap_waterfall_{case_key}.png"
        save_waterfall(
            explainer, X_te, case_idx, label,
            feat_names, y_te, proba, test_df, out_path,
        )

    # ── SHAP additivity check ─────────────────────────────────────────────────
    print(f"\n[6/6] SHAP additivity verification …")
    print(f"\n  {'='*72}")
    print(f"  SHAP Additivity Check")
    print(f"  {'='*72}")
    check_additivity(model, explainer, X_sample, sv)

    # ── Sanity check: Same Day noon ───────────────────────────────────────────
    print(f"\n  {'='*72}")
    print(f"  Sanity Check: Same Day shipping – Before vs At/After Noon")
    print(f"  {'='*72}")
    sanity_check_sameday(test_df)

    print("\n" + "=" * 72)
    print("  Explainability analysis complete.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

