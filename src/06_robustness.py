"""
src/06_robustness.py
====================
Step 7 – Robustness, Ablations, Backtest & Drift Checks.

Sections
--------
  1. Feature-group ablations (6 variants) → bar chart
  2. Per-shipping-mode evaluation (full model vs baseline on TEST)
  3. Rolling-origin expanding-window backtest (5 folds) → line plot
  4. Secondary target: aux_target_cancel_fraud evaluation
  5. Population Stability Index (PSI) for top-8 features (train vs test)

Usage
-----
  python src/06_robustness.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

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
from features import _select_feature_cols, build_feature_matrix
from utils    import compute_metrics

ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"
DATE_COL      = "order_date"

# ---------------------------------------------------------------------------
# Feature group definitions for ablations
# ---------------------------------------------------------------------------
SHIPPING_CAT   = ["Shipping_Mode"]
SHIPPING_NUM   = ["Days_for_shipment_scheduled"]
HISTORY_NUM    = [
    "customer_prev_orders", "customer_prev_late_rate",
    "country_prev_orders",  "country_prev_late_rate",
    "customer_prev_cash_share", "days_since_prev_order",
]
TIME_NUM       = ["order_month", "order_dow", "order_hour", "is_weekend", "is_night_order"]
PAYMENT_CAT    = ["Type"]
PAYMENT_NUM    = ["is_cash"]

ABLATION_SPECS: list[dict] = [
    {
        "label": "All features",
        "cat":   CATEGORICAL_FEATURES,
        "num":   NUMERIC_FEATURES,
    },
    {
        "label": "No shipping mode/days",
        "cat":   [c for c in CATEGORICAL_FEATURES if c not in SHIPPING_CAT],
        "num":   [n for n in NUMERIC_FEATURES    if n not in SHIPPING_NUM],
    },
    {
        "label": "No history features",
        "cat":   CATEGORICAL_FEATURES,
        "num":   [n for n in NUMERIC_FEATURES if n not in HISTORY_NUM],
    },
    {
        "label": "No time features",
        "cat":   CATEGORICAL_FEATURES,
        "num":   [n for n in NUMERIC_FEATURES if n not in TIME_NUM],
    },
    {
        "label": "No payment/is_cash",
        "cat":   [c for c in CATEGORICAL_FEATURES if c not in PAYMENT_CAT],
        "num":   [n for n in NUMERIC_FEATURES    if n not in PAYMENT_NUM],
    },
    {
        "label": "Shipping only (baseline)",
        "cat":   BASELINE_CAT_FEATURES,
        "num":   BASELINE_NUM_FEATURES,
    },
]

# Top-8 raw feature names (from SHAP on final_model, see 04_explain output)
TOP8_RAW = [
    "cat__Shipping_Mode_Standard Class",
    "num__Days_for_shipment_scheduled",
    "cat__Shipping_Mode_First Class",
    "cat__Type_TRANSFER",
    "num__order_hour",
    "cat__Shipping_Mode_Same Day",
    "cat__Shipping_Mode_Second Class",
    "cat__Type_DEBIT",
]


# ===========================================================================
# Helpers
# ===========================================================================

def _sep(char="─", w=74): return char * w


def _load_splits():
    orders = (
        pd.read_parquet(ORDER_PARQUET)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    n       = len(orders)
    tr_end  = int(np.floor(n * TRAIN_FRAC))
    val_end = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    return (
        orders,
        orders.iloc[:tr_end].copy(),
        orders.iloc[tr_end:val_end].copy(),
        orders.iloc[val_end:].copy(),
        tr_end, val_end, n,
    )


def _make_lgb(seed: int = SEED, **extra) -> lgb.LGBMClassifier:
    """Construct final-tuned LightGBM params from saved model."""
    saved = joblib.load(MODELS_DIR / "final_model.joblib")
    params = saved.get_params()
    params.update({"random_state": seed, "n_jobs": -1, "verbose": -1})
    params.update(extra)
    return lgb.LGBMClassifier(**params)


def _fit_eval(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    target:   str = TARGET_COL,
    seed:     int = SEED,
) -> dict:
    """
    Build fresh preprocessor on train_df, fit LGB, calibrate on val,
    return {'roc_auc', 'pr_auc', 'f1', 'recall', 'precision', 'brier'} on test.
    """
    X_tr_raw  = _select_feature_cols(train_df, cat_cols, num_cols)
    X_val_raw = _select_feature_cols(val_df,   cat_cols, num_cols)
    X_te_raw  = _select_feature_cols(test_df,  cat_cols, num_cols)

    _, (X_val_arr, X_te_arr), _, prep_local = build_feature_matrix(
        train_df, [val_df, test_df],
        cat_cols=cat_cols, num_cols=num_cols,
        save_preprocessor=False,
    )
    X_tr_arr = prep_local.transform(X_tr_raw)

    y_tr  = train_df[target].values
    y_val = val_df[target].values
    y_te  = test_df[target].values

    model = _make_lgb(seed=seed)
    model.fit(
        X_tr_arr, y_tr,
        eval_set=[(X_val_arr, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    # Calibrate on val
    raw_val = model.predict_proba(X_val_arr)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_val, y_val)

    cal_te = ir.predict(model.predict_proba(X_te_arr)[:, 1])

    # Threshold: max-F1 on calibrated val
    cal_val = ir.predict(raw_val)
    thrs = np.linspace(0.01, 0.99, 990)
    best_thr, best_f1 = 0.5, 0.0
    for t in thrs:
        yp = (cal_val >= t).astype(int)
        tp = int(((yp == 1) & (y_val == 1)).sum())
        fp = int(((yp == 1) & (y_val == 0)).sum())
        fn = int(((yp == 0) & (y_val == 1)).sum())
        pr_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_ = 2 * pr_ * rc_ / (pr_ + rc_) if (pr_ + rc_) > 0 else 0.0
        if f1_ > best_f1:
            best_f1, best_thr = f1_, t

    m = compute_metrics(y_te, cal_te, threshold=best_thr)
    return {k: m[k] for k in ["roc_auc", "pr_auc", "f1", "recall", "precision", "brier"]}


# ===========================================================================
# 1. Ablations
# ===========================================================================

def run_ablations(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    print(f"\n  {'Ablation':<30}  {'ROC-AUC':>8}  {'PR-AUC':>8}  {'F1':>8}")
    print(f"  {_sep('-', 60)}")

    for spec in ABLATION_SPECS:
        label    = spec["label"]
        cat_cols = spec["cat"]
        num_cols = spec["num"]
        n_feats  = len(cat_cols) + len(num_cols)

        res = _fit_eval(train_df, val_df, test_df, cat_cols, num_cols)

        print(
            f"  {label:<30}  {res['roc_auc']:>8.4f}  "
            f"{res['pr_auc']:>8.4f}  {res['f1']:>8.4f}  "
            f"({n_feats} features)"
        )
        rows.append({"label": label, "n_features": n_feats, **res})

    return pd.DataFrame(rows)


def save_ablation_chart(abl_df: pd.DataFrame, out_path: Path) -> None:
    labels = abl_df["label"].tolist()
    x      = np.arange(len(labels))
    width  = 0.28

    # Reference line = "All features" ROC-AUC
    all_roc = abl_df.loc[abl_df["label"] == "All features", "roc_auc"].values[0]

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width, abl_df["roc_auc"], width, label="ROC-AUC", color="#1976D2", alpha=0.88)
    b2 = ax.bar(x,          abl_df["pr_auc"],  width, label="PR-AUC",  color="#43A047", alpha=0.88)
    b3 = ax.bar(x + width,  abl_df["f1"],      width, label="F1",      color="#FB8C00", alpha=0.88)

    ax.axhline(all_roc, color="#1976D2", ls="--", lw=1.1, alpha=0.5, label=f"All-feat ROC-AUC ({all_roc:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Feature Ablation Study – Test Set Metrics", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Value annotations
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Ablation chart saved → {out_path}")


# ===========================================================================
# 2. Per-shipping-mode evaluation
# ===========================================================================

def per_mode_evaluation(
    test_df:   pd.DataFrame,
    full_proba: np.ndarray,
    base_proba: np.ndarray,
    y_te:       np.ndarray,
) -> None:
    modes = sorted(test_df["Shipping_Mode"].unique())

    print(f"\n  {'Mode':<20}  {'N':>5}  "
          f"{'Full ROC-AUC':>12}  {'Full PR-AUC':>11}  "
          f"{'Base ROC-AUC':>12}  {'Base PR-AUC':>11}  "
          f"{'ΔROC':>7}")
    print(f"  {_sep('-', 90)}")

    for mode in modes:
        mask = (test_df["Shipping_Mode"] == mode).values
        n    = mask.sum()
        y_m  = y_te[mask]
        # Skip if only one class present
        if y_m.sum() == 0 or y_m.sum() == n:
            print(f"  {mode:<20}  {n:>5}  {'(one class)':>12}")
            continue

        roc_full = roc_auc_score(y_m, full_proba[mask])
        pra_full = average_precision_score(y_m, full_proba[mask])
        roc_base = roc_auc_score(y_m, base_proba[mask])
        pra_base = average_precision_score(y_m, base_proba[mask])
        delta    = roc_full - roc_base
        sign     = "▲" if delta > 0 else "▼"

        print(
            f"  {mode:<20}  {n:>5,}  "
            f"{roc_full:>12.4f}  {pra_full:>11.4f}  "
            f"{roc_base:>12.4f}  {pra_base:>11.4f}  "
            f"{sign}{abs(delta):>5.4f}"
        )


# ===========================================================================
# 3. Rolling-origin expanding-window backtest
# ===========================================================================

def rolling_backtest(
    orders:   pd.DataFrame,
    n_folds:  int = 5,
) -> pd.DataFrame:
    """
    5 expanding-window folds. Each fold uses all data up to fold boundary
    for training and the next chunk for validation and test.

    Strategy
    --------
    - Sort orders by date.
    - Divide into (n_folds + 1) equal time blocks.
    - Fold k: train on blocks 0..k, validate on block k+1, "test" on block k+1
      (since we have no future holdout per fold, we report val = test here;
       the purpose is to track metric stability across time, not leakage-free eval).
    - Actually, we use the following proper setup:
        train  = first k/(n_folds) fraction
        val    = next 1/(n_folds+1) fraction  (for calibration)
        test   = next 1/(n_folds+1) fraction  (reported)
      with expanding train window.
    """
    n = len(orders)
    # We create 5 folds by expanding the train cutoff across the timeline
    # Minimum train = 40% of data; each fold adds ~12% more.
    min_train_frac = 0.40
    rows = []

    fold_boundaries = np.linspace(min_train_frac, 0.80, n_folds)

    print(f"\n  {'Fold':>5}  {'Train N':>8}  {'Test N':>7}  "
          f"{'ROC-AUC':>8}  {'PR-AUC':>8}  "
          f"{'TestStart':<12}  {'TestEnd':<12}")
    print(f"  {_sep('-', 72)}")

    for fold_i, train_frac in enumerate(fold_boundaries, 1):
        tr_end  = int(np.floor(n * train_frac))
        # val = next 5%, test = following 10%
        val_end = int(np.floor(n * (train_frac + 0.05)))
        te_end  = int(np.floor(n * (train_frac + 0.15)))
        te_end  = min(te_end, n)

        if val_end >= te_end or te_end <= val_end:
            continue

        fold_train = orders.iloc[:tr_end].copy()
        fold_val   = orders.iloc[tr_end:val_end].copy()
        fold_test  = orders.iloc[val_end:te_end].copy()

        if len(fold_val) < 50 or len(fold_test) < 50:
            continue
        if fold_test[TARGET_COL].sum() == 0:
            continue

        res = _fit_eval(
            fold_train, fold_val, fold_test,
            CATEGORICAL_FEATURES, NUMERIC_FEATURES,
        )

        te_start = fold_test[DATE_COL].min().date()
        te_end_d = fold_test[DATE_COL].max().date()

        print(
            f"  {fold_i:>5}  {len(fold_train):>8,}  {len(fold_test):>7,}  "
            f"{res['roc_auc']:>8.4f}  {res['pr_auc']:>8.4f}  "
            f"{str(te_start):<12}  {str(te_end_d):<12}"
        )
        rows.append({
            "fold": fold_i, "train_n": len(fold_train),
            "test_n": len(fold_test),
            "test_start": str(te_start), "test_end": str(te_end_d),
            **res,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f"  {_sep('-', 72)}")
        print(
            f"  {'Mean':>5}  {'':>8}  {'':>7}  "
            f"{df['roc_auc'].mean():>8.4f}  {df['pr_auc'].mean():>8.4f}"
        )
    return df


def save_backtest_plot(bt_df: pd.DataFrame, out_path: Path) -> None:
    if len(bt_df) == 0:
        print("  No backtest data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(bt_df["fold"], bt_df["roc_auc"], "o-", color="#1976D2",
            lw=2, ms=7, label="ROC-AUC")
    ax.plot(bt_df["fold"], bt_df["pr_auc"],  "s-", color="#43A047",
            lw=2, ms=7, label="PR-AUC")

    mean_roc = bt_df["roc_auc"].mean()
    mean_pra = bt_df["pr_auc"].mean()
    ax.axhline(mean_roc, color="#1976D2", ls="--", lw=1, alpha=0.5,
               label=f"Mean ROC-AUC ({mean_roc:.4f})")
    ax.axhline(mean_pra, color="#43A047", ls="--", lw=1, alpha=0.5,
               label=f"Mean PR-AUC ({mean_pra:.4f})")

    ax.set_xlabel("Expanding-window fold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Rolling-Origin Backtest – ROC-AUC & PR-AUC per Fold", fontsize=12)
    ax.set_xticks(bt_df["fold"])
    xticklabels = [f"F{int(f)}\n{s}" for f, s in
                   zip(bt_df["fold"], bt_df["test_start"])]
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Backtest plot saved → {out_path}")


# ===========================================================================
# 4. Secondary target: aux_target_cancel_fraud
# ===========================================================================

def evaluate_secondary_target(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """
    Retrain the model using aux_target_cancel_fraud as the label.
    Report ROC-AUC and PR-AUC on test.
    This is a ROBUSTNESS CHECK ONLY — not a production model.
    """
    AUX = "aux_target_cancel_fraud"
    pos_rate_tr = train_df[AUX].mean()
    pos_rate_te = test_df[AUX].mean()

    print(f"\n  Aux target positive rate → train: {pos_rate_tr:.4f}  "
          f"test: {pos_rate_te:.4f}")

    # Need at least some positives in val for calibration
    if val_df[AUX].sum() < 5:
        print("  [WARN] Too few positives in val for secondary target — skipping calibration.")

    res = _fit_eval(
        train_df, val_df, test_df,
        CATEGORICAL_FEATURES, NUMERIC_FEATURES,
        target=AUX,
    )

    print(f"\n  {'Metric':<12}  {'Value':>8}")
    print(f"  {_sep('-', 24)}")
    for k in ["roc_auc", "pr_auc", "f1", "recall", "precision"]:
        print(f"  {k:<12}  {res[k]:>8.4f}")
    print(
        f"\n  [INFO] Secondary target (cancel/fraud) ROC-AUC = {res['roc_auc']:.4f}.  "
        "This is a robustness check only; the production target remains Late_delivery_risk."
    )


# ===========================================================================
# 5. Population Stability Index (PSI)
# ===========================================================================

def _psi_one(train_vals: np.ndarray, test_vals: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI for a single continuous feature using equal-frequency bins on train.
    PSI < 0.1  → no significant shift
    PSI 0.1-0.2 → moderate shift
    PSI > 0.2  → major shift
    """
    # Use train quantiles for bin edges
    pcts  = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(train_vals[~np.isnan(train_vals)], pcts))

    if len(edges) < 3:
        return 0.0

    def _safe_hist(vals, edges):
        counts, _ = np.histogram(vals[~np.isnan(vals)], bins=edges)
        freq = counts / max(counts.sum(), 1)
        freq = np.where(freq == 0, 1e-4, freq)
        return freq

    f_tr = _safe_hist(train_vals, edges)
    f_te = _safe_hist(test_vals,  edges)
    psi  = float(np.sum((f_te - f_tr) * np.log(f_te / f_tr)))
    return psi


def compute_psi(
    orders:    pd.DataFrame,
    tr_end:    int,
    val_end:   int,
    prep,
) -> pd.DataFrame:
    """
    Compute PSI for top-8 features between train and test (transformed space).
    """
    train_df = orders.iloc[:tr_end]
    test_df  = orders.iloc[val_end:]

    X_tr = prep.transform(_select_feature_cols(train_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    X_te = prep.transform(_select_feature_cols(test_df,  CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    feat_names = list(prep.get_feature_names_out())

    rows = []
    for raw_name in TOP8_RAW:
        if raw_name not in feat_names:
            continue
        fi    = feat_names.index(raw_name)
        clean = raw_name.split("__", 1)[-1]
        psi   = _psi_one(X_tr[:, fi].astype(float), X_te[:, fi].astype(float))
        flag  = ("OK"       if psi < 0.1  else
                 "Moderate" if psi < 0.2 else
                 "Major")
        rows.append({"feature": clean, "psi": round(psi, 5), "status": flag})

    return pd.DataFrame(rows)


def print_psi_table(psi_df: pd.DataFrame) -> None:
    print(f"\n  {'Feature':<45}  {'PSI':>7}  {'Status'}")
    print(f"  {_sep('-', 65)}")
    for _, row in psi_df.iterrows():
        print(f"  {row['feature']:<45}  {row['psi']:>7.5f}  {row['status']}")
    n_ok  = (psi_df["psi"] < 0.1).sum()
    n_mod = ((psi_df["psi"] >= 0.1) & (psi_df["psi"] < 0.2)).sum()
    n_maj = (psi_df["psi"] >= 0.2).sum()
    print(f"\n  Summary: {n_ok} stable / {n_mod} moderate / {n_maj} major drift")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 74)
    print("  RTO Shield AI – Step 7: Robustness, Ablations, Backtest & Drift")
    print("=" * 74)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[0] Loading data …")
    orders, train_df, val_df, test_df, tr_end, val_end, n = _load_splits()
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # Load saved artefacts
    prep         = joblib.load(MODELS_DIR / "preprocessor.joblib")
    final_model  = joblib.load(MODELS_DIR / "final_model.joblib")
    final_cal    = joblib.load(MODELS_DIR / "final_calibrator.joblib")
    base_model   = joblib.load(MODELS_DIR / "baseline_lr.joblib")
    base_prep    = joblib.load(MODELS_DIR / "baseline_preprocessor.joblib")

    # Precompute full-model and baseline calibrated probas on test
    from features import _select_feature_cols
    X_val_full = prep.transform(_select_feature_cols(val_df,  CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    X_te_full  = prep.transform(_select_feature_cols(test_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    y_te       = test_df[TARGET_COL].values

    full_proba = final_cal.predict(final_model.predict_proba(X_te_full)[:, 1])

    X_val_b   = base_prep.transform(_select_feature_cols(val_df,  BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES))
    X_te_b    = base_prep.transform(_select_feature_cols(test_df, BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES))
    y_val     = val_df[TARGET_COL].values
    raw_val_b = base_model.predict_proba(X_val_b)[:, 1]
    ir_b      = IsotonicRegression(out_of_bounds="clip")
    ir_b.fit(raw_val_b, y_val)
    base_proba = ir_b.predict(base_model.predict_proba(X_te_b)[:, 1])

    # ── 1. Ablations ──────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"[1] Feature Ablation Study")
    print(f"{'='*74}")
    abl_df = run_ablations(train_df, val_df, test_df)
    save_ablation_chart(abl_df, OUTPUTS_DIR / "ablation_chart.png")
    abl_df.to_csv(OUTPUTS_DIR / "ablation_results.csv", index=False)
    print(f"  Ablation results saved → {OUTPUTS_DIR}/ablation_results.csv")

    # ── 2. Per-shipping-mode ──────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"[2] Per-Shipping-Mode Evaluation (TEST set)")
    print(f"{'='*74}")
    per_mode_evaluation(test_df, full_proba, base_proba, y_te)

    # ── 3. Rolling-origin backtest ────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"[3] Rolling-Origin Expanding-Window Backtest (5 folds)")
    print(f"{'='*74}")
    bt_df = rolling_backtest(orders, n_folds=5)
    save_backtest_plot(bt_df, OUTPUTS_DIR / "backtest_plot.png")
    if len(bt_df) > 0:
        bt_df.to_csv(OUTPUTS_DIR / "backtest_results.csv", index=False)
        print(f"  Backtest results saved → {OUTPUTS_DIR}/backtest_results.csv")

    # ── 4. Secondary target ───────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"[4] Secondary Target: aux_target_cancel_fraud  (robustness check only)")
    print(f"{'='*74}")
    evaluate_secondary_target(train_df, val_df, test_df)

    # ── 5. PSI drift check ────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"[5] Population Stability Index – Top-8 Features (Train vs Test)")
    print(f"{'='*74}")
    psi_df = compute_psi(orders, tr_end, val_end, prep)
    print_psi_table(psi_df)
    psi_df.to_csv(OUTPUTS_DIR / "psi_results.csv", index=False)
    print(f"\n  PSI results saved → {OUTPUTS_DIR}/psi_results.csv")

    print("\n" + "=" * 74)
    print("  Robustness analysis complete.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()

