"""
src/06_robustness.py
====================
Step 7 - Robustness, Ablations, Backtest & Drift Checks.

Metric conventions (used everywhere in this project)
----------------------------------------------------
  * RANKING metrics (ROC-AUC, PR-AUC, lifts) use the RAW model scores.
    Isotonic calibration collapses scores into ties, which distorts rankings.
  * DECISION / CALIBRATION outputs (thresholds, precision/recall/F1/accuracy/
    balanced accuracy/specificity/MCC, Brier, ECE) use CALIBRATED scores.
  * Every ablated model gets its OWN max-F1 threshold, chosen on validation.

Sections
--------
  1. Feature-group ablations: ROC-AUC (95% bootstrap CI), PR-AUC, Brier and the
     threshold metrics at each model's own validation-chosen max-F1 threshold,
     plus a paired bootstrap of the ROC-AUC change vs the full model.
  2. Per-shipping-mode evaluation: n, prevalence, ROC-AUC with bootstrap CI,
     PR-AUC and PR-AUC lift over prevalence, informative/limited flag.
  3. Rolling-origin expanding-window backtest (5 folds), incl. PR-AUC lift.
  4. Secondary target: aux_target_cancel_fraud evaluation.
  5. Population Stability Index (PSI) for top-8 features (train vs test).

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
from utils    import compute_metrics, bootstrap_ci

ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"
DATE_COL      = "order_date"
N_BOOT        = 1000
MODES         = ["Same Day", "First Class", "Second Class", "Standard Class"]

# A shipping mode is "limited" when its ROC-AUC lower CI bound is below this
# (this also covers any CI that includes 0.5).
INFORMATIVE_AUC_FLOOR = 0.55

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

def _sep(char="-", w=74): return char * w


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


def _max_f1_threshold(y_val: np.ndarray, cal_val: np.ndarray) -> float:
    """Threshold maximising F1 on the (calibrated) validation scores."""
    best_thr, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 990):
        yp = (cal_val >= t).astype(int)
        tp = int(((yp == 1) & (y_val == 1)).sum())
        fp = int(((yp == 1) & (y_val == 0)).sum())
        fn = int(((yp == 0) & (y_val == 1)).sum())
        pr_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_ = 2 * pr_ * rc_ / (pr_ + rc_) if (pr_ + rc_) > 0 else 0.0
        if f1_ > best_f1:
            best_f1, best_thr = f1_, t
    return float(best_thr)


RESULT_KEYS = ["roc_auc", "pr_auc", "brier", "threshold", "f1", "precision", "recall",
               "accuracy", "balanced_accuracy", "specificity", "mcc"]


def _fit_eval(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    target:   str = TARGET_COL,
    seed:     int = SEED,
    n_boot:   int = 0,
    return_scores: bool = False,
):
    """
    Build a fresh preprocessor on train_df, fit LGB (early stopping on val),
    calibrate on val, choose this model's OWN max-F1 threshold on validation,
    and report on test:

      ranking   : roc_auc, pr_auc (RAW scores), optional roc_auc 95% bootstrap CI
      calibrated: brier
      decision  : threshold, precision, recall, f1, accuracy, balanced_accuracy,
                  specificity, mcc  (calibrated scores at the validation threshold)
    """
    X_tr_raw = _select_feature_cols(train_df, cat_cols, num_cols)

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

    raw_val = model.predict_proba(X_val_arr)[:, 1]
    raw_te  = model.predict_proba(X_te_arr)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_val, y_val)
    cal_val = ir.predict(raw_val)
    cal_te  = ir.predict(raw_te)

    thr = _max_f1_threshold(y_val, cal_val)
    m = compute_metrics(y_te, cal_te, threshold=thr, y_rank=raw_te)
    res = {k: m[k] for k in RESULT_KEYS}
    res["prevalence"] = float(y_te.mean())
    res["pr_auc_lift"] = float(m["pr_auc"] / y_te.mean()) if y_te.mean() > 0 else float("nan")
    if n_boot:
        lo, hi = bootstrap_ci(y_te, raw_te, roc_auc_score, n=n_boot)
        res["roc_auc_ci_lo"], res["roc_auc_ci_hi"] = lo, hi
    if return_scores:
        return res, raw_te, y_te
    return res


def paired_delta_ci(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                    n: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Paired bootstrap of ROC-AUC(a) - ROC-AUC(b) on identical resamples."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        yb = y[b]
        if yb.min() == yb.max():
            continue
        diffs.append(roc_auc_score(yb, s_a[b]) - roc_auc_score(yb, s_b[b]))
    point = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ===========================================================================
# 1. Ablations
# ===========================================================================

def run_ablations(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Returns (ablation table, plain-English verdict on the history features)."""
    rows, scores = [], {}
    y_te = test_df[TARGET_COL].values

    print(f"\n  {'Ablation':<26}{'ROC-AUC [95% CI]':>28}{'PR-AUC':>9}{'Brier':>8}"
          f"{'thr':>7}{'F1':>7}{'Prec':>7}{'Rec':>7}")
    print(f"  {_sep('-', 108)}")

    for spec in ABLATION_SPECS:
        label    = spec["label"]
        n_feats  = len(spec["cat"]) + len(spec["num"])
        res, raw_te, _ = _fit_eval(train_df, val_df, test_df, spec["cat"], spec["num"],
                                   n_boot=N_BOOT, return_scores=True)
        scores[label] = raw_te
        print(f"  {label:<26}{res['roc_auc']:>8.4f} [{res['roc_auc_ci_lo']:.4f}, {res['roc_auc_ci_hi']:.4f}]"
              f"{res['pr_auc']:>9.4f}{res['brier']:>8.4f}{res['threshold']:>7.3f}"
              f"{res['f1']:>7.3f}{res['precision']:>7.3f}{res['recall']:>7.3f}")
        rows.append({"label": label, "n_features": n_feats, **res})

    df = pd.DataFrame(rows)

    # Paired bootstrap of ROC-AUC change vs the full model
    d_pt, d_lo, d_hi, d_sig = [], [], [], []
    for label in df["label"]:
        if label == "All features":
            d_pt.append(0.0); d_lo.append(0.0); d_hi.append(0.0); d_sig.append(False)
            continue
        pt, lo, hi = paired_delta_ci(y_te, scores["All features"], scores[label])
        d_pt.append(pt); d_lo.append(lo); d_hi.append(hi); d_sig.append(bool(lo > 0 or hi < 0))
    df["delta_roc_vs_all"] = d_pt          # ROC-AUC(all) - ROC-AUC(ablated)
    df["delta_ci_lo"] = d_lo
    df["delta_ci_hi"] = d_hi
    df["delta_significant"] = d_sig

    h = df[df["label"] == "No history features"].iloc[0]
    pr_all = float(df.loc[df["label"] == "All features", "pr_auc"].iloc[0])
    if h["delta_significant"] and h["delta_ci_lo"] > 0:
        verdict = (f"History features add a statistically significant ROC-AUC gain: "
                   f"ROC-AUC(all) - ROC-AUC(no history) = {h['delta_roc_vs_all']:+.4f} "
                   f"(95% paired-bootstrap CI [{h['delta_ci_lo']:+.4f}, {h['delta_ci_hi']:+.4f}]).")
    else:
        verdict = (f"History features add NO significant gain: ROC-AUC(all) - ROC-AUC(no history) = "
                   f"{h['delta_roc_vs_all']:+.4f} (95% paired-bootstrap CI [{h['delta_ci_lo']:+.4f}, "
                   f"{h['delta_ci_hi']:+.4f}] includes 0 or is negative); PR-AUC is "
                   f"{pr_all:.4f} with history vs {h['pr_auc']:.4f} without.")
    print(f"\n  VERDICT: {verdict}")
    return df, verdict


def save_ablation_chart(abl_df: pd.DataFrame, out_path: Path) -> None:
    labels = abl_df["label"].tolist()
    x      = np.arange(len(labels))
    width  = 0.36
    roc = abl_df["roc_auc"].values
    err = np.vstack([roc - abl_df["roc_auc_ci_lo"].values, abl_df["roc_auc_ci_hi"].values - roc])
    all_roc = float(abl_df.loc[abl_df["label"] == "All features", "roc_auc"].iloc[0])

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width / 2, roc, width, yerr=err, capsize=3, label="ROC-AUC (95% bootstrap CI)",
                color="#1976D2", alpha=0.88)
    b2 = ax.bar(x + width / 2, abl_df["pr_auc"], width, label="PR-AUC", color="#43A047", alpha=0.88)
    ax.axhline(all_roc, color="#1976D2", ls="--", lw=1.1, alpha=0.5, label=f"All-feature ROC-AUC ({all_roc:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Score (raw model scores)", fontsize=11)
    ax.set_title("Feature Ablation Study - Test Set (ranking metrics from raw scores)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Ablation chart saved -> {out_path}")


# ===========================================================================
# 2. Per-shipping-mode evaluation (no 0.5 baseline)
# ===========================================================================

def per_mode_table(
    test_df:  pd.DataFrame,
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
    y_te:      np.ndarray,
    n_boot:    int = N_BOOT,
) -> pd.DataFrame:
    """
    Per-shipping-mode ranking quality of the final model on TEST.

    ROC-AUC / PR-AUC use RAW scores; mean predicted vs prevalence uses the
    calibrated scores.  flag = "limited" if the ROC-AUC 95% CI lower bound is
    below INFORMATIVE_AUC_FLOOR (this also covers any CI that includes 0.5),
    else "informative".
    """
    rows = []
    for mode in MODES:
        mask = (test_df["Shipping_Mode"] == mode).values
        n = int(mask.sum())
        if n == 0:
            continue
        y_m = y_te[mask]
        prev = float(y_m.mean())
        row = {"mode": mode, "orders": n, "share": n / len(test_df), "prevalence": prev,
               "mean_predicted": float(cal_proba[mask].mean()),
               "calibration_error": float(cal_proba[mask].mean() - prev)}
        if len(np.unique(y_m)) < 2:
            row.update(roc_auc=None, roc_auc_ci_lo=None, roc_auc_ci_hi=None, pr_auc=None,
                       pr_auc_lift=None, flag="limited")
        else:
            roc = float(roc_auc_score(y_m, raw_proba[mask]))
            lo, hi = bootstrap_ci(y_m, raw_proba[mask], roc_auc_score, n=n_boot)
            pr = float(average_precision_score(y_m, raw_proba[mask]))
            limited = (not np.isfinite(lo)) or lo < INFORMATIVE_AUC_FLOOR or lo <= 0.5
            row.update(roc_auc=roc, roc_auc_ci_lo=lo, roc_auc_ci_hi=hi, pr_auc=pr,
                       pr_auc_lift=pr / prev, flag="limited" if limited else "informative")
        rows.append(row)
    return pd.DataFrame(rows)


def print_per_mode(pm: pd.DataFrame) -> None:
    print(f"\n  {'Mode':<16}{'n':>6}{'prev':>8}{'ROC-AUC [95% CI]':>28}{'PR-AUC':>9}{'lift':>7}  flag")
    print(f"  {_sep('-', 84)}")
    for _, r in pm.iterrows():
        print(f"  {r['mode']:<16}{r['orders']:>6,}{r['prevalence']:>8.3f}"
              f"{r['roc_auc']:>8.4f} [{r['roc_auc_ci_lo']:.4f}, {r['roc_auc_ci_hi']:.4f}]"
              f"{r['pr_auc']:>9.4f}{r['pr_auc_lift']:>7.2f}  {r['flag']}")


# ===========================================================================
# 3. Rolling-origin expanding-window backtest
# ===========================================================================

def rolling_backtest(
    orders:   pd.DataFrame,
    n_folds:  int = 5,
) -> pd.DataFrame:
    """
    5 expanding-window folds: train = everything up to the fold boundary,
    val = next 5% (calibration + threshold), test = following 10% (reported).
    Reports ROC-AUC/PR-AUC (raw scores), PR-AUC lift over the fold prevalence
    and the decision metrics at the fold's own validation-chosen threshold.
    """
    n = len(orders)
    min_train_frac = 0.40
    rows = []
    fold_boundaries = np.linspace(min_train_frac, 0.80, n_folds)

    print(f"\n  {'Fold':>5}  {'Train N':>8}  {'Test N':>7}  {'prev':>6}  "
          f"{'ROC-AUC':>8}  {'PR-AUC':>8}  {'PR lift':>7}  {'TestStart':<11} {'TestEnd':<11}")
    print(f"  {_sep('-', 92)}")

    for fold_i, train_frac in enumerate(fold_boundaries, 1):
        tr_end  = int(np.floor(n * train_frac))
        val_end = int(np.floor(n * (train_frac + 0.05)))
        te_end  = min(int(np.floor(n * (train_frac + 0.15))), n)

        if val_end >= te_end:
            continue

        fold_train = orders.iloc[:tr_end].copy()
        fold_val   = orders.iloc[tr_end:val_end].copy()
        fold_test  = orders.iloc[val_end:te_end].copy()

        if len(fold_val) < 50 or len(fold_test) < 50:
            continue
        if fold_test[TARGET_COL].sum() == 0:
            continue

        res = _fit_eval(fold_train, fold_val, fold_test, CATEGORICAL_FEATURES, NUMERIC_FEATURES)

        te_start = fold_test[DATE_COL].min().date()
        te_end_d = fold_test[DATE_COL].max().date()

        print(
            f"  {fold_i:>5}  {len(fold_train):>8,}  {len(fold_test):>7,}  {res['prevalence']:>6.3f}  "
            f"{res['roc_auc']:>8.4f}  {res['pr_auc']:>8.4f}  {res['pr_auc_lift']:>7.3f}  "
            f"{str(te_start):<11} {str(te_end_d):<11}"
        )
        rows.append({
            "fold": fold_i, "train_n": len(fold_train), "test_n": len(fold_test),
            "test_start": str(te_start), "test_end": str(te_end_d),
            **res,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print(f"  {_sep('-', 92)}")
        print(f"  {'Mean':>5}  {'':>8}  {'':>7}  {df['prevalence'].mean():>6.3f}  "
              f"{df['roc_auc'].mean():>8.4f}  {df['pr_auc'].mean():>8.4f}  {df['pr_auc_lift'].mean():>7.3f}")
    return df


def save_backtest_plot(bt_df: pd.DataFrame, out_path: Path) -> None:
    if len(bt_df) == 0:
        print("  No backtest data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(bt_df["fold"], bt_df["roc_auc"], "o-", color="#1976D2", lw=2, ms=7, label="ROC-AUC")
    ax.plot(bt_df["fold"], bt_df["pr_auc"],  "s-", color="#43A047", lw=2, ms=7, label="PR-AUC")

    mean_roc = bt_df["roc_auc"].mean()
    mean_pra = bt_df["pr_auc"].mean()
    ax.axhline(mean_roc, color="#1976D2", ls="--", lw=1, alpha=0.5, label=f"Mean ROC-AUC ({mean_roc:.4f})")
    ax.axhline(mean_pra, color="#43A047", ls="--", lw=1, alpha=0.5, label=f"Mean PR-AUC ({mean_pra:.4f})")

    ax.set_xlabel("Expanding-window fold", fontsize=11)
    ax.set_ylabel("Score (raw model scores)", fontsize=11)
    ax.set_title("Rolling-Origin Backtest - ROC-AUC & PR-AUC per Fold", fontsize=12)
    ax.set_xticks(bt_df["fold"])
    ax.set_xticklabels([f"F{int(f)}\n{s}" for f, s in zip(bt_df["fold"], bt_df["test_start"])], fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Backtest plot saved -> {out_path}")


# ===========================================================================
# 4. Secondary target: aux_target_cancel_fraud
# ===========================================================================

def evaluate_secondary_target(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Robustness check only - NOT a production model."""
    AUX = "aux_target_cancel_fraud"
    print(f"\n  Aux target positive rate -> train: {train_df[AUX].mean():.4f}  test: {test_df[AUX].mean():.4f}")
    if val_df[AUX].sum() < 5:
        print("  [WARN] Too few positives in val for secondary target - skipping calibration.")

    res = _fit_eval(train_df, val_df, test_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES, target=AUX)

    print(f"\n  {'Metric':<14}  {'Value':>8}")
    print(f"  {_sep('-', 24)}")
    for k in ["roc_auc", "pr_auc", "f1", "recall", "precision"]:
        print(f"  {k:<14}  {res[k]:>8.4f}")
    print(
        f"\n  [INFO] Secondary target (cancel/fraud) ROC-AUC = {res['roc_auc']:.4f}.  "
        "Robustness check only; the production target remains Late_delivery_risk."
    )


# ===========================================================================
# 5. Population Stability Index (PSI)
# ===========================================================================

def _psi_one(train_vals: np.ndarray, test_vals: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI for a single feature using equal-frequency bins on train.
    PSI < 0.1 no significant shift | 0.1-0.2 moderate | > 0.2 major
    """
    pcts  = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(train_vals[~np.isnan(train_vals)], pcts))

    if len(edges) < 3:
        return 0.0

    def _safe_hist(vals, edges):
        counts, _ = np.histogram(vals[~np.isnan(vals)], bins=edges)
        freq = counts / max(counts.sum(), 1)
        return np.where(freq == 0, 1e-4, freq)

    f_tr = _safe_hist(train_vals, edges)
    f_te = _safe_hist(test_vals,  edges)
    return float(np.sum((f_te - f_tr) * np.log(f_te / f_tr)))


def compute_psi(orders: pd.DataFrame, tr_end: int, val_end: int, prep) -> pd.DataFrame:
    """PSI for top-8 features between train and test (transformed space)."""
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
        flag  = "OK" if psi < 0.1 else "Moderate" if psi < 0.2 else "Major"
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
    print("  RTO Shield AI - Step 7: Robustness, Ablations, Backtest & Drift")
    print("=" * 74)

    print("\n[0] Loading data ...")
    orders, train_df, val_df, test_df, tr_end, val_end, n = _load_splits()
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    prep        = joblib.load(MODELS_DIR / "preprocessor.joblib")
    final_model = joblib.load(MODELS_DIR / "final_model.joblib")
    final_cal   = joblib.load(MODELS_DIR / "final_calibrator.joblib")
    X_te_full   = prep.transform(_select_feature_cols(test_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    y_te        = test_df[TARGET_COL].values
    raw_proba   = final_model.predict_proba(X_te_full)[:, 1]
    cal_proba   = final_cal.predict(raw_proba)

    print(f"\n{'=' * 74}\n[1] Feature Ablation Study (each model: own validation max-F1 threshold)\n{'=' * 74}")
    abl_df, verdict = run_ablations(train_df, val_df, test_df)
    save_ablation_chart(abl_df, OUTPUTS_DIR / "ablation_chart.png")
    abl_df.to_csv(OUTPUTS_DIR / "ablation_results.csv", index=False)
    (OUTPUTS_DIR / "ablation_verdict.txt").write_text(verdict + "\n", encoding="utf-8")
    print(f"  Ablation results saved -> {OUTPUTS_DIR}/ablation_results.csv")

    print(f"\n{'=' * 74}\n[2] Per-Shipping-Mode Evaluation (TEST, final model, raw-score ranking)\n{'=' * 74}")
    pm = per_mode_table(test_df, raw_proba, cal_proba, y_te)
    print_per_mode(pm)
    pm.to_csv(OUTPUTS_DIR / "per_mode_results.csv", index=False)

    print(f"\n{'=' * 74}\n[3] Rolling-Origin Expanding-Window Backtest (5 folds)\n{'=' * 74}")
    bt_df = rolling_backtest(orders, n_folds=5)
    save_backtest_plot(bt_df, OUTPUTS_DIR / "backtest_plot.png")
    if len(bt_df) > 0:
        bt_df.to_csv(OUTPUTS_DIR / "backtest_results.csv", index=False)
        print(f"  Backtest results saved -> {OUTPUTS_DIR}/backtest_results.csv")

    print(f"\n{'=' * 74}\n[4] Secondary Target: aux_target_cancel_fraud  (robustness check only)\n{'=' * 74}")
    evaluate_secondary_target(train_df, val_df, test_df)

    print(f"\n{'=' * 74}\n[5] Population Stability Index - Top-8 Features (Train vs Test)\n{'=' * 74}")
    psi_df = compute_psi(orders, tr_end, val_end, prep)
    print_psi_table(psi_df)
    psi_df.to_csv(OUTPUTS_DIR / "psi_results.csv", index=False)
    print(f"\n  PSI results saved -> {OUTPUTS_DIR}/psi_results.csv")

    print("\n" + "=" * 74)
    print("  Robustness analysis complete.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
