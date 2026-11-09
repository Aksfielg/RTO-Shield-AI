"""
src/05_tune_multiseed.py
========================
Step 6 – Hyperparameter Tuning, Multi-Seed Stability & Bootstrap CIs.

Pipeline
--------
  1. RandomizedSearchCV (n_iter=30) on TRAIN only with TimeSeriesSplit(n_splits=4),
     scoring ROC-AUC.  Search space covers 7 LightGBM hyperparameters.
  2. Compare tuned vs default on VALIDATION; keep the tuned one only if it improves.
  3. Select threshold on VALIDATION (max-F1 on calibrated probabilities),
     then evaluate the final model ONCE on TEST.
  4. Train with 10 seeds; report mean ± std for ROC-AUC, PR-AUC, F1, Recall, Brier.
  5. Bootstrap 95 % CIs (1000 resamples) for test ROC-AUC and PR-AUC of the final
     model AND for the difference vs the baseline LR.
  6. Save outputs/final_results.csv and models/final_model.joblib.

Usage
-----
  python src/05_tune_multiseed.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

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
)
from features  import _select_feature_cols
from utils     import compute_metrics, bootstrap_ci
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ORDER_PARQUET  = PROCESSED_DIR / "orders.parquet"
DATE_COL       = "order_date"
N_ITER         = 30
N_SPLITS_CV    = 4
N_SEEDS        = 10
N_BOOTSTRAP    = 1000
MULTI_SEEDS    = list(range(N_SEEDS))   # 0 … 9


# ===========================================================================
# Helpers
# ===========================================================================

def _sep(char="─", w=74): return char * w


def _load_splits():
    """Return (train_df, val_df, test_df, prep, feat_names)."""
    orders = (
        pd.read_parquet(ORDER_PARQUET)
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    n       = len(orders)
    tr_end  = int(np.floor(n * TRAIN_FRAC))
    val_end = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))

    train_df = orders.iloc[:tr_end].copy()
    val_df   = orders.iloc[tr_end:val_end].copy()
    test_df  = orders.iloc[val_end:].copy()

    prep       = joblib.load(MODELS_DIR / "preprocessor.joblib")
    feat_names = list(prep.get_feature_names_out())

    def _X(df):
        return prep.transform(_select_feature_cols(df, CATEGORICAL_FEATURES, NUMERIC_FEATURES))

    X_tr  = _X(train_df)
    X_val = _X(val_df)
    X_te  = _X(test_df)

    y_tr  = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_te  = test_df[TARGET_COL].values

    return (
        train_df, val_df, test_df,
        X_tr, X_val, X_te,
        y_tr, y_val, y_te,
        prep, feat_names,
    )


def _make_lgb(params: dict | None = None, seed: int = SEED) -> lgb.LGBMClassifier:
    """Construct a LGBMClassifier with default or provided params."""
    base = dict(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    if params:
        base.update(params)
    return lgb.LGBMClassifier(**base)


def _calibrate(model, X_val, y_val):
    """Fit IsotonicRegression on val predictions."""
    raw_val = model.predict_proba(X_val)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_val, y_val)
    return ir


def _calibrated_proba(model, ir, X):
    return ir.predict(model.predict_proba(X)[:, 1])


def _select_threshold_max_f1(y_val, proba_val):
    """Select threshold that maximises F1 on validation set."""
    thresholds = np.linspace(0.01, 0.99, 990)
    best_thr, best_f1 = 0.5, 0.0
    for t in thresholds:
        yp = (proba_val >= t).astype(int)
        tp = int(((yp == 1) & (y_val == 1)).sum())
        fp = int(((yp == 1) & (y_val == 0)).sum())
        fn = int(((yp == 0) & (y_val == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr), float(best_f1)


# ===========================================================================
# 1. RandomizedSearchCV
# ===========================================================================

PARAM_DIST = {
    "learning_rate":    loguniform(1e-3, 0.3),      # ~0.001 … 0.3
    "max_depth":        [3, 4, 5, 6, 7, 8, 10, -1],
    "n_estimators":     [100, 200, 300, 400, 500, 600, 800],
    "subsample":        uniform(0.5, 0.5),           # 0.5 … 1.0
    "colsample_bytree": uniform(0.5, 0.5),           # 0.5 … 1.0
    "min_child_weight": [1, 3, 5, 10, 20, 50],
    "reg_lambda":       loguniform(1e-2, 100),       # ~0.01 … 100
}


def run_random_search(X_tr, y_tr) -> tuple[dict, float]:
    """
    Run RandomizedSearchCV on training data with TimeSeriesSplit.
    Returns (best_params, best_cv_roc_auc).
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    base = _make_lgb(seed=SEED)

    rscv = RandomizedSearchCV(
        estimator=base,
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=SEED,
        verbose=0,
        refit=False,            # we'll refit manually with early stopping
    )

    t0 = time.time()
    rscv.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    best_params = rscv.best_params_
    best_score  = rscv.best_score_

    print(f"\n  RandomizedSearchCV finished in {elapsed:.1f}s")
    print(f"  Best CV ROC-AUC : {best_score:.4f}")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<22} = {v_str}")

    return best_params, best_score


# ===========================================================================
# 2. Compare tuned vs default on validation
# ===========================================================================

def compare_tuned_vs_default(
    X_tr, y_tr, X_val, y_val, best_params: dict
) -> tuple[lgb.LGBMClassifier, str, float, float]:
    """
    Train tuned model, compare val ROC-AUC with default.
    Returns (winner_model, winner_label, default_val_roc, tuned_val_roc).
    """
    # Default
    default_model = joblib.load(MODELS_DIR / "lightgbm.joblib")
    p_default = default_model.predict_proba(X_val)[:, 1]
    roc_default = roc_auc_score(y_val, p_default)

    # Tuned – retrain on full train set with best params + early stopping on val
    tuned_params = {k: v for k, v in best_params.items()}
    tuned_model = _make_lgb(params=tuned_params, seed=SEED)
    tuned_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    p_tuned = tuned_model.predict_proba(X_val)[:, 1]
    roc_tuned = roc_auc_score(y_val, p_tuned)

    improved = roc_tuned > roc_default
    winner_label = "tuned" if improved else "default"
    winner_model = tuned_model if improved else default_model

    print(f"\n  {'Model':<12}  {'Val ROC-AUC':>12}")
    print(f"  {_sep('-', 28)}")
    print(f"  {'Default':<12}  {roc_default:>12.4f}")
    tuned_arrow = " ← WINNER" if improved else ""
    default_arrow = " ← WINNER" if not improved else ""
    print(f"  {'Default':<12}  {roc_default:>12.4f}{default_arrow}")
    print(f"  {'Tuned':<12}  {roc_tuned:>12.4f}{tuned_arrow}")
    print(f"\n  → Keeping [{winner_label}] model (Δ ROC-AUC = {roc_tuned - roc_default:+.4f})")
    if not improved:
        print(f"  [WARN] Tuning did not improve validation ROC-AUC; default model retained.")

    return winner_model, winner_label, roc_default, roc_tuned


# ===========================================================================
# 3. Threshold on val → evaluate on test
# ===========================================================================

def evaluate_final(
    model, X_val, y_val, X_te, y_te
) -> tuple[float, dict, IsotonicRegression]:
    """
    Calibrate on val → select threshold on val → evaluate on test.
    Returns (threshold, test_metrics, calibrator).
    """
    ir = _calibrate(model, X_val, y_val)

    cal_val = _calibrated_proba(model, ir, X_val)
    cal_te  = _calibrated_proba(model, ir, X_te)

    threshold, val_f1 = _select_threshold_max_f1(y_val, cal_val)
    print(f"\n  Threshold (max-F1 on val calibrated proba) : {threshold:.4f}  "
          f"(val F1={val_f1:.4f})")

    metrics = compute_metrics(y_te, cal_te, threshold=threshold)
    return threshold, metrics, ir, cal_te


def print_final_metrics(metrics: dict, threshold: float, label: str = "FINAL TEST") -> None:
    print(f"\n  {'='*60}")
    print(f"  {label}  (threshold={threshold:.4f})")
    print(f"  {'='*60}")
    col_order = [
        ("precision",             "Precision"),
        ("recall",                "Recall"),
        ("f1",                    "F1"),
        ("roc_auc",               "ROC-AUC"),
        ("pr_auc",                "PR-AUC"),
        ("brier",                 "Brier↓"),
        ("precision_at_top10pct", "P@Top10%"),
        ("ece",                   "ECE↓"),
    ]
    for col, label_str in col_order:
        if col in metrics:
            print(f"  {label_str:<24} {metrics[col]:.4f}")


# ===========================================================================
# 4. Multi-seed training
# ===========================================================================

def multi_seed_train(
    best_params: dict,
    X_tr, y_tr, X_val, y_val, X_te, y_te,
    winner_label: str,
) -> pd.DataFrame:
    """
    Train the final model (tuned or default params) with 10 different seeds.
    Return DataFrame of per-seed test metrics.
    """
    params = best_params if winner_label == "tuned" else {}

    metric_keys = ["roc_auc", "pr_auc", "f1", "recall", "brier"]
    rows = []

    print(f"\n  {'Seed':>5}  {'ROC-AUC':>8}  {'PR-AUC':>8}  {'F1':>8}  "
          f"{'Recall':>8}  {'Brier↓':>8}  {'Thr':>6}")
    print(f"  {_sep('-', 60)}")

    for seed in MULTI_SEEDS:
        m = _make_lgb(params=params if params else None, seed=seed)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        ir_s      = _calibrate(m, X_val, y_val)
        cal_val_s = _calibrated_proba(m, ir_s, X_val)
        cal_te_s  = _calibrated_proba(m, ir_s, X_te)

        thr_s, _ = _select_threshold_max_f1(y_val, cal_val_s)
        mets_s   = compute_metrics(y_te, cal_te_s, threshold=thr_s)

        row = {"seed": seed, "threshold": thr_s}
        row.update({k: mets_s[k] for k in metric_keys})
        rows.append(row)

        print(
            f"  {seed:>5}  {mets_s['roc_auc']:>8.4f}  {mets_s['pr_auc']:>8.4f}  "
            f"{mets_s['f1']:>8.4f}  {mets_s['recall']:>8.4f}  "
            f"{mets_s['brier']:>8.4f}  {thr_s:>6.4f}"
        )

    df = pd.DataFrame(rows)
    print(f"  {_sep('-', 60)}")
    print(f"  {'Mean':>5}  ", end="")
    for k in metric_keys:
        print(f"{df[k].mean():>8.4f}  ", end="")
    print()
    print(f"  {'Std':>5}  ", end="")
    for k in metric_keys:
        print(f"{df[k].std():>8.4f}  ", end="")
    print()

    return df


def print_seed_summary(seed_df: pd.DataFrame) -> None:
    metric_keys = ["roc_auc", "pr_auc", "f1", "recall", "brier"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1", "Recall", "Brier↓"]

    print(f"\n  {'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {_sep('-', 48)}")
    for k, label in zip(metric_keys, metric_labels):
        print(
            f"  {label:<12}  {seed_df[k].mean():>8.4f}  "
            f"{seed_df[k].std():>8.4f}  "
            f"{seed_df[k].min():>8.4f}  "
            f"{seed_df[k].max():>8.4f}"
        )


# ===========================================================================
# 5. Bootstrap confidence intervals
# ===========================================================================

def compute_bootstrap_cis(
    y_te:         np.ndarray,
    final_proba:  np.ndarray,
    baseline_proba: np.ndarray,
    n_boot:       int = N_BOOTSTRAP,
) -> dict:
    """
    Bootstrap 95 % CIs for:
      – final model ROC-AUC
      – final model PR-AUC
      – Δ ROC-AUC (final − baseline)
      – Δ PR-AUC  (final − baseline)
    Returns dict of (lower, upper) tuples.
    """
    rng  = np.random.default_rng(SEED)
    idx  = np.arange(len(y_te))

    roc_final_boots, pra_final_boots = [], []
    roc_diff_boots,  pra_diff_boots  = [], []

    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        y_b   = y_te[b]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            continue   # degenerate sample

        roc_f = roc_auc_score(y_b, final_proba[b])
        pra_f = average_precision_score(y_b, final_proba[b])
        roc_b = roc_auc_score(y_b, baseline_proba[b])
        pra_b = average_precision_score(y_b, baseline_proba[b])

        roc_final_boots.append(roc_f)
        pra_final_boots.append(pra_f)
        roc_diff_boots.append(roc_f - roc_b)
        pra_diff_boots.append(pra_f - pra_b)

    def _ci(boots):
        a = np.array(boots)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return {
        "final_roc_auc":  _ci(roc_final_boots),
        "final_pr_auc":   _ci(pra_final_boots),
        "delta_roc_auc":  _ci(roc_diff_boots),
        "delta_pr_auc":   _ci(pra_diff_boots),
    }


def print_bootstrap_cis(cis: dict, final_metrics: dict) -> None:
    print(f"\n  {'Metric':<26}  {'Point est':>10}  {'95% CI':>22}")
    print(f"  {_sep('-', 62)}")

    point_map = {
        "final_roc_auc": final_metrics["roc_auc"],
        "final_pr_auc":  final_metrics["pr_auc"],
        "delta_roc_auc": None,
        "delta_pr_auc":  None,
    }
    labels = {
        "final_roc_auc": "Final ROC-AUC",
        "final_pr_auc":  "Final PR-AUC",
        "delta_roc_auc": "Δ ROC-AUC (final−baseline)",
        "delta_pr_auc":  "Δ PR-AUC  (final−baseline)",
    }
    for key, (lo, hi) in cis.items():
        pt  = point_map[key]
        pt_str = f"{pt:.4f}" if pt is not None else "  —   "
        ci_str = f"[{lo:.4f}, {hi:.4f}]"
        sig    = ""
        if key.startswith("delta") and lo > 0:
            sig = "  [SIG BETTER]"
        elif key.startswith("delta") and hi < 0:
            sig = "  [SIG WORSE]"
        print(f"  {labels[key]:<26}  {pt_str:>10}  {ci_str:>22}{sig}")


# ===========================================================================
# 6. Save outputs
# ===========================================================================

def save_final_results(
    seed_df:       pd.DataFrame,
    final_metrics: dict,
    cis:           dict,
    threshold:     float,
    winner_label:  str,
    best_params:   dict,
    best_cv_roc:   float,
) -> None:
    rows = []

    # Per-seed rows
    for _, row in seed_df.iterrows():
        rows.append({
            "section": "multi_seed",
            "seed":    int(row["seed"]),
            "metric":  "per_seed",
            "roc_auc": row["roc_auc"],
            "pr_auc":  row["pr_auc"],
            "f1":      row["f1"],
            "recall":  row["recall"],
            "brier":   row["brier"],
        })

    # Summary row
    rows.append({
        "section":  "multi_seed_summary",
        "seed":     -1,
        "metric":   "mean",
        "roc_auc":  seed_df["roc_auc"].mean(),
        "pr_auc":   seed_df["pr_auc"].mean(),
        "f1":       seed_df["f1"].mean(),
        "recall":   seed_df["recall"].mean(),
        "brier":    seed_df["brier"].mean(),
    })
    rows.append({
        "section":  "multi_seed_summary",
        "seed":     -1,
        "metric":   "std",
        "roc_auc":  seed_df["roc_auc"].std(),
        "pr_auc":   seed_df["pr_auc"].std(),
        "f1":       seed_df["f1"].std(),
        "recall":   seed_df["recall"].std(),
        "brier":    seed_df["brier"].std(),
    })

    # Final metrics row
    rows.append({
        "section":  "final_test",
        "seed":     SEED,
        "metric":   "final",
        "roc_auc":  final_metrics["roc_auc"],
        "pr_auc":   final_metrics["pr_auc"],
        "f1":       final_metrics["f1"],
        "recall":   final_metrics["recall"],
        "brier":    final_metrics["brier"],
    })

    df = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / "final_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Final results saved → {out_path}")

    # Also append bootstrap CIs as a small JSON sidecar
    ci_dict = {
        "bootstrap_cis": {k: list(v) for k, v in cis.items()},
        "threshold":      threshold,
        "winner":         winner_label,
        "best_cv_roc":    round(best_cv_roc, 6),
        "best_params":    {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                           for k, v in best_params.items()},
    }
    ci_path = OUTPUTS_DIR / "final_results_meta.json"
    ci_path.write_text(json.dumps(ci_dict, indent=2))
    print(f"  Meta / CI sidecar saved → {ci_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 74)
    print("  RTO Shield AI – Step 6: Tuning, Multi-Seed Stability & Bootstrap CIs")
    print("=" * 74)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data and feature matrices …")
    (
        train_df, val_df, test_df,
        X_tr, X_val, X_te,
        y_tr, y_val, y_te,
        prep, feat_names,
    ) = _load_splits()
    print(f"  train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}  "
          f"features={X_tr.shape[1]}")

    # ── Hyperparameter search ─────────────────────────────────────────────────
    print(f"\n[2/6] RandomizedSearchCV  "
          f"(n_iter={N_ITER}, TimeSeriesSplit(n_splits={N_SPLITS_CV}), "
          f"scoring=roc_auc) …")
    best_params, best_cv_roc = run_random_search(X_tr, y_tr)

    # ── Compare tuned vs default ──────────────────────────────────────────────
    print(f"\n[3/6] Comparing tuned vs default on VALIDATION …")
    print(f"\n  {'='*60}")
    print(f"  Validation Comparison")
    print(f"  {'='*60}")
    winner_model, winner_label, roc_default, roc_tuned = compare_tuned_vs_default(
        X_tr, y_tr, X_val, y_val, best_params
    )

    # ── Final evaluation on TEST ──────────────────────────────────────────────
    print(f"\n[4/6] Threshold selection (VAL) → final evaluation (TEST) …")
    threshold, final_metrics, ir, cal_te = evaluate_final(
        winner_model, X_val, y_val, X_te, y_te
    )
    print_final_metrics(final_metrics, threshold)

    # ── Multi-seed stability ──────────────────────────────────────────────────
    print(f"\n[5/6] Multi-seed training ({N_SEEDS} seeds × full TRAIN) …")
    print(f"\n  {'='*74}")
    print(f"  Per-Seed Test Metrics")
    print(f"  {'='*74}")
    seed_df = multi_seed_train(best_params, X_tr, y_tr, X_val, y_val, X_te, y_te, winner_label)

    print(f"\n  {'='*60}")
    print(f"  Multi-Seed Summary (TEST set, {N_SEEDS} seeds)")
    print(f"  {'='*60}")
    print_seed_summary(seed_df)

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    print(f"\n[6/6] Bootstrap CIs ({N_BOOTSTRAP} resamples) …")

    # Baseline LR calibrated probabilities
    baseline_model  = joblib.load(MODELS_DIR / "baseline_lr.joblib")
    baseline_prep   = joblib.load(MODELS_DIR / "baseline_preprocessor.joblib")
    from config import BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES
    X_val_b = baseline_prep.transform(
        _select_feature_cols(val_df, BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES)
    )
    X_te_b  = baseline_prep.transform(
        _select_feature_cols(test_df, BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES)
    )
    ir_base    = _calibrate(baseline_model, X_val_b, y_val)
    cal_te_b   = _calibrated_proba(baseline_model, ir_base, X_te_b)

    cis = compute_bootstrap_cis(y_te, cal_te, cal_te_b, n_boot=N_BOOTSTRAP)

    print(f"\n  {'='*74}")
    print(f"  Bootstrap 95% Confidence Intervals (TEST set)")
    print(f"  {'='*74}")
    print_bootstrap_cis(cis, final_metrics)

    # ── Save artifacts ────────────────────────────────────────────────────────
    print(f"\n  {'='*60}")
    print(f"  Saving artifacts …")
    print(f"  {'='*60}")
    save_final_results(seed_df, final_metrics, cis, threshold, winner_label, best_params, best_cv_roc)

    # Save final model and its calibrator
    joblib.dump(winner_model, MODELS_DIR / "final_model.joblib")
    joblib.dump(ir,           MODELS_DIR / "final_calibrator.joblib")
    print(f"  Final model saved → {MODELS_DIR}/final_model.joblib")
    print(f"  Final calibrator saved → {MODELS_DIR}/final_calibrator.joblib")

    # Update thresholds.json with the tuned threshold
    thr_json_path = MODELS_DIR / "thresholds.json"
    if thr_json_path.exists():
        thr_data = json.loads(thr_json_path.read_text())
        thr_data["thresholds"]["final_tuned_max_f1"] = round(threshold, 4)
        thr_data["winner_model"] = winner_label
        thr_json_path.write_text(json.dumps(thr_data, indent=2))
        print(f"  Thresholds.json updated → {thr_json_path}")

    print("\n" + "=" * 74)
    print("  Tuning & stability analysis complete.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()

