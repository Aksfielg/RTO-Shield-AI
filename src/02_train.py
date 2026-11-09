"""
src/02_train.py
===============
Step 3 – Train and evaluate all models for RTO Shield AI - Model 1.

Models trained
--------------
  (a) Baseline LR   – Shipping_Mode + Days_for_shipment_scheduled only
  (b) Full LR       – All pre-dispatch features (StandardScaler, max_iter=2000)
  (c) Random Forest – 300 trees, n_jobs=-1
  (d) XGBoost       – 400 estimators, max_depth=6, lr=0.05
  (e) LightGBM      – matching settings

Outputs
-------
  models/baseline_lr.joblib
  models/logistic_regression.joblib
  models/random_forest.joblib
  models/xgboost.joblib
  models/lightgbm.joblib
  models/preprocessor.joblib           ← full-feature preprocessor
  models/baseline_preprocessor.joblib ← baseline preprocessor
  outputs/metrics_model_comparison.csv
  outputs/roc_curves.png
  outputs/pr_curves.png

Usage
-----
  python src/02_train.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
import xgboost  as xgb
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Make src/ importable when called from the project root
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    TARGET_COL,
    ORDER_ID_COL,
    SEED,
    TRAIN_FRAC,
    VAL_FRAC,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BASELINE_CAT_FEATURES,
    BASELINE_NUM_FEATURES,
)
from features import build_feature_matrix
from utils   import compute_metrics

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"
DATE_COL      = "order_date"       # column name in the order-level parquet


# ===========================================================================
# 1. Chronological split (order-level parquet — one row per order)
# ===========================================================================

def chrono_split_orders(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological 70/15/15 split on the order-level table.

    The parquet already has one row per order, so we sort by order_date
    directly and cut by row-count fractions.  We confirm no overlap in
    Order Id sets.
    """
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    n          = len(df)
    train_end  = int(np.floor(n * TRAIN_FRAC))
    val_end    = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    # ── Overlap check ────────────────────────────────────────────────────────
    train_ids = set(train_df[ORDER_ID_COL])
    val_ids   = set(val_df[ORDER_ID_COL])
    test_ids  = set(test_df[ORDER_ID_COL])

    assert len(train_ids & val_ids)  == 0, "Overlap detected: train ∩ val"
    assert len(train_ids & test_ids) == 0, "Overlap detected: train ∩ test"
    assert len(val_ids   & test_ids) == 0, "Overlap detected: val ∩ test"
    assert len(train_df) + len(val_df) + len(test_df) == n, "Row-count mismatch"

    return train_df, val_df, test_df


def _print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    sep = "─" * 70
    print(f"\n  {sep}")
    print(f"  {'Split':<10} {'Orders':>8}  {'Date range':<35}  {'Pos rate':>9}")
    print(f"  {sep}")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        lo  = part[DATE_COL].min().date()
        hi  = part[DATE_COL].max().date()
        pos = part[TARGET_COL].mean()
        print(
            f"  {name:<10} {len(part):>8,}  "
            f"{str(lo) + ' → ' + str(hi):<35}  "
            f"{pos:>8.4f}"
        )
    print(f"  {sep}\n")


# ===========================================================================
# 2. Build feature matrices
# ===========================================================================

def build_matrices(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> dict:
    """
    Returns a dict with keys:
        'full'     → (X_train, X_val, X_test, feat_names, preprocessor)
        'baseline' → (X_train, X_val, X_test, feat_names, preprocessor)
        'y_train', 'y_val', 'y_test'
    """
    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    # ── Full feature set ─────────────────────────────────────────────────────
    print("\n[Full feature set]")
    X_tr, (X_val_a, X_te), feat_names, prep = build_feature_matrix(
        train_df,
        [val_df, test_df],
        cat_cols=CATEGORICAL_FEATURES,
        num_cols=NUMERIC_FEATURES,
        save_preprocessor=True,
        preprocessor_path=MODELS_DIR / "preprocessor.joblib",
    )

    # ── Baseline feature set (shipping mode + scheduled days) ────────────────
    print("\n[Baseline feature set]")
    X_tr_b, (X_val_b, X_te_b), feat_names_b, prep_b = build_feature_matrix(
        train_df,
        [val_df, test_df],
        cat_cols=BASELINE_CAT_FEATURES,
        num_cols=BASELINE_NUM_FEATURES,
        save_preprocessor=True,
        preprocessor_path=MODELS_DIR / "baseline_preprocessor.joblib",
    )

    return {
        "full":     (X_tr, X_val_a, X_te, feat_names, prep),
        "baseline": (X_tr_b, X_val_b, X_te_b, feat_names_b, prep_b),
        "y_train":  y_train,
        "y_val":    y_val,
        "y_test":   y_test,
    }


# ===========================================================================
# 3. Train all models
# ===========================================================================

def train_all(mats: dict) -> dict[str, object]:
    X_tr,   X_val,   X_te   = mats["full"][:3]
    X_tr_b, X_val_b, X_te_b = mats["baseline"][:3]
    y_train = mats["y_train"]

    models: dict[str, object] = {}

    # (a) Baseline LR ────────────────────────────────────────────────────────
    print("\n[a] Training Baseline Logistic Regression …")
    blr = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
    blr.fit(X_tr_b, y_train)
    models["baseline_lr"] = blr
    print("    Done.")

    # (b) Full Logistic Regression ────────────────────────────────────────────
    print("\n[b] Training Full Logistic Regression …")
    lr = LogisticRegression(max_iter=2000, random_state=SEED, n_jobs=-1)
    lr.fit(X_tr, y_train)
    models["logistic_regression"] = lr
    print("    Done.")

    # (c) Random Forest ───────────────────────────────────────────────────────
    print("\n[c] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_tr, y_train)
    models["random_forest"] = rf
    print("    Done.")

    # (d) XGBoost ─────────────────────────────────────────────────────────────
    print("\n[d] Training XGBoost …")
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(
        X_tr, y_train,
        eval_set=[(X_val, mats["y_val"])],
        verbose=False,
    )
    models["xgboost"] = xgb_model
    print("    Done.")

    # (e) LightGBM ────────────────────────────────────────────────────────────
    print("\n[e] Training LightGBM …")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr, y_train,
        eval_set=[(X_val, mats["y_val"])],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    models["lightgbm"] = lgb_model
    print("    Done.")

    return models


# ===========================================================================
# 4. Evaluate all models
# ===========================================================================

def _predict(model, X_full, X_base, name: str):
    """Return predicted probabilities; use baseline matrix for baseline_lr."""
    if name == "baseline_lr":
        return model.predict_proba(X_base)[:, 1]
    return model.predict_proba(X_full)[:, 1]


def evaluate_all(
    models: dict,
    mats:   dict,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute metrics on val AND test for each model.
    Returns a tidy DataFrame with columns:
        model, split, threshold, precision, recall, f1,
        roc_auc, pr_auc, brier, precision_at_top10pct
    """
    X_val,  y_val  = mats["full"][1], mats["y_val"]
    X_te,   y_te   = mats["full"][2], mats["y_test"]
    X_val_b         = mats["baseline"][1]
    X_te_b          = mats["baseline"][2]

    rows = []
    for name, model in models.items():
        for split, X_full, X_base, y in [
            ("val",  X_val,  X_val_b, y_val),
            ("test", X_te,   X_te_b,  y_te),
        ]:
            proba = _predict(model, X_full, X_base, name)
            m     = compute_metrics(y, proba, threshold=threshold)
            rows.append({"model": name, "split": split, **m})

    return pd.DataFrame(rows)


# ===========================================================================
# 5. Print comparison tables
# ===========================================================================

_METRIC_DISPLAY = [
    ("precision",             "Precision"),
    ("recall",                "Recall"),
    ("f1",                    "F1"),
    ("roc_auc",               "ROC-AUC"),
    ("pr_auc",                "PR-AUC"),
    ("brier",                 "Brier↓"),
    ("precision_at_top10pct", "P@Top10%"),
]

_MODEL_LABELS = {
    "baseline_lr":        "Baseline LR",
    "logistic_regression":"Full LR",
    "random_forest":      "Random Forest",
    "xgboost":            "XGBoost",
    "lightgbm":           "LightGBM",
}


def _pivot_table(metrics_df: pd.DataFrame, split: str) -> str:
    sub = metrics_df[metrics_df["split"] == split].copy()
    sub["model_label"] = sub["model"].map(_MODEL_LABELS)

    lines = []
    hdr_model = f"{'Model':<20}"
    hdr_cols  = "".join(f"  {disp:>9}" for _, disp in _METRIC_DISPLAY)
    sep       = "─" * (20 + 11 * len(_METRIC_DISPLAY))
    lines.append(sep)
    lines.append(f"{hdr_model}{hdr_cols}")
    lines.append(sep)
    for _, row in sub.iterrows():
        line = f"{row['model_label']:<20}"
        for col, _ in _METRIC_DISPLAY:
            line += f"  {row[col]:>9.4f}"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def print_comparison_tables(metrics_df: pd.DataFrame) -> None:
    for split in ("val", "test"):
        print(f"\n{'=' * 70}")
        print(f"  Metrics on {split.upper()} set  (threshold = 0.50)")
        print(f"{'=' * 70}")
        print(_pivot_table(metrics_df, split))


# ===========================================================================
# 6. Confusion matrices for XGBoost and LightGBM
# ===========================================================================

def print_confusion_matrices(models, mats, threshold=0.5):
    X_te = mats["full"][2]
    y_te = mats["y_test"]

    for name in ("xgboost", "lightgbm"):
        proba = models[name].predict_proba(X_te)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()

        label = _MODEL_LABELS[name]
        print(f"\n  Confusion Matrix – {label} (TEST, threshold={threshold})")
        print(f"  {'':20}  Pred 0 (On-time)  Pred 1 (Late)")
        print(f"  {'True 0 (On-time)':<20}  {tn:>16,}  {fp:>13,}")
        print(f"  {'True 1 (Late)':<20}  {fn:>16,}  {tp:>13,}")
        print(
            f"  TPR={tp/(tp+fn):.4f}  FPR={fp/(fp+tn):.4f}  "
            f"TNR={tn/(tn+fp):.4f}  FNR={fn/(fn+tp):.4f}"
        )


# ===========================================================================
# 7. ROC and PR curve plots
# ===========================================================================

def _get_all_curves(models, mats):
    """Compute ROC and PR curve data for every model on val and test."""
    curves = {}
    for split, X_full, X_base, y in [
        ("val",  mats["full"][1], mats["baseline"][1], mats["y_val"]),
        ("test", mats["full"][2], mats["baseline"][2], mats["y_test"]),
    ]:
        for name, model in models.items():
            proba = _predict(model, X_full, X_base, name)
            fpr, tpr, _  = roc_curve(y, proba)
            roc_auc_val  = auc(fpr, tpr)
            prec, rec, _ = precision_recall_curve(y, proba)
            pr_auc_val   = auc(rec, prec)
            curves[(name, split)] = dict(
                fpr=fpr, tpr=tpr, roc_auc=roc_auc_val,
                prec=prec, rec=rec, pr_auc=pr_auc_val,
                label=_MODEL_LABELS[name],
            )
    return curves


_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def save_roc_curves(models, mats, out_path: Path) -> None:
    curves = _get_all_curves(models, mats)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, split in zip(axes, ["val", "test"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
        for (name, s), c_data in curves.items():
            if s != split:
                continue
            ax.plot(
                c_data["fpr"], c_data["tpr"],
                label=f"{c_data['label']}  (AUC={c_data['roc_auc']:.4f})",
                lw=1.8,
            )
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"ROC Curves – {split.upper()} set", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curves saved → {out_path}")


def save_pr_curves(models, mats, out_path: Path) -> None:
    curves   = _get_all_curves(models, mats)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, split in zip(axes, ["val", "test"]):
        # baseline = overall positive rate
        y = mats["y_val"] if split == "val" else mats["y_test"]
        base_rate = y.mean()
        ax.axhline(base_rate, color="k", ls="--", lw=1, alpha=0.4,
                   label=f"No-skill ({base_rate:.3f})")
        for (name, s), c_data in curves.items():
            if s != split:
                continue
            ax.plot(
                c_data["rec"], c_data["prec"],
                label=f"{c_data['label']}  (AUC={c_data['pr_auc']:.4f})",
                lw=1.8,
            )
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"Precision-Recall Curves – {split.upper()} set", fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PR curves saved → {out_path}")


# ===========================================================================
# 8. Persist models and metrics
# ===========================================================================

def save_models(models: dict) -> None:
    for name, model in models.items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"  Saved model → {path}")


def save_metrics(metrics_df: pd.DataFrame) -> None:
    path = OUTPUTS_DIR / "metrics_model_comparison.csv"
    metrics_df.to_csv(path, index=False)
    print(f"  Metrics saved → {path}")


# ===========================================================================
# 9. Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  RTO Shield AI – Step 3: Train & Evaluate All Models")
    print("=" * 70)

    # ── Load orders ──────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {ORDER_PARQUET.name} …")
    orders = pd.read_parquet(ORDER_PARQUET)
    print(f"      Shape: {orders.shape[0]:,} rows × {orders.shape[1]} cols")

    # ── Chronological split ──────────────────────────────────────────────────
    print("\n[2/6] Chronological 70/15/15 split …")
    train_df, val_df, test_df = chrono_split_orders(orders)
    _print_split_summary(train_df, val_df, test_df)

    # ── Feature matrices ─────────────────────────────────────────────────────
    print("[3/6] Building feature matrices …")
    mats = build_matrices(train_df, val_df, test_df)
    print(f"\n  Full feature set : {mats['full'][0].shape[1]} features")
    print(f"  Baseline         : {mats['baseline'][0].shape[1]} features")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[4/6] Training models …")
    models = train_all(mats)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5/6] Evaluating models …")
    metrics_df = evaluate_all(models, mats, threshold=0.5)
    print_comparison_tables(metrics_df)

    print("\n" + "=" * 70)
    print("  Confusion Matrices (TEST set, threshold=0.50)")
    print("=" * 70)
    print_confusion_matrices(models, mats)

    # ── Plots + save ──────────────────────────────────────────────────────────
    print("\n[6/6] Saving plots, models, and metrics …")
    save_roc_curves(models, mats, OUTPUTS_DIR / "roc_curves.png")
    save_pr_curves (models, mats, OUTPUTS_DIR / "pr_curves.png")
    save_models(models)
    save_metrics(metrics_df)

    print("\n" + "=" * 70)
    print("  Training complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

