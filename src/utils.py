"""
src/utils.py
============
Shared utility functions for RTO Shield AI - Model 1.

Functions
---------
load_raw()                 – Load the raw CSV correctly (latin-1, stripped cols).
chrono_split(df)           – Order-level chronological 70/15/15 split.
compute_metrics()          – Full evaluation metric bundle.
precision_at_k()           – Precision among top-k% predicted-probability rows.
expected_calibration_error() – ECE for probability calibration assessment.
bootstrap_ci()             – 95 % bootstrap confidence interval for any scalar metric.
save_table()               – Persist a DataFrame as a CSV artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    RAW_CSV, TARGET_COL, DATE_COL, ORDER_ID_COL,
    TRAIN_FRAC, VAL_FRAC, SEED,
)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """
    Load the raw DataCo CSV with the correct encoding and clean column names.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with whitespace stripped from every column name.
    """
    df = pd.read_csv(RAW_CSV, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Chronological Split
# ---------------------------------------------------------------------------

def chrono_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* chronologically at the Order level to avoid leakage across
    line-items that belong to the same order.

    Strategy
    --------
    1. Parse order date; derive the *earliest* date seen for each Order Id.
    2. Sort unique Order Ids by that date.
    3. Cut at TRAIN_FRAC and TRAIN_FRAC+VAL_FRAC boundaries.
    4. Filter rows by Order Id membership → train / val / test DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (all rows, all columns).

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Non-overlapping splits that preserve row order within each split.
    """
    df = df.copy()

    # Parse dates (coerce bad values to NaT) - support both raw CSV and processed parquet column names
    date_col = (
        DATE_COL if DATE_COL in df.columns
        else ("order date (DateOrders)" if "order date (DateOrders)" in df.columns else DATE_COL)
    )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Earliest date per order (handles multi-item orders)
    order_dates = (
        df.groupby(ORDER_ID_COL)[date_col]
        .min()
        .sort_values()
        .reset_index()
        .rename(columns={date_col: "first_date"})
    )

    n_orders = len(order_dates)
    train_end = int(np.floor(n_orders * TRAIN_FRAC))
    val_end   = int(np.floor(n_orders * (TRAIN_FRAC + VAL_FRAC)))

    train_ids = set(order_dates.iloc[:train_end][ORDER_ID_COL])
    val_ids   = set(order_dates.iloc[train_end:val_end][ORDER_ID_COL])
    test_ids  = set(order_dates.iloc[val_end:][ORDER_ID_COL])

    train_df = df[df[ORDER_ID_COL].isin(train_ids)].copy()
    val_df   = df[df[ORDER_ID_COL].isin(val_ids)].copy()
    test_df  = df[df[ORDER_ID_COL].isin(test_ids)].copy()

    # Sanity: no row should appear in more than one split
    assert len(train_ids & val_ids) == 0,  "Overlap: train ∩ val"
    assert len(train_ids & test_ids) == 0, "Overlap: train ∩ test"
    assert len(val_ids   & test_ids) == 0, "Overlap: val ∩ test"
    assert (len(train_df) + len(val_df) + len(test_df)) == len(df), \
        "Row count mismatch after split"

    print(
        f"[chrono_split] "
        f"train={len(train_df):,} rows ({len(train_ids):,} orders) | "
        f"val={len(val_df):,} rows ({len(val_ids):,} orders) | "
        f"test={len(test_df):,} rows ({len(test_ids):,} orders)"
    )

    # Print date ranges for verification
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        lo, hi = part[date_col].min(), part[date_col].max()
        print(f"  {name:>5}: {lo.date()} → {hi.date()}")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: float = 0.10,
) -> float:
    """
    Precision among the top-k% rows ranked by predicted probability.

    Parameters
    ----------
    y_true : array of 0/1 labels.
    y_prob : array of predicted positive-class probabilities.
    k      : fraction of rows to consider (default 0.10 = top 10 %).

    Returns
    -------
    float : precision@k
    """
    n = max(1, int(np.ceil(len(y_true) * k)))
    top_idx = np.argsort(y_prob)[::-1][:n]
    return float(np.mean(y_true[top_idx]))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) — equal-width probability bins.

    Lower is better; a perfectly calibrated model returns 0.

    Parameters
    ----------
    y_true : array of 0/1 labels.
    y_prob : array of predicted positive-class probabilities.
    n_bins : number of equal-width bins (default 10).

    Returns
    -------
    float : ECE score.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos  = y_true[mask].mean()          # observed frequency
        mean_pred = y_prob[mask].mean()           # mean predicted prob
        ece += (mask.sum() / n) * abs(frac_pos - mean_pred)

    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    y_rank: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute a full evaluation metric bundle for binary classification.

    Parameters
    ----------
    y_true    : 1-D array of true 0/1 labels.
    y_prob    : 1-D array of predicted positive-class probabilities.  Used for
                DECISION and CALIBRATION outputs (threshold metrics, Brier, ECE).
    threshold : decision threshold applied to y_prob (default 0.5).
    y_rank    : optional scores used for RANKING metrics (ROC-AUC, PR-AUC,
                precision@top10%).  Pass the RAW (uncalibrated) model scores:
                isotonic calibration collapses scores into ties, which distorts
                rankings.  Defaults to y_prob.

    Returns
    -------
    dict with keys:
        threshold, precision, recall, f1, accuracy, balanced_accuracy,
        specificity, mcc, roc_auc, pr_auc, brier, precision_at_top10pct, ece
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rank = y_prob if y_rank is None else np.asarray(y_rank)
    y_pred = (y_prob >= threshold).astype(int)

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0

    return {
        "threshold"            : round(threshold, 4),
        "precision"            : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"               : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"                   : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "accuracy"             : round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy"    : round(balanced_accuracy_score(y_true, y_pred), 4),
        "specificity"          : round(specificity, 4),
        "mcc"                  : round(mcc, 4),
        "roc_auc"              : round(roc_auc_score(y_true, rank), 4),
        "pr_auc"               : round(average_precision_score(y_true, rank), 4),
        "brier"                : round(brier_score_loss(y_true, y_prob), 4),
        "precision_at_top10pct": round(precision_at_k(y_true, rank, k=0.10), 4),
        "ece"                  : round(expected_calibration_error(y_true, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Bootstrap Confidence Interval
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n: int = 1000,
    ci: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap 95 % confidence interval for a scalar metric.

    Parameters
    ----------
    y_true    : array of true 0/1 labels.
    y_prob    : array of predicted positive-class probabilities.
    metric_fn : callable(y_true, y_prob) → float.
    n         : number of bootstrap resamples (default 1 000).
    ci        : confidence level (default 0.95).
    seed      : random seed for reproducibility.

    Returns
    -------
    (lower, upper) : tuple of floats — the CI bounds.
    """
    rng    = np.random.default_rng(seed)
    scores = np.empty(n)
    idx    = np.arange(len(y_true))

    scores_list: list[float] = []
    attempts = 0
    max_attempts = n * 10  # allow extra draws to fill n valid resamples

    while len(scores_list) < n and attempts < max_attempts:
        boot_idx = rng.choice(idx, size=len(idx), replace=True)
        try:
            val = metric_fn(y_true[boot_idx], y_prob[boot_idx])
            if np.isfinite(val):
                scores_list.append(val)
        except Exception:
            pass  # skip degenerate draws (e.g. single-class bootstrap sample)
        attempts += 1

    scores = np.array(scores_list) if scores_list else np.array([float("nan")])

    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(scores, 100 * alpha)),
        float(np.percentile(scores, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# Artefact Persistence
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save a DataFrame to CSV (UTF-8, no index by default).

    Parameters
    ----------
    df   : DataFrame to save.
    path : Destination file path; parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[save_table] Saved {len(df):,} rows × {df.shape[1]} cols → {path}")
