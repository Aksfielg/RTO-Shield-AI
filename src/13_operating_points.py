"""
src/13_operating_points.py
==========================
Operating points for the shipped model.

Part 1  Thresholds on the CALIBRATED score chosen on VALIDATION only:
        (a) max balanced accuracy  (b) max MCC  (c) max accuracy
        Reported on TEST with 95% bootstrap CIs. A row is flagged when
        calibrated score >= threshold.
Part 2  Review-capacity table: review the top 10/20/30/40% of orders by RAW
        model score and report precision, recall and lift.

Read-only with respect to models/. The only artefact written is
outputs/operating_points.md.

Usage
-----
  python src/13_operating_points.py
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    SEED,
    TARGET_COL,
)

warnings.filterwarnings("ignore", category=UserWarning)

REPORT_PATH = OUTPUTS_DIR / "operating_points.md"
N_BOOT = 1000
CAPACITY_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
METRICS = ["accuracy", "balanced_accuracy", "mcc", "precision", "recall", "specificity", "flagged_share"]
OBJECTIVES = {
    "(a) Max balanced accuracy": "balanced_accuracy",
    "(b) Max MCC": "mcc",
    "(c) Max accuracy": "accuracy",
}


# ===========================================================================
# Split + scoring (same chronological split as src/02_train.py)
# ===========================================================================

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = importlib.util.spec_from_file_location("train02", SRC_DIR / "02_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.chrono_split_orders(pd.read_parquet(PROCESSED_DIR / "orders.parquet"))


def score(df: pd.DataFrame, prep, model, calibrator) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw, calibrated) scores for df using the shipped artefacts."""
    raw = model.predict_proba(prep.transform(df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]))[:, 1]
    return raw, calibrator.predict(raw)


# ===========================================================================
# Metrics
# ===========================================================================

def confusion_metrics(y: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    f = np.asarray(flagged).astype(bool)
    tp, fp = float(np.sum(f & y)), float(np.sum(f & ~y))
    fn, tn = float(np.sum(~f & y)), float(np.sum(~f & ~y))
    return _from_counts(tp, fp, fn, tn)


def _from_counts(tp, fp, fn, tn) -> dict[str, float]:
    n = tp + fp + fn + tn
    div = lambda a, b: a / b if b > 0 else 0.0  # noqa: E731
    recall, spec = div(tp, tp + fn), div(tn, tn + fp)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": div(tp + tn, n),
        "balanced_accuracy": (recall + spec) / 2,
        "mcc": float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0,
        "precision": div(tp, tp + fp),
        "recall": recall,
        "specificity": spec,
        "flagged_share": div(tp + fp, n),
    }


def choose_threshold(y_val: np.ndarray, s_val: np.ndarray, metric: str) -> tuple[float, float]:
    """Threshold t (flag when score >= t) maximising `metric` on VALIDATION data.

    Candidates are the distinct validation scores (the calibrated score is a step
    function, so nothing between two steps can change the confusion matrix).
    Ties on the objective resolve to the lowest threshold, so the choice is deterministic.
    Returns (threshold, validation value of the metric).
    """
    candidates = np.unique(s_val)
    values = np.array([confusion_metrics(y_val, s_val >= t)[metric] for t in candidates])
    best = int(np.argmax(values))          # argmax returns the first (lowest-threshold) maximum
    return float(candidates[best]), float(values[best])


def bootstrap_metrics(y, s, threshold: float, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, tuple[float, float]]:
    """95% percentile CIs for each metric at a FIXED threshold (resampling test rows)."""
    y, s = np.asarray(y), np.asarray(s)
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    draws = {m: [] for m in METRICS}
    for i in idx:
        r = confusion_metrics(y[i], s[i] >= threshold)
        for m in METRICS:
            draws[m].append(r[m])
    return {m: tuple(float(v) for v in np.percentile(draws[m], [2.5, 97.5])) for m in METRICS}


def capacity_table(y: np.ndarray, raw: np.ndarray, fractions=CAPACITY_FRACTIONS) -> list[dict]:
    """Review the top-k% of orders by raw score: precision, recall, lift (= precision / prevalence).

    Order is by descending raw score with a stable sort, so ties (if any) resolve by row order.
    """
    y = np.asarray(y)
    order = np.argsort(-np.asarray(raw), kind="stable")
    prevalence, total_pos = float(y.mean()), float(y.sum())
    rows = []
    for k in fractions:
        n = max(1, int(np.ceil(len(y) * k)))
        hit = float(y[order[:n]].sum())
        precision = hit / n
        rows.append({"fraction": k, "n_reviewed": n, "precision": precision,
                     "recall": hit / total_pos if total_pos else 0.0,
                     "lift": precision / prevalence if prevalence else 0.0})
    return rows


# ===========================================================================
# Report
# ===========================================================================

def _fmt(v: float, ci: tuple[float, float]) -> str:
    return f"{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def build_report(split_info, op_rows, cap_rows, prevalence) -> str:
    L = ["# Operating points", "",
         "<!-- GENERATED by src/13_operating_points.py - do not edit by hand. -->", "",
         "Shipped model (`models/final_model.joblib`) with its isotonic calibrator. Thresholds act on the "
         "CALIBRATED score (flag when score >= threshold) and are chosen on VALIDATION only; every figure below "
         f"is on the TEST split with 95% percentile bootstrap CIs ({N_BOOT} resamples of test rows, threshold held fixed). "
         "Target is the proxy `Late_delivery_risk`, not RTO.", "",
         "| Split | Orders | Date range | Positive rate |", "|---|---|---|---|"]
    for name, n, lo, hi, pos in split_info:
        L.append(f"| {name} | {n:,} | {lo} -> {hi} | {pos:.2%} |")

    L += ["", "## Thresholds chosen on validation, evaluated on test", "",
          "| Objective | Threshold | Validation value | Accuracy | Balanced acc. | MCC | Precision | Recall | Specificity | Flagged share |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in op_rows:
        cells = " | ".join(_fmt(r["test"][m], r["ci"][m]) for m in METRICS)
        L.append(f"| {r['name']} | {r['threshold']:.4f} | {r['val_value']:.4f} | {cells} |")
    L += ["", "Candidate thresholds are the distinct validation calibrated scores (the calibrator is a step function); "
          "ties on the objective resolve to the lowest threshold. Different objectives can select the same threshold.", ""]

    L += ["## Review capacity (top-k% by raw score, TEST)", "",
          f"Test prevalence is {prevalence:.2%}, so lift = precision / {prevalence:.4f}. "
          "Recall is the share of all late orders that fall inside the reviewed slice.", "",
          "| Review share | Orders reviewed | Precision | Recall | Lift |", "|---|---|---|---|---|"]
    for r in cap_rows:
        L.append(f"| {r['fraction']:.0%} | {r['n_reviewed']:,} | {r['precision']:.4f} | {r['recall']:.4f} | {r['lift']:.3f}x |")
    L += ["", "Because the label is near rule-based (see `MODEL_CARD.md`), the top slice is dominated by First Class orders; "
          "high top-k precision reflects that structure rather than courier insight.", ""]
    return "\n".join(L) + "\n"


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70 + "\n  RTO Shield AI - Operating points\n" + "=" * 70)
    train_df, val_df, test_df = load_splits()
    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    cal = joblib.load(MODELS_DIR / "final_calibrator.joblib")

    _, cal_val = score(val_df, prep, model, cal)
    raw_te, cal_te = score(test_df, prep, model, cal)
    y_val, y_te = val_df[TARGET_COL].to_numpy(), test_df[TARGET_COL].to_numpy()

    op_rows = []
    for name, metric in OBJECTIVES.items():
        thr, val_value = choose_threshold(y_val, cal_val, metric)
        test = confusion_metrics(y_te, cal_te >= thr)
        ci = bootstrap_metrics(y_te, cal_te, thr)
        op_rows.append({"name": name, "threshold": thr, "val_value": val_value, "test": test, "ci": ci})
        print(f"  {name:<28} thr={thr:.4f}  val {metric}={val_value:.4f}  "
              f"test acc={test['accuracy']:.4f} bacc={test['balanced_accuracy']:.4f} "
              f"mcc={test['mcc']:.4f} flagged={test['flagged_share']:.2%}")

    cap_rows = capacity_table(y_te, raw_te)
    for r in cap_rows:
        print(f"  top {r['fraction']:.0%}: precision={r['precision']:.4f} recall={r['recall']:.4f} lift={r['lift']:.3f}x")

    split_info = [(n, len(d), d["order_date"].min().date(), d["order_date"].max().date(), d[TARGET_COL].mean())
                  for n, d in (("Train", train_df), ("Validation", val_df), ("Test", test_df))]
    REPORT_PATH.write_text(build_report(split_info, op_rows, cap_rows, float(y_te.mean())), encoding="utf-8")
    print(f"\n  Report written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
