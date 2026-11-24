"""
src/rigour.py
=============
Statistical-rigour helpers used by 10_finalize.py (which stores every number in outputs/final_metrics.json), the tests,
and nothing else: the app and the docs only READ the JSON.  Pure functions, no I/O, every random step takes a seed.

  * paired_auc_bootstrap      - ROC-AUC difference of two score vectors, SAME resample indices, 95% percentile CI
  * rule_flags                - the 3-line plain-rule baseline (flag First/Second Class + Same Day from 12:00)
  * topk_flags                - flag the k highest raw scores (equal-flagged-share comparison)
  * paired_flag_bootstrap     - paired bootstrap of the precision / recall difference between two fixed flag sets
  * equal_frequency_bins      - 10 equal-frequency reliability bins with bootstrap 95% CI band
  * ece_equal_width / ece_equal_frequency, brier_skill_score
  * wilson_interval           - Wilson 95% interval for a proportion
  * per_mode_calibration      - mean predicted vs observed late rate and ECE inside each shipping mode
  * significance_summary / rule_verdict - the sentences the app and docs display (built from the numbers)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

N_BOOT = 1000
Z95 = 1.959963984540054
RULE_DEFINITION = ("Flag First Class and Second Class orders, plus Same Day orders placed at or after 12:00 "
                   "(order hour >= 12); flag nothing else.")
RULE_MODES = ("First Class", "Second Class")
RULE_SAME_DAY_FROM_HOUR = 12
CAL_GAP_FLAG_PP = 5.0
MODEL_LABELS = {"baseline_lr": "Baseline LR", "logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
                "xgboost": "XGBoost", "lightgbm": "LightGBM (default)"}


# --------------------------------------------------------------------------- AUC and paired bootstrap
def fast_auc(y, s) -> float:
    y = np.asarray(y)
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_auc_bootstrap(y, scores_a, scores_b, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) with a percentile 95% CI; both models are scored on the SAME resample indices."""
    y = np.asarray(y)
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        deltas[i] = fast_auc(yb, a[idx]) - fast_auc(yb, b[idx])
    point = fast_auc(y, a) - fast_auc(y, b)
    lo, hi = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
    return {"delta": float(point), "ci_lo": lo, "ci_hi": hi, "distinguishable": bool(lo > 0 or hi < 0), "n_boot": n_boot, "seed": seed}


# --------------------------------------------------------------------------- plain-rule baseline
def rule_flags(shipping_mode, order_hour) -> np.ndarray:
    """Boolean flags of the 3-line rule (First/Second Class, or Same Day from 12:00)."""
    mode = np.asarray(shipping_mode)
    hour = np.asarray(order_hour, dtype=float)
    return np.isin(mode, RULE_MODES) | ((mode == "Same Day") & (hour >= RULE_SAME_DAY_FROM_HOUR))


def topk_flags(scores, k: int) -> np.ndarray:
    """Flag the k highest scores (stable order for ties)."""
    f = np.zeros(len(scores), dtype=bool)
    f[np.argsort(-np.asarray(scores), kind="stable")[: int(k)]] = True
    return f


def precision_recall(y, flag) -> tuple[float, float]:
    y, flag = np.asarray(y), np.asarray(flag, dtype=bool)
    tp = int((flag & (y == 1)).sum())
    return (tp / flag.sum() if flag.sum() else 0.0), (tp / (y == 1).sum() if (y == 1).sum() else 0.0)


def paired_flag_bootstrap(y, flag_a, flag_b, n_boot: int = N_BOOT, seed: int = 1) -> dict:
    """Precision / recall difference (a - b) of two FIXED flag sets, bootstrapping orders with shared indices."""
    y = np.asarray(y)
    fa, fb = np.asarray(flag_a, dtype=bool), np.asarray(flag_b, dtype=bool)
    rng = np.random.default_rng(seed)
    n = len(y)
    dp, dr = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        pa, ra = precision_recall(y[idx], fa[idx])
        pb, rb = precision_recall(y[idx], fb[idx])
        dp[i], dr[i] = pa - pb, ra - rb
    pa, ra = precision_recall(y, fa)
    pb, rb = precision_recall(y, fb)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"precision_a": pa, "precision_b": pb, "recall_a": ra, "recall_b": rb,
            "delta_precision": float(pa - pb), "delta_precision_ci": ci(dp),
            "delta_recall": float(ra - rb), "delta_recall_ci": ci(dr), "n_boot": n_boot, "seed": seed}


# --------------------------------------------------------------------------- calibration
def ece_equal_width(y, p, n_bins: int = 10) -> float:
    """Equal-width ECE: the project's own utils.expected_calibration_error, so it matches final_metrics.json's headline ECE."""
    from utils import expected_calibration_error
    return float(expected_calibration_error(np.asarray(y, dtype=float), np.asarray(p, dtype=float), n_bins=n_bins))


def _rank_bins(score_for_order, n: int, n_bins: int) -> list[np.ndarray]:
    """Equal-frequency bins by rank (ties in the calibrated score are broken by the raw score, which is monotone with it)."""
    order = np.argsort(score_for_order, kind="stable")
    return [order[b] for b in np.array_split(np.arange(n), n_bins)]


def equal_frequency_bins(y, p, order_by=None, n_bins: int = 10, n_boot: int = N_BOOT, seed: int = 2) -> list[dict]:
    """
    n_bins equal-frequency reliability bins.  ``order_by`` (default p) ranks the orders; pass the RAW score when p is the
    isotonic-calibrated score so plateaus (many identical calibrated values) are split by the underlying raw rank.
    Each bin: n, mean predicted, observed rate and a bootstrap 95% CI of the observed rate (orders resampled within the bin).
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for k, idx in enumerate(_rank_bins(p if order_by is None else np.asarray(order_by), len(y), n_bins), start=1):
        n = len(idx)
        obs = float(y[idx].mean())
        draws = rng.binomial(n, obs, n_boot) / n            # == resampling the bin's 0/1 outcomes with replacement
        rows.append({"bin": k, "n": int(n), "mean_predicted": float(p[idx].mean()), "observed": obs,
                     "ci_lo": float(np.percentile(draws, 2.5)), "ci_hi": float(np.percentile(draws, 97.5))})
    return rows


def ece_equal_frequency(y, p, order_by=None, n_bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(sum(len(i) / len(y) * abs(y[i].mean() - p[i].mean())
                     for i in _rank_bins(p if order_by is None else np.asarray(order_by), len(y), n_bins)))


def brier_skill_score(y, p) -> float:
    """1 - Brier / (prevalence * (1 - prevalence)): skill over always predicting the base rate."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    prev = float(y.mean())
    return float(1.0 - np.mean((p - y) ** 2) / (prev * (1.0 - prev)))


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def per_mode_calibration(modes, y, cal, order=("Same Day", "First Class", "Second Class", "Standard Class")) -> list[dict]:
    modes, y, cal = np.asarray(modes), np.asarray(y, dtype=float), np.asarray(cal, dtype=float)
    rows = []
    for mode in order:
        m = modes == mode
        if not m.any():
            continue
        gap = float(cal[m].mean() - y[m].mean())
        rows.append({"mode": mode, "orders": int(m.sum()), "mean_predicted": float(cal[m].mean()), "observed": float(y[m].mean()),
                     "gap_pp": gap * 100.0, "ece": ece_equal_width(y[m], cal[m]),
                     "flagged": bool(abs(gap * 100.0) > CAL_GAP_FLAG_PP)})
    return rows


# --------------------------------------------------------------------------- sentences (built from the numbers)
def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def significance_summary(rows: list[dict]) -> str:
    """rows: dicts with label, delta (candidate - final), distinguishable."""
    same = [r["label"] for r in rows if not r["distinguishable"]]
    worse = [r["label"] for r in rows if r["distinguishable"] and r["delta"] < 0]
    better = [r["label"] for r in rows if r["distinguishable"] and r["delta"] > 0]
    parts = []
    if same:
        parts.append(f"{_join(same)} {'is' if len(same) == 1 else 'are'} not statistically distinguishable from the tuned model "
                     "(the paired-bootstrap 95% CI of the ROC-AUC difference includes 0)")
    if worse:
        parts.append(f"{_join(worse)} {'is' if len(worse) == 1 else 'are'} significantly worse")
    if better:
        parts.append(f"{_join(better)} {'is' if len(better) == 1 else 'are'} significantly better")
    text = "; ".join(parts) + "."
    if same:
        text += " LightGBM was chosen for ranking and calibration, not because it wins significantly."
    return text[0].upper() + text[1:]


def rule_verdict(matches: bool, delta_precision: float = 0.0) -> str:
    if matches:
        return ("The plain rule matches the model at the same flagged share (the precision difference is not statistically "
                "distinguishable from 0). The model's added value is calibrated probabilities, three tiers, SHAP reasons and the "
                "information-level warning, not higher accuracy than a hand-written rule.")
    word = "higher" if delta_precision > 0 else "lower"
    return (f"At the same flagged share the model's precision is significantly {word} than the plain rule's (the paired-bootstrap "
            "95% CI of the difference excludes 0).")
