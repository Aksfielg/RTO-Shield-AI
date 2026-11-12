"""
tests/test_metrics_reporting.py
===============================
Guards the metric-reporting contract:

  * compute_metrics returns accuracy / balanced accuracy / specificity / MCC and
    takes ranking metrics from y_rank (raw scores) while decisions use y_prob
  * evaluate_batch warns on small labelled batches, attaches bootstrap CIs and
    suppresses tier calibration gaps for tiers with n < 30
  * per-mode table: no 0.5 baseline, CI + prevalence + PR-AUC lift, informative/limited rule
  * ablations: ROC-AUC CI + own validation threshold + explicit history verdict
  * backtest: PR-AUC lift per fold
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import predict  # noqa: E402
from config import OUTPUTS_DIR  # noqa: E402
from utils import compute_metrics  # noqa: E402


@pytest.fixture(scope="module")
def fm():
    p = OUTPUTS_DIR / "final_metrics.json"
    if not p.exists():
        pytest.skip("run src/10_finalize.py first")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ compute_metrics
def test_compute_metrics_has_new_keys_and_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.6, 0.1, 0.7])
    m = compute_metrics(y, p, threshold=0.5)
    for k in ["accuracy", "balanced_accuracy", "specificity", "mcc"]:
        assert k in m
    pred = (p >= 0.5).astype(int)
    assert m["accuracy"] == round((pred == y).mean(), 4)
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    assert m["specificity"] == round(tn / (tn + fp), 4)
    assert m["mcc"] == round(matthews_corrcoef(y, pred), 4)
    assert m["balanced_accuracy"] == round((m["recall"] + m["specificity"]) / 2, 4)


def test_ranking_uses_y_rank_decisions_use_y_prob():
    y = np.array([0, 0, 1, 1, 0, 1])
    cal = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])      # fully tied (like a collapsed calibrator)
    raw = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    m = compute_metrics(y, cal, threshold=0.5, y_rank=raw)
    assert m["roc_auc"] == round(roc_auc_score(y, raw), 4) == 1.0
    assert compute_metrics(y, cal, threshold=0.5)["roc_auc"] == 0.5   # ties destroy ranking
    assert m["recall"] == 1.0 and m["specificity"] == 0.0              # decisions from cal scores


# ------------------------------------------------------------------ evaluate_batch
def _batch(n, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    raw = np.clip(0.5 * y + rng.normal(0.25, 0.2, n), 0, 1)
    tiers = rng.choice(["Low", "Medium", "High"], n, p=[0.6, 0.38, 0.02])
    return y, raw, raw.copy(), tiers


def test_small_batch_warns_with_cis_and_suppresses_small_tiers():
    y, raw, cal, tiers = _batch(120)
    ev = predict.evaluate_batch(y, raw, cal, tiers, n_boot=200)
    assert ev["small_sample"] and "Small sample" in ev["warning"]
    assert ev["roc_auc_ci"][0] is not None and ev["roc_auc_ci"][0] <= ev["roc_auc"] <= ev["roc_auc_ci"][1]
    for row in ev["tier_calibration"]:
        if row["n"] < predict.MIN_TIER_N:
            assert row["suppressed"] and row["gap"] is None
        else:
            assert not row["suppressed"] and row["gap"] is not None
    assert any(r["suppressed"] for r in ev["tier_calibration"])   # the ~2% High tier has n < 30


def test_large_batch_has_no_small_sample_warning():
    y, raw, cal, tiers = _batch(400, seed=1)
    ev = predict.evaluate_batch(y, raw, cal, tiers, n_boot=100)
    assert not ev["small_sample"] and ev["warning"] is None


def test_single_class_batch_does_not_crash():
    ev = predict.evaluate_batch(np.ones(50, int), np.linspace(0, 1, 50), np.linspace(0, 1, 50),
                                np.array(["Low"] * 50), n_boot=50)
    assert ev["roc_auc"] is None and ev["small_sample"]


# ------------------------------------------------------------------ per mode
def test_per_mode_table_structure_and_flag_rule(fm):
    assert {r["mode"] for r in fm["per_mode"]} == {"Same Day", "First Class", "Second Class", "Standard Class"}
    for r in fm["per_mode"]:
        assert "baseline_roc_auc" not in r and "delta_roc_auc" not in r      # 0.5 baseline removed
        for k in ["orders", "prevalence", "roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi", "pr_auc", "pr_auc_lift", "flag"]:
            assert k in r
        assert r["roc_auc_ci_lo"] <= r["roc_auc"] <= r["roc_auc_ci_hi"]
        assert abs(r["pr_auc_lift"] - r["pr_auc"] / r["prevalence"]) < 1e-9
        expected = "limited" if (r["roc_auc_ci_lo"] < 0.55 or r["roc_auc_ci_lo"] <= 0.5) else "informative"
        assert r["flag"] == expected
    std = next(r for r in fm["per_mode"] if r["mode"] == "Standard Class")
    assert std["flag"] == "limited"


# ------------------------------------------------------------------ ablation
def test_ablation_has_ci_own_threshold_and_verdict(fm):
    labels = [r["label"] for r in fm["ablation"]]
    assert "No history features" in labels
    for r in fm["ablation"]:
        for k in ["roc_auc", "roc_auc_ci_lo", "roc_auc_ci_hi", "pr_auc", "brier", "threshold", "f1", "precision",
                  "recall", "accuracy", "balanced_accuracy", "specificity", "mcc"]:
            assert k in r, k
        assert r["roc_auc_ci_lo"] <= r["roc_auc"] <= r["roc_auc_ci_hi"]
    thresholds = {r["threshold"] for r in fm["ablation"]}
    assert len(thresholds) > 1, "each ablated model must have its own validation-chosen threshold"
    hist = next(r for r in fm["ablation"] if r["label"] == "No history features")
    verdict = fm["ablation_summary"]["history_verdict"]
    if hist["delta_ci_lo"] <= 0 <= hist["delta_ci_hi"] or hist["delta_ci_hi"] < 0:
        assert "NO significant gain" in verdict
    else:
        assert "statistically significant" in verdict


# ------------------------------------------------------------------ backtest
def test_backtest_reports_pr_auc_lift(fm):
    rows = fm["backtest"]["rows"]
    assert len(rows) >= 3
    for r in rows:
        assert abs(r["pr_auc_lift"] - r["pr_auc"] / r["prevalence"]) < 1e-3
    assert "pr_auc_lift_mean" in fm["backtest"]


def test_final_metrics_note_raw_ranking(fm):
    assert "raw" in fm["test"]["ranking_basis"].lower()
    for k in ["accuracy_at_max_f1", "balanced_accuracy_at_max_f1", "specificity_at_max_f1", "mcc_at_max_f1"]:
        assert k in fm["test"]
    for p in fm["policies"]:
        for k in ["accuracy", "balanced_accuracy", "specificity", "mcc"]:
            assert k in p
