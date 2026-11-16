"""
tests/test_operating_points.py
==============================
Fast checks for src/13_operating_points.py.

Run:
    pytest tests/test_operating_points.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path[:0] = [str(ROOT), str(SRC)]


@pytest.fixture(scope="module")
def op():
    spec = importlib.util.spec_from_file_location("op13", SRC / "13_operating_points.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data(seed=0, n=800):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.round(np.clip(0.5 * y + rng.normal(0.25, 0.25, n), 0, 1), 2)   # stepped, tied scores
    return y, s


def test_confusion_metrics_match_sklearn(op):
    y, s = _data()
    flagged = s >= 0.5
    m = op.confusion_metrics(y, flagged)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, flagged))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, flagged))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, flagged))
    assert m["flagged_share"] == pytest.approx(flagged.mean())
    assert m["recall"] == pytest.approx(flagged[y == 1].mean())
    assert m["specificity"] == pytest.approx((~flagged[y == 0]).mean())
    assert m["precision"] == pytest.approx(y[flagged].mean())


def test_degenerate_flags_do_not_divide_by_zero(op):
    y = np.array([1, 0, 1, 0])
    none, everything = op.confusion_metrics(y, np.zeros(4, bool)), op.confusion_metrics(y, np.ones(4, bool))
    assert none["precision"] == 0.0 and none["mcc"] == 0.0 and none["flagged_share"] == 0.0
    assert everything["recall"] == 1.0 and everything["specificity"] == 0.0 and everything["mcc"] == 0.0


@pytest.mark.parametrize("metric", ["balanced_accuracy", "mcc", "accuracy"])
def test_choose_threshold_is_optimal_over_candidates(op, metric):
    y, s = _data(1)
    thr, val = op.choose_threshold(y, s, metric)
    best = max(op.confusion_metrics(y, s >= t)[metric] for t in np.unique(s))
    assert val == pytest.approx(best)
    assert op.confusion_metrics(y, s >= thr)[metric] == pytest.approx(best)
    assert thr in set(np.unique(s))


def test_threshold_depends_only_on_validation_data(op):
    """Choosing on validation must not see test data: changing test labels cannot move the threshold."""
    y_val, s_val = _data(2)
    thr_before = op.choose_threshold(y_val, s_val, "mcc")[0]
    y_test, s_test = _data(3)
    _ = op.confusion_metrics(1 - y_test, s_test >= thr_before)     # "using" a different test set
    assert op.choose_threshold(y_val, s_val, "mcc")[0] == thr_before


def test_threshold_ties_resolve_to_lowest(op):
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])           # any cut between 0.2 and 0.8 is perfect
    assert op.choose_threshold(y, s, "accuracy")[0] == pytest.approx(0.8)
    # candidates are scores, so the lowest perfect candidate is 0.8 (0.2 would flag a negative)


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible(op):
    y, s = _data(4, 600)
    thr = 0.5
    point = op.confusion_metrics(y, s >= thr)
    ci = op.bootstrap_metrics(y, s, thr, n_boot=200)
    for m in op.METRICS:
        assert ci[m][0] <= point[m] <= ci[m][1]
    assert ci == op.bootstrap_metrics(y, s, thr, n_boot=200)


def test_capacity_table_known_values(op):
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 1])      # prevalence 0.4
    raw = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
    rows = {round(r["fraction"], 2): r for r in op.capacity_table(y, raw, (0.2, 0.4, 1.0))}
    assert rows[0.2]["n_reviewed"] == 2 and rows[0.2]["precision"] == 1.0
    assert rows[0.2]["recall"] == pytest.approx(0.5) and rows[0.2]["lift"] == pytest.approx(2.5)
    assert rows[0.4]["precision"] == pytest.approx(0.75) and rows[0.4]["recall"] == pytest.approx(0.75)
    assert rows[1.0]["recall"] == 1.0 and rows[1.0]["lift"] == pytest.approx(1.0)


def test_capacity_recall_is_monotone_and_row_order_invariant(op):
    y, s = _data(5)
    rows = op.capacity_table(y, s)
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls)
    perm = np.random.default_rng(0).permutation(len(y))
    rows_p = op.capacity_table(y[perm], s[perm])
    assert [r["n_reviewed"] for r in rows_p] == [r["n_reviewed"] for r in rows]


def test_shipped_artefacts_reproduce_card_auc_and_report(op):
    """Scoring path sanity: raw TEST AUC must match MODEL_CARD (0.7716) and the report must exist."""
    from sklearn.metrics import roc_auc_score
    import joblib
    train_df, val_df, test_df = op.load_splits()
    prep = joblib.load(ROOT / "models" / "preprocessor.joblib")
    model = joblib.load(ROOT / "models" / "final_model.joblib")
    cal = joblib.load(ROOT / "models" / "final_calibrator.joblib")
    raw, cal_s = op.score(test_df, prep, model, cal)
    assert roc_auc_score(test_df["Late_delivery_risk"], raw) == pytest.approx(0.7716, abs=5e-4)
    assert ((cal_s >= 0) & (cal_s <= 1)).all()

    report = ROOT / "outputs" / "operating_points.md"
    if not report.exists():
        pytest.skip("run src/13_operating_points.py first")
    text = report.read_text(encoding="utf-8")
    for token in ("(a) Max balanced accuracy", "(b) Max MCC", "(c) Max accuracy", "Review capacity", "| 10% |", "| 40% |"):
        assert token in text
