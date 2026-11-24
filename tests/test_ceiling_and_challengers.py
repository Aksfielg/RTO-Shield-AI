"""
tests/test_ceiling_and_challengers.py
=====================================
Fast checks for src/12_ceiling_and_challengers.py (no model training).

Run:
    pytest tests/test_ceiling_and_challengers.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path[:0] = [str(ROOT), str(SRC)]


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SRC / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cc():
    return _load("cc12", "12_ceiling_and_challengers.py")


def _toy() -> pd.DataFrame:
    return pd.DataFrame({
        "Shipping_Mode": ["Same Day", "Same Day", "Same Day", "Standard Class", "Standard Class", "First Class"],
        "order_hour": [3, 11, 12, 23, 12, 12],
        "Customer_Id": [7, 8, 14, 15, 1, 6],
        "Type": ["DEBIT", "DEBIT", "DEBIT", "CASH", "CASH", "PAYMENT"],
        "Late_delivery_risk": [0, 0, 1, 1, 0, 1],
    })


def test_same_day_pm_definition(cc):
    out = cc.add_derived_columns(_toy())
    assert out["same_day_pm"].tolist() == [False, False, True, False, False, False]
    # hour 12 is the first "pm" hour; non-Same-Day modes are never same_day_pm
    assert not out.loc[out["Shipping_Mode"] != "Same Day", "same_day_pm"].any()


def test_hash_control_is_customer_id_mod_7(cc):
    out = cc.add_derived_columns(_toy())
    assert out[cc.HASH_COL].tolist() == [0, 1, 0, 1, 1, 6]


def test_lookup_group_means_and_unseen_fallback(cc):
    train = cc.add_derived_columns(_toy())
    lk = cc.GroupMeanLookup(["Shipping_Mode"]).fit(train)
    test = pd.DataFrame({"Shipping_Mode": ["Same Day", "Standard Class", "Second Class"]})
    pred = lk.predict(test)
    assert pred[0] == pytest.approx(1 / 3)                 # Same Day: 0,0,1
    assert pred[1] == pytest.approx(0.5)                   # Standard: 1,0
    assert pred[2] == pytest.approx(train["Late_delivery_risk"].mean())  # unseen -> global mean
    assert lk.unseen_share(test) == pytest.approx(1 / 3)


def test_lookup_preserves_row_order_with_multi_column_key(cc):
    train = cc.add_derived_columns(_toy())
    lk = cc.GroupMeanLookup(["Shipping_Mode", "same_day_pm"]).fit(train)
    test = train.iloc[::-1].reset_index(drop=True)
    pred = lk.predict(test)
    expected = test.apply(
        lambda r: train[(train.Shipping_Mode == r.Shipping_Mode) & (train.same_day_pm == r.same_day_pm)]
        ["Late_delivery_risk"].mean(), axis=1).to_numpy()
    np.testing.assert_allclose(pred, expected)


def test_auc_ci_brackets_point_estimate(cc):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    s = y * 0.8 + rng.normal(size=600)
    auc, lo, hi = cc.auc_with_ci(y, s, n_boot=200)
    assert lo <= auc <= hi
    assert 0.5 < auc < 1.0


def test_paired_diff_of_identical_scores_is_zero(cc):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = rng.normal(size=400) + y
    d = cc.paired_auc_diff(y, s, s, n_boot=100)
    assert d["diff"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_paired_diff_detects_better_scorer(cc):
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    good = y * 1.5 + rng.normal(size=2000)
    noise = rng.normal(size=2000)
    d = cc.paired_auc_diff(y, good, noise, n_boot=200)
    assert d["lo"] > 0 and d["diff"] > 0.2


def test_expanding_windows_are_chronological_and_disjoint(cc):
    folds = cc.expanding_window_folds(1000, 4)
    assert len(folds) == 4
    prev_end = 0
    for tr_end, te_start, te_end in folds:
        assert tr_end == te_start and te_end > te_start
        # expanding: this fold's training window ends where the previous test block ended
        assert tr_end >= prev_end
        if prev_end:
            assert tr_end == prev_end
        prev_end = te_end
    assert folds[0][0] > 0 and folds[-1][2] == 1000
    # training window grows every fold
    assert [f[0] for f in folds] == sorted({f[0] for f in folds})


def test_split_matches_02_train(cc):
    """Test split must equal the one 02_train.py / 05_tune_multiseed.py use."""
    orders = pd.read_parquet(ROOT / "data" / "processed" / "orders.parquet").sort_values("order_date").reset_index(drop=True)
    n = len(orders)
    tr_end, val_end = int(np.floor(n * 0.70)), int(np.floor(n * 0.85))
    train_df, val_df, test_df = cc.load_splits()
    assert (len(train_df), len(val_df), len(test_df)) == (tr_end, val_end - tr_end, n - val_end)
    assert train_df["order_date"].max() <= val_df["order_date"].min()
    assert val_df["order_date"].max() <= test_df["order_date"].min()
    assert list(test_df["Order Id"]) == list(orders.iloc[val_end:]["Order Id"])
    assert {"same_day_pm", cc.HASH_COL} <= set(test_df.columns)


def test_report_written_with_all_parts():
    report = ROOT / "outputs" / "ceiling_and_challengers.md"
    if not report.exists():
        pytest.skip("run src/12_ceiling_and_challengers.py first")
    text = report.read_text(encoding="utf-8")
    for heading in ("## Part A", "## Part B", "## Part C", "## Part D"):
        assert heading in text
    for label in ("(a) Shipping_Mode", "(d) (c) + order_hour", "Same Day", "Standard Class"):
        assert label in text
