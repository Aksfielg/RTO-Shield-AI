"""Reproducibility: split sizes/dates equal the metrics file, scoring is deterministic, calibrated scores are monotone in raw scores."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import predict  # noqa: E402
from config import TRAIN_FRAC, VAL_FRAC  # noqa: E402

FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def orders():
    return pd.read_parquet(ROOT / "data" / "processed" / "orders.parquet").sort_values("order_date").reset_index(drop=True)


def _splits(o):
    n = len(o)
    a, b = int(np.floor(n * TRAIN_FRAC)), int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    return {"train": o.iloc[:a], "val": o.iloc[a:b], "test": o.iloc[b:]}


def test_split_sizes_and_dates_equal_the_metrics_file(orders):
    for name, df in _splits(orders).items():
        rec = FM["splits"][name]
        assert rec["orders"] == len(df), name
        assert rec["date_min"] == str(df["order_date"].min().date()), name
        assert rec["date_max"] == str(df["order_date"].max().date()), name
    sp = FM["splits"]
    assert sp["train"]["date_max"] <= sp["val"]["date_min"] <= sp["val"]["date_max"] <= sp["test"]["date_min"]      # chronological
    assert (sp["train"]["orders"], sp["val"]["orders"], sp["test"]["orders"]) == (46026, 9863, 9863)
    assert sum(s["orders"] for s in sp.values()) == FM["dataset"]["n_orders"]


def test_scoring_the_same_200_test_orders_twice_gives_identical_scores(orders):
    sample = _splits(orders)["test"].sample(200, random_state=11)
    a, b = predict.score_batch(sample), predict.score_batch(sample)
    for col in ("risk_score", "raw_score"):
        assert np.array_equal(a[col].values, b[col].values), f"{col} differs between two identical runs"
    assert (a["tier"].values == b["tier"].values).all() and (a["flagged"].values == b["flagged"].values).all()
    assert list(a["reason_1"]) == list(b["reason_1"])
    one = predict.score_order(sample.iloc[0].to_dict())                           # the single-order path agrees with the batch path
    assert one["risk_score"] == pytest.approx(a["risk_score"].iloc[0], abs=1e-6)


def test_calibrated_scores_are_monotone_in_the_raw_scores(orders):
    sample = _splits(orders)["test"].sample(200, random_state=5)
    res = predict.score_batch(sample).sort_values("raw_score", kind="stable")
    assert (np.diff(res["risk_score"].values) >= -1e-9).all(), "calibrated score decreases while the raw score increases"
    cal = joblib.load(ROOT / "models" / "final_calibrator.joblib")
    grid = np.linspace(0, 1, 2001)
    assert (np.diff(cal.predict(grid)) >= -1e-12).all()                           # isotonic map itself is non-decreasing
