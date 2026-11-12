"""
tests/test_consistency.py
========================
Guards the decisioning + number-consistency contract:

  * tier / threshold cutoffs sit in low-density score gaps (validation AND test)
  * flagged share is stable for threshold shifts of +/-0.01 and +/-0.02
  * flagged == High tier everywhere
  * README / MODEL_CARD / report_tables show the numbers in final_metrics.json
  * app.py / predict.py contain no hand-typed copies of the cutoffs

Run:  pytest tests/test_consistency.py -v
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tiering  # noqa: E402
from config import MODELS_DIR, OUTPUTS_DIR  # noqa: E402

FM_PATH = OUTPUTS_DIR / "final_metrics.json"
THR_PATH = MODELS_DIR / "thresholds.json"


@pytest.fixture(scope="module")
def fm():
    if not FM_PATH.exists():
        pytest.skip("run src/10_finalize.py first")
    return json.loads(FM_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def thr():
    return json.loads(THR_PATH.read_text(encoding="utf-8"))


def test_cutoffs_sit_in_gaps(thr):
    checks = thr["cutoff_selection"]["checks"]
    assert set(checks) >= {"tier low_max", "tier high_min", "precision_80"}
    for name, c in checks.items():
        for split in ("validation", "test"):
            r = c[split]
            assert r["in_gap"], f"{name} falls inside a score cluster on {split}"
            assert r["big_clusters_split"] == 0, f"{name} splits a >2% cluster on {split}"
            assert r["share_within_0.02"] <= tiering.MAX_WINDOW_MASS, f"{name} not in a low-density gap on {split}"


def test_stability_within_limit(fm, thr):
    limit = tiering.MAX_FLAG_SWING_PP
    for name, swing in thr["stability_swing_pp"].items():
        assert swing <= limit, f"{name} flagged share moves {swing:.2f} pp (> {limit})"
    assert fm["stability"]["passed"]
    assert {r["shift"] for r in fm["stability"]["rows"]} == set(tiering.SHIFTS)


def test_tier_thresholds_agree_between_files(fm, thr):
    assert fm["tiers"]["cutoffs"] == thr["tier_cutoffs"] or (
        fm["tiers"]["cutoffs"]["low_max"] == thr["tier_cutoffs"]["low_max"]
        and fm["tiers"]["cutoffs"]["high_min"] == thr["tier_cutoffs"]["high_min"])
    assert thr["thresholds"]["flag_high_tier"] == thr["tier_cutoffs"]["high_min"]
    high_policy = next(p for p in fm["policies"] if p["policy"].startswith("High tier"))
    assert high_policy["threshold"] == thr["tier_cutoffs"]["high_min"]
    # flagged share of the High policy equals the High tier share
    assert abs(high_policy["flagged_share"] - fm["tiers"]["test"]["High"]["share"]) < 1e-9


def test_tier_descriptions_use_relative_language(thr):
    for tier, d in thr["tier_descriptions"].items():
        text = d["description"].lower() + " " + d["action"].lower()
        assert "zero friction" not in text
        assert "% of these orders were late" in d["description"]


def test_tiers_cover_all_test_orders(fm):
    assert sum(t["orders"] for t in fm["tiers"]["test"].values()) == fm["test"]["n"]
    for t in fm["tiers"]["test"].values():
        assert t["orders"] > 0


def test_flag_equals_high_tier_in_predict():
    import pandas as pd
    import predict
    orders = pd.read_parquet(ROOT / "data" / "processed" / "orders.parquet").sort_values("order_date").reset_index(drop=True)
    sample = orders.iloc[-300:]
    res = predict.score_batch(sample)
    assert ((res["tier"] == "High") == res["flagged"]).all()
    assert res["tier_description"].str.len().gt(0).all()


@pytest.mark.parametrize("doc", ["README.md", "MODEL_CARD.md"])
def test_docs_show_json_numbers(doc, fm, thr):
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert f"{fm['test']['roc_auc']:.4f}" in text
    assert f"{thr['tier_cutoffs']['high_min']:.4f}" in text
    assert f"{fm['test']['ece_calibrated']:.4f}" in text
    assert "{{" not in text, "unrendered placeholder left in " + doc


def test_report_tables_generated_from_json(fm):
    text = (OUTPUTS_DIR / "report_tables.md").read_text(encoding="utf-8")
    assert f"{fm['test']['pr_auc']:.4f}" in text
    assert "11_render_docs.py" in text


def test_no_hand_typed_cutoffs_in_code(thr):
    """The cutoffs may only appear in JSON, never as literals in app/predict."""
    cutoffs = [thr["tier_cutoffs"]["low_max"], thr["tier_cutoffs"]["high_min"], thr["thresholds"]["precision_80"],
               thr["thresholds"]["max_f1"]]
    files = ["app.py", "src/predict.py"] + [str(p.relative_to(ROOT)) for p in (ROOT / "ui").rglob("*.py")]
    for fname in files:
        text = (ROOT / fname).read_text(encoding="utf-8")
        nums = set(re.findall(r"(?<![\d.])0\.\d{3,4}(?!\d)", text))
        for c in cutoffs:
            assert f"{c:.4f}" not in nums and f"{c:.3f}" not in nums, f"{fname} hard-codes cutoff {c}"
