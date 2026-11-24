"""Statistical-rigour additions: paired bootstrap, plain-rule baseline, equal-frequency calibration, Wilson tiers, rendered docs."""

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import rigour  # noqa: E402
from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL  # noqa: E402
from features import _select_feature_cols  # noqa: E402

FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))


def _split():
    o = pd.read_parquet(ROOT / "data" / "processed" / "orders.parquet").sort_values("order_date").reset_index(drop=True)
    n = len(o)
    a, b = int(np.floor(n * 0.70)), int(np.floor(n * 0.85))
    return o.iloc[:a], o.iloc[a:b], o.iloc[b:]


# ------------------------------------------------------------------ (a) paired bootstrap
def test_paired_bootstrap_is_reproducible_and_contains_the_observed_delta():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1500)
    a = y * 0.6 + rng.normal(0, 1, 1500)
    b = y * 0.4 + rng.normal(0, 1, 1500)
    r1 = rigour.paired_auc_bootstrap(y, a, b, n_boot=300, seed=11)
    r2 = rigour.paired_auc_bootstrap(y, a, b, n_boot=300, seed=11)
    assert r1 == r2                                                   # fixed seed -> identical output
    assert r1["ci_lo"] <= r1["delta"] <= r1["ci_hi"]
    assert r1["distinguishable"] is True and r1["delta"] > 0
    assert rigour.paired_auc_bootstrap(y, a, b, n_boot=300, seed=12)["ci_lo"] != r1["ci_lo"]
    same = rigour.paired_auc_bootstrap(y, a, a, n_boot=100, seed=1)    # identical models: delta exactly 0, never distinguishable
    assert same["delta"] == 0 and same["distinguishable"] is False


def test_stored_significance_rows_are_consistent_and_reproducible():
    rows = FM["model_comparison_significance"]["rows"]
    assert {r["model"] for r in rows} >= {"baseline_lr", "logistic_regression", "xgboost", "lightgbm"}
    for r in rows:
        assert r["ci_lo"] <= r["delta"] <= r["ci_hi"], r["model"]
        assert r["distinguishable"] == bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)
        assert r["n_boot"] == 1000
    assert FM["model_comparison_significance"]["summary"]
    # recompute one row from the saved models (shared resample indices, same seed)
    if not (ROOT / "models" / "baseline_lr.joblib").exists():
        pytest.skip("baseline model artefact missing")
    _, _, te = _split()
    y = te[TARGET_COL].values
    prep, final = joblib.load(ROOT / "models" / "preprocessor.joblib"), joblib.load(ROOT / "models" / "final_model.joblib")
    base, bprep = joblib.load(ROOT / "models" / "baseline_lr.joblib"), joblib.load(ROOT / "models" / "baseline_preprocessor.joblib")
    from config import BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES
    raw_final = final.predict_proba(prep.transform(_select_feature_cols(te, CATEGORICAL_FEATURES, NUMERIC_FEATURES)))[:, 1]
    raw_base = base.predict_proba(bprep.transform(_select_feature_cols(te, BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES)))[:, 1]
    again = rigour.paired_auc_bootstrap(y, raw_base, raw_final, seed=7)
    stored = next(r for r in rows if r["model"] == "baseline_lr")
    assert again["delta"] == pytest.approx(stored["delta"], abs=1e-9)
    assert again["ci_lo"] == pytest.approx(stored["ci_lo"], abs=1e-9) and again["ci_hi"] == pytest.approx(stored["ci_hi"], abs=1e-9)


# ------------------------------------------------------------------ (b) plain-rule baseline
def _brute_rule(mode: str, hour: int) -> bool:
    if mode == "First Class" or mode == "Second Class":
        return True
    return mode == "Same Day" and hour >= 12


def test_rule_flags_match_a_brute_force_recomputation_on_200_orders():
    _, _, te = _split()
    sample = te.sample(200, random_state=3)
    brute = np.array([_brute_rule(m, int(h)) for m, h in zip(sample["Shipping_Mode"], sample["order_hour"])])
    assert (rigour.rule_flags(sample["Shipping_Mode"].values, sample["order_hour"].values) == brute).all()
    assert brute.any() and not brute.all()


def test_stored_rule_baseline_equals_a_brute_force_count_on_the_whole_test_set():
    _, _, te = _split()
    rb = FM["rule_baseline"]
    y = te[TARGET_COL].values
    flag = np.array([_brute_rule(m, int(h)) for m, h in zip(te["Shipping_Mode"], te["order_hour"])])
    tp = int((flag & (y == 1)).sum())
    assert rb["rule"]["flagged_orders"] == int(flag.sum())
    assert rb["rule"]["precision"] == pytest.approx(tp / flag.sum())
    assert rb["rule"]["recall"] == pytest.approx(tp / (y == 1).sum())
    assert rb["model_top_k"]["k"] == rb["rule"]["flagged_orders"] == rb["model_top_k"]["flagged_orders"]
    lo, hi = rb["equal_share_comparison"]["delta_precision_ci"]
    assert lo <= rb["equal_share_comparison"]["delta_precision"] <= hi
    assert rb["matches_model"] == bool(lo <= 0 <= hi)
    # the rule was written from TRAIN evidence only
    ev = rb["train_evidence"]
    assert ev["orders"] == len(_split()[0]) and ev["late_rate_by_mode"]["First Class"]["orders"] > 0


def test_topk_flags_select_exactly_k_highest_scores():
    s = np.array([0.1, 0.9, 0.5, 0.9, 0.2])
    f = rigour.topk_flags(s, 2)
    assert f.sum() == 2 and f[1] and f[3]


# ------------------------------------------------------------------ (c) tiers + Wilson
def test_per_tier_orders_sum_to_the_test_size_and_wilson_intervals_are_valid():
    for split, n_expected in (("test", FM["test"]["n"]), ("validation", FM["splits"]["val"]["orders"])):
        tiers = FM["tiers"][split]
        assert sum(t["orders"] for t in tiers.values()) == n_expected
        for t in tiers.values():
            assert t["late_rate_ci_lo"] <= t["late_rate"] <= t["late_rate_ci_hi"]
            assert 0 <= t["late_rate_ci_lo"] < t["late_rate_ci_hi"] <= 1
    med = FM["tiers"]["test"]["Medium"]
    k, n, z = med["late_orders"], med["orders"], 1.959963984540054        # independent closed form of the Wilson interval
    ph = k / n
    half = z / (1 + z * z / n) * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    assert med["late_rate_ci_lo"] == pytest.approx(centre - half) and med["late_rate_ci_hi"] == pytest.approx(centre + half)
    wide = {k_: v["late_rate_ci_hi"] - v["late_rate_ci_lo"] for k_, v in FM["tiers"]["test"].items()}
    assert wide["Medium"] == max(wide.values())                          # the smallest tier has the widest interval


def test_equal_frequency_calibration_block():
    ef = FM["calibration_ef"]
    assert ef["n_bins"] == len(ef["calibrated"]) == len(ef["raw"]) == 10
    sizes = [b["n"] for b in ef["calibrated"]]
    assert sum(sizes) == FM["test"]["n"] and max(sizes) - min(sizes) <= 1
    for b in ef["calibrated"]:
        assert b["ci_lo"] <= b["observed"] <= b["ci_hi"]
    assert set(ef["ece"]) == {"raw_equal_width", "raw_equal_frequency", "calibrated_equal_width", "calibrated_equal_frequency"}
    assert ef["ece"]["calibrated_equal_width"] == pytest.approx(FM["test"]["ece_calibrated"])
    p = FM["test"]["base_rate"]
    assert ef["brier_skill_score"]["calibrated"] == pytest.approx(1 - FM["test"]["brier_calibrated"] / (p * (1 - p)))


def test_per_mode_calibration_flags_gaps_above_5pp():
    pmc = FM["per_mode_calibration"]
    assert sum(r["orders"] for r in pmc["rows"]) == FM["test"]["n"]
    for r in pmc["rows"]:
        assert r["gap_pp"] == pytest.approx((r["mean_predicted"] - r["observed"]) * 100)
        assert r["flagged"] == (abs(r["gap_pp"]) > 5.0)
    existing = {r["mode"]: r["calibration_error"] for r in FM["per_mode"]}
    for r in pmc["rows"]:
        assert r["gap_pp"] == pytest.approx(existing[r["mode"]] * 100)    # same number the Robustness table already showed


# ------------------------------------------------------------------ (d) rendered docs
def test_rendered_docs_contain_the_new_numbers_and_no_placeholders():
    sig = FM["model_comparison_significance"]
    rb, ef = FM["rule_baseline"], FM["calibration_ef"]
    r0 = sig["rows"][0]
    med = FM["tiers"]["test"]["Medium"]
    needles = [f"{r0['ci_lo']:+.4f}", f"{r0['ci_hi']:+.4f}", sig["summary"], f"{rb['rule']['precision']:.4f}",
               f"{ef['ece']['calibrated_equal_frequency']:.4f}", f"{ef['brier_skill_score']['calibrated']:.4f}",
               f"[{med['late_rate_ci_lo'] * 100:.1f}%, {med['late_rate_ci_hi'] * 100:.1f}%]", rb["verdict"]]
    for doc in ("README.md", "MODEL_CARD.md", "outputs/report_tables.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert "{{" not in text, f"unrendered placeholder in {doc}"
        for n in needles:
            assert n in text, f"{doc} is missing {n!r}"
    viva = (ROOT / "docs" / "VIVA_CHEATSHEET.md").read_text(encoding="utf-8")
    assert "{{" not in viva
    assert rb["verdict"] in viva and sig["summary"] in viva
    assert f"{rb['equal_share_comparison']['delta_precision']:+.4f}" in viva
    mc = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "Plain-rule baseline" in mc and "Calibration within shipping mode" in mc
    if rb["matches_model"]:
        assert "calibrated probabilities, three tiers, SHAP reasons and the information-level warning" in mc


def test_no_hard_coded_copies_of_the_new_numbers_in_ui_or_templates():
    import re
    nums = set()
    for r in FM["model_comparison_significance"]["rows"]:
        nums |= {f"{abs(r[k]):.4f}" for k in ("delta", "ci_lo", "ci_hi")}
    rb = FM["rule_baseline"]
    nums |= {f"{rb['rule'][k]:.4f}" for k in ("precision", "recall", "f1", "mcc")}
    nums |= {f"{v:.4f}" for v in FM["calibration_ef"]["ece"].values()}
    nums |= {f"{abs(v):.4f}" for v in FM["calibration_ef"]["brier_skill_score"].values()}
    for f in list((ROOT / "ui").rglob("*.py")) + list((ROOT / "templates").glob("*")):
        lit = set(re.findall(r"(?<![\d.])0\.\d{4}(?!\d)", f.read_text(encoding="utf-8")))
        assert not (lit & nums), f"{f.name} hard-codes {sorted(lit & nums)}"
