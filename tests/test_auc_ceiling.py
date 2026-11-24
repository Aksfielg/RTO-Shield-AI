"""
Leakage tripwire. The honest ceiling of checkout-time features on this data is a test ROC-AUC of about 0.77.
If any of these assertions fails, the model has probably started seeing the answer.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))

MSG = "AUC looks too good - suspect leakage; check FORBIDDEN_FEATURES and the history gating."


def _mode_auc(mode: str) -> float:
    return next(r["roc_auc"] for r in FM["per_mode"] if r["mode"] == mode)


def test_final_test_auc_below_ceiling():
    auc = FM["test"]["roc_auc"]
    assert auc < 0.80, f"{MSG} (final test ROC-AUC = {auc:.4f}, limit 0.80)"


def test_standard_and_second_class_stay_near_chance():
    for mode in ("Standard Class", "Second Class"):
        auc = _mode_auc(mode)
        assert auc < 0.60, f"{MSG} ({mode} within-mode test ROC-AUC = {auc:.4f}, limit 0.60)"


def test_label_shuffle_auc_stays_at_chance():
    ls = FM["label_shuffle"]
    for split in ("val_roc_auc", "test_roc_auc"):
        assert 0.47 <= ls[split] <= 0.53, f"{MSG} (label-shuffle {split} = {ls[split]:.4f}, expected 0.47-0.53)"
