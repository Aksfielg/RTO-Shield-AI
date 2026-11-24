"""Payment-type proxy disclosure (the audit block, the dagger on reason codes, and the three pages that must mention it)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

import reasons  # noqa: E402
from test_app_pages import all_text, page_app  # noqa: E402

FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))


def test_final_metrics_has_the_type_audit_block():
    ta = FM["type_audit"]
    for sub in ("all_orders", "cleaned_orders"):
        assert {"model_a", "model_b", "delta_roc_auc", "delta_pr_auc"} <= set(ta[sub])
        assert len(ta[sub]["delta_roc_auc"]["ci"]) == 2
    assert {"all_orders", "cleaned_orders"} <= set(ta["equal_coverage"])
    assert ta["drop_list"]["dropped_features"] == ["Type", "is_cash"]
    assert isinstance(ta["cleaned_delta_ci_includes_zero"], bool)


def test_type_reason_codes_end_with_the_dagger_and_other_reasons_do_not():
    dagger = reasons.DAGGER
    assert reasons.reason_text("Payment type", "TRANSFER", True).endswith(dagger)
    assert not reasons.reason_text("Shipping mode", "Same Day", True).endswith(dagger)
    names = ["cat__Type_TRANSFER", "cat__Type_DEBIT", "cat__Shipping_Mode_Same Day", "num__order_hour"]
    codes = reasons.get_reason_codes(np.array([0.9, 0.0, 0.5, 0.0]), names, raw_dict={"Type": "TRANSFER", "Shipping_Mode": "Same Day"}, k=2)
    pay = [c for c in codes if c.startswith("Payment type")]
    assert pay and all(c.endswith(dagger) for c in pay)
    assert all(not c.endswith(dagger) for c in codes if not c.startswith("Payment type"))
    assert "Payment-type effects reflect how DataCo labels cancelled/fraud orders (all TRANSFER, all on time), not a real payment risk." \
        in reasons.TYPE_FOOTNOTE


def test_home_page_mentions_payment_type():
    at = page_app("home").run()
    assert not at.exception
    assert "payment type" in all_text(at).lower()


def test_robustness_ablation_tab_mentions_payment_type_and_the_artifact():
    at = page_app("robustness").run()
    assert not at.exception
    text = all_text(at).lower()
    assert "payment type" in text and "labelling artifact" in text


def test_limitations_page_mentions_payment_type():
    at = page_app("limitations").run()
    assert not at.exception
    assert "payment type" in all_text(at).lower()
