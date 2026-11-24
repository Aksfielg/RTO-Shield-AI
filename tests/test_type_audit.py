"""Payment-type proxy disclosure: audit present, dagger on every Type reason/label, no hard-coded copies of audit numbers."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import reasons  # noqa: E402

FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))
TA = FM["type_audit"]


def test_audit_structure_and_equal_coverage():
    for sub in ("all_orders", "cleaned_orders"):
        for m in ("model_a", "model_b"):
            assert {"roc_auc", "roc_auc_ci", "pr_auc", "pr_auc_ci"} <= set(TA[sub][m])
        assert len(TA[sub]["delta_roc_auc"]["ci"]) == 2
        eq = TA["equal_coverage"][sub]
        assert eq["k"] > 0 and abs(eq["flagged_share"] - eq["k"] / eq["n"]) < 1e-12
    # equal coverage on all orders uses the size of Model A's High tier
    high = next(p for p in FM["policies"] if p["policy"].startswith("High tier"))
    assert TA["equal_coverage"]["all_orders"]["k"] == high["flagged_orders"]
    assert TA["all_orders"]["model_a"]["roc_auc"] == pytest.approx(FM["test"]["roc_auc"], abs=1e-9)


def test_disclosure_wording_still_valid():
    """The docs/app say 'the cleaned-order interval includes zero'. If that stops being true, the wording must change."""
    lo, hi = TA["cleaned_orders"]["delta_roc_auc"]["ci"]
    assert lo <= 0 <= hi, "cleaned-order delta CI no longer includes 0: update the 'Payment-type proxy' wording in templates/ and ui/"


def test_payment_reason_codes_end_with_dagger():
    assert reasons.reason_text("Payment type", "TRANSFER", True).endswith(reasons.DAGGER)
    assert reasons.reason_text("Shipping mode", "Same Day", True) == "Shipping mode: Same Day raises risk"
    names = ["cat__Type_TRANSFER", "cat__Type_DEBIT", "num__order_hour"]
    codes = reasons.get_reason_codes(np.array([0.9, 0.0, 0.0]), names, raw_dict={"Type": "TRANSFER"}, k=1)
    assert codes and codes[0].endswith(reasons.DAGGER) and "Payment type" in codes[0]
    assert reasons.has_payment_reason(codes) and not reasons.has_payment_reason(["Shipping mode: Same Day raises risk"])
    assert "cancelled/fraud" in reasons.TYPE_FOOTNOTE and "not a real payment risk" in reasons.TYPE_FOOTNOTE


def test_ui_filter_reasons_and_charts_carry_dagger():
    from ui import charts, components as C
    out = C.filter_reasons(["Payment type", "Shipping mode"], [0.8, 0.6], ["TRANSFER", "Same Day"])
    assert out[0].endswith(reasons.DAGGER) and not out[1].endswith(reasons.DAGGER)
    fig = charts.grouped_shap_bar(["Payment type", "Shipping mode"], np.array([0.8, 0.6]), ["TRANSFER", "Same Day"])
    assert any(reasons.DAGGER in str(y) for y in fig.data[0].y)
    fig2 = charts.importance_grouped([{"factor": "Payment type", "share": 0.2}, {"factor": "Shipping mode", "share": 0.5}])
    assert any(str(y) == "Payment type" + reasons.DAGGER for y in fig2.data[0].y)
    assert charts.pretty_feature("cat__Type_TRANSFER").startswith("Payment type" + reasons.DAGGER)


def _audit_numbers() -> set[str]:
    nums = []
    for sub in ("all_orders", "cleaned_orders"):
        for m in ("model_a", "model_b"):
            nums += [TA[sub][m]["roc_auc"], TA[sub][m]["pr_auc"]]
        nums += [TA[sub]["delta_roc_auc"]["point"], TA[sub]["delta_pr_auc"]["point"]]
        nums += TA[sub]["delta_roc_auc"]["ci"]
        eq = TA["equal_coverage"][sub]
        nums += [eq["model_a"]["precision"], eq["model_b"]["precision"], eq["delta_precision"]["point"]]
    return {f"{abs(n):.4f}" for n in nums}


def test_no_hard_coded_audit_numbers_in_ui_or_templates():
    files = list((ROOT / "ui").rglob("*.py")) + list((ROOT / "templates").glob("*"))
    for f in files:
        text = f.read_text(encoding="utf-8")
        literals = set(re.findall(r"(?<![\d.])0\.\d{4}(?!\d)", text))
        hits = literals & _audit_numbers()
        assert not hits, f"{f.relative_to(ROOT)} hard-codes audit number(s) {sorted(hits)}; use M.type_audit.* / fm['type_audit']"
