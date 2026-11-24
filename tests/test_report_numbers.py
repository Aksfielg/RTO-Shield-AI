"""outputs/report_numbers.md (every number with its JSON path) and docs/figure_captions.md (exact order/settings/score per screenshot)."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))
THR = json.loads((ROOT / "models" / "thresholds.json").read_text(encoding="utf-8"))
REPORT = (ROOT / "outputs" / "report_numbers.md").read_text(encoding="utf-8")


def resolve(path: str, root=FM):
    """Resolve 'a.b[key=value].c' inside the metrics JSON (same selector syntax as the doc templates)."""
    cur = root
    for m in re.finditer(r"\[([^\]=]+)=([^\]]+)\]|([^.\[\]]+)", path):
        k, v, key = m.groups()
        cur = cur[key] if key is not None else next(r for r in cur if str(r[k]) == v)
    return cur


def _rows():
    for line in REPORT.splitlines():
        m = re.match(r"\| (.+?) \| (.+?) \| `final_metrics.json` → `([^`]+)` \|$", line)
        if m and m.group(1) not in ("Quantity",):
            yield m.group(1), m.group(2), m.group(3)


def test_every_two_column_row_points_at_a_real_json_value_and_prints_it_correctly():
    rows = list(_rows())
    assert len(rows) > 40
    for label, shown, path in rows:
        raw = resolve(path)                                   # raises if the path is wrong
        if isinstance(raw, list):
            nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", shown)]
            assert len(nums) == 2 and nums == pytest.approx([round(v, len(re.findall(r"\.(\d+)", shown)[0])) for v in raw], abs=1e-9), label
        elif isinstance(raw, bool) or isinstance(raw, str):
            assert str(raw) in shown, label
        elif isinstance(raw, (int, float)):
            txt = shown.replace(",", "")
            if txt.endswith("%"):
                assert float(txt[:-1]) == pytest.approx(raw * 100, abs=0.051), label
            else:
                dec = len(txt.split(".")[1]) if "." in txt else 0
                assert float(txt) == pytest.approx(raw, abs=0.5 * 10 ** -dec + 1e-12), f"{label}: shows {shown}, JSON has {raw}"


def test_report_contains_every_requested_table():
    for heading in ("1. Test ROC-AUC", "2. PR-AUC", "3. Calibration", "4. Risk tiers with Wilson 95% intervals", "5. Per shipping mode",
                    "6. Per-mode flagging", "7. Model comparison with paired deltas", "8. Feature ablation", "9. Plain-rule baseline",
                    "10. Rolling-origin backtest", "11. Population Stability Index", "12. Label-shuffle check", "13. Split sizes and dates",
                    "14. Test suite and run time"):
        assert f"## {heading}" in REPORT, heading
    t = FM["test"]
    assert f"{t['roc_auc']:.3f}" in REPORT and f"{t['roc_auc']:.4f}" in REPORT                      # 3 and 4 decimals
    assert f"{FM['multi_seed']['roc_auc_mean']:.4f}" in REPORT and f"{FM['multi_seed']['roc_auc_std']:.4f}" in REPORT
    for lab in ("All features", "No shipping mode/days", "No history features", "No time features", "No payment/is_cash", "Shipping only (baseline)"):
        assert f"ablation[label={lab}]" in REPORT, lab
    assert REPORT.count("backtest.rows[fold=") == len(FM["backtest"]["rows"]) == 5
    assert REPORT.count("psi.rows[feature=") == len(FM["psi"]["rows"])
    for r in FM["tiers"]["test"].values():
        assert f"[{r['late_rate_ci_lo'] * 100:.1f}%, {r['late_rate_ci_hi'] * 100:.1f}%]" in REPORT
    assert f"{FM['tests']['total']:,}" in REPORT and f"{FM['run']['total_minutes']:.1f}" in REPORT
    assert f"{THR['tier_cutoffs']['high_min']:.4f}" in REPORT
    for k in ("train", "val", "test"):
        assert FM["splits"][k]["date_min"] in REPORT and FM["splits"][k]["date_max"] in REPORT
    assert "{{" not in REPORT and "None" not in REPORT and "nan" not in REPORT.lower().replace("finance", "")


# ------------------------------------------------------------------ figure captions
CAP = json.loads((ROOT / "docs" / "figure_captions.json").read_text(encoding="utf-8"))
CAP_MD = (ROOT / "docs" / "figure_captions.md").read_text(encoding="utf-8")


def test_every_screenshot_has_a_caption_and_every_readme_image_is_captioned_by_name():
    shots = {p.name for p in (ROOT / "docs" / "screenshots").glob("*.png")}
    assert {f["file"] for f in CAP.values()} == shots
    for f in CAP.values():
        assert f"docs/screenshots/{f['file']}" in CAP_MD and f["page"] in f["caption"] and f["settings"] in f["caption"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for link in re.findall(r"!\[[^\]]*\]\(docs/screenshots/([^)]+)\)", readme):
        fig = next(f for f in CAP.values() if f["file"] == link)
        assert fig["caption"] in readme, f"README embeds {link} without its exact caption"


def test_score_order_caption_names_preset_versus_custom_with_exact_scores():
    import predict
    from ui import data
    from ui.pages import score_order
    from ui import components as C
    reps, opts = data.representative_orders(), data.option_lists()
    hi = reps["High"]
    cap = CAP["score-order"]
    assert f"#{hi['Order Id']}" in cap["caption"] and "preset" in cap["caption"].lower() and "Custom input" in cap["caption"]
    assert C.fmt_pct(predict.score_order(hi)["risk_score"]) in cap["caption"]
    d = score_order.default_values({}, opts)
    vals = {"Shipping_Mode": "Second Class", "Type": "CASH", "order_value": d["order_value"], "n_items": d["n_items"],
            "avg_discount_rate": d["avg_discount_rate"], "Market": d["Market"], "Department_Name": d["Department_Name"],
            "Customer_Segment": d["Customer_Segment"], "order_hour": d["order_hour"], "order_dow": d["order_dow"], "order_month": d["order_month"]}
    sc = predict.score_order(score_order.build_order({}, vals, d, opts["mode_days"]))["risk_score"]
    assert C.fmt_pct(sc) in CAP_MD and "Second Class + CASH" in CAP_MD


def test_captions_were_checked_against_the_text_read_from_the_screenshots():
    manifest = json.loads((ROOT / "docs" / "screenshots" / "manifest.json").read_text(encoding="utf-8"))["pages"]
    assert set(manifest) == set(CAP)
    so = manifest["score-order"]
    assert f"#{CAP['score-order']['facts']['preset_order_id']}" in so["radio"]
    from ui import components as C
    assert C.fmt_pct(CAP["score-order"]["facts"]["preset_score"]) in so["text"]
    t = FM["test"]
    assert f"{t['roc_auc']:.3f}" in manifest["home"]["text"] and f"{t['roc_auc']:.3f}" in manifest["performance"]["text"]
