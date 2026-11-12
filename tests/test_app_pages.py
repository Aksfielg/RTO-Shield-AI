"""
tests/test_app_pages.py
=======================
Opens EVERY dashboard page with streamlit.testing.v1.AppTest and asserts:
  * no exception
  * no empty container (a Block with zero children)
  * no "$-" formatting anywhere in rendered text (negative money must read -$222.50)
plus page-specific content checks (home KPIs, presets == real rows, cascading filters, impact break-even, batch rules ...).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

PAGES = ["home", "score_order", "real_orders", "impact", "batch", "performance", "explainability", "robustness",
         "limitations", "methodology"]


def page_app(name: str, extra: str = "", call: str | None = None) -> AppTest:
    code = (f"import sys\nsys.path[:0] = [r'{ROOT}', r'{ROOT / 'src'}']\nfrom ui.pages import {name}\n{extra}\n"
            f"{call or name + '.render()'}")
    return AppTest.from_string(code, default_timeout=180)


def all_text(at: AppTest) -> str:
    """Every user-visible string we can see through AppTest."""
    parts = []
    for kind in ("markdown", "caption", "subheader", "header", "title", "text", "success", "info", "warning", "error", "code"):
        parts += [str(e.value) for e in at.get(kind)]
    for m in at.metric:
        parts += [str(m.value), str(m.delta or ""), str(m.label)]
    for df in list(at.dataframe) + list(at.get("table")):
        frame = pd.DataFrame(df.value)
        parts += [str(c) for c in frame.columns] + [str(x) for x in frame.astype(str).to_numpy().ravel()]
    return "\n".join(parts)


def empty_containers(node, path="main") -> list[str]:
    bad = []
    children = getattr(node, "children", None)
    if children is None:
        return bad
    if getattr(node, "type", None) == "flex_container" and len(children) == 0:
        bad.append(path)
    for k, ch in children.items():
        bad += empty_containers(ch, f"{path}/{k}")
    return bad


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders_cleanly(name):
    at = page_app(name).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not empty_containers(at.main), f"empty container(s) on {name}: {empty_containers(at.main)}"
    assert "$-" not in all_text(at), f"'$-' formatting on {name}"


def test_full_app_boots_with_sidebar_chrome_and_methodology_page():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    side = " ".join(m.value for m in at.sidebar.markdown)
    assert "Constante" in side and "doi:10.17632/8gx2fvg2k6.5" in side
    assert any("Proxy target" in m.value for m in at.main.markdown)
    assert "Methodology" in (ROOT / "app.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------ HOME
def test_home_titles_kpis_and_cards():
    at = page_app("home").run()
    text = all_text(at)
    assert "RTO Shield AI - Model 1" in text
    assert "Pre-dispatch delivery-failure risk scoring (public-data proxy)" in text
    for needle in ("ROC-AUC (95% CI)", "High-tier precision", "Standard + Second Class AUC", "What works", "What does not", "Next: Model 2 (uplift)"):
        assert needle in text, needle
    assert "near rule-based labels" in text.lower() and "synthetic" not in text.lower()
    assert len(at.get("plotly_chart")) == 1                        # split timeline
    assert "F1" not in "".join(m.value for m in at.markdown if "kpi-label" in m.value)


# ------------------------------------------------------------------ SCORE
def test_presets_equal_real_test_rows_and_real_orders_page():
    from ui import data
    from ui.pages import score_order
    import predict
    opts, reps = data.option_lists(), data.representative_orders()
    _, test_df = data.datasets()
    for tier, base in reps.items():
        d = score_order.default_values(base, opts)
        order = score_order.build_order(base, dict(d), d, opts["mode_days"])        # untouched form
        real = test_df[test_df["Order Id"] == base["Order Id"]].iloc[0].to_dict()
        a, b = predict.score_order(order), predict.score_order(real)
        assert a["risk_score"] == b["risk_score"] and a["tier"] == b["tier"] == tier
        for h in ("customer_prev_orders", "customer_prev_late_rate", "country_prev_orders", "country_prev_late_rate"):
            assert order[h] == real[h], f"history feature {h} not carried by the preset"


def test_score_page_info_level_note_and_flag_text():
    from ui import data
    from ui.pages import score_order
    kind, title, body = score_order.information_level("Standard Class", data.fm()["per_mode"])
    assert kind == "warning" and title.endswith("Limited")
    assert "cannot rank orders within this shipping mode" in body and "base rate" in body
    assert score_order.information_level("Same Day", data.fm()["per_mode"])[1].endswith("Informative")
    at = page_app("score_order").run()
    text = all_text(at)
    assert "imputed" in text and "Only the High tier is flagged" in text


def test_reasons_are_material_and_never_contradictory():
    from ui import components as C
    labels = ["A", "B", "C", "D"]
    out = C.filter_reasons(labels, [0.5, -0.4, 0.03, 0.2], ["a", "b", "c", "d"])
    assert out == ["A: a raises risk", "D: d raises risk"]                  # net positive -> only raising, |SHAP| >= 0.05
    assert all("lowers" in r for r in C.filter_reasons(labels, [-0.5, 0.1, -0.2, 0.0], list("abcd")))


# ------------------------------------------------------------------ REAL ORDERS
def test_real_orders_filters_cascade_and_reset():
    at = page_app("real_orders").run()
    assert not at.exception
    sel = [s for s in at.selectbox if s.label == "Shipping mode"][0]
    sel.select("Same Day").run()
    order_box = [s for s in at.selectbox if s.label.startswith("Order (first")][0]
    first_same_day = order_box.options[0]
    from ui import data
    _, test_df = data.datasets()
    assert int(test_df[test_df["Order Id"] == int(first_same_day)]["Shipping_Mode"].iloc[0] == "Same Day") == 1
    assert str(order_box.value) == str(first_same_day)                       # reset to the first matching order
    out = [s for s in at.selectbox if s.label == "Actual outcome"][0]
    out.select("Late only").run()
    order_box = [s for s in at.selectbox if s.label.startswith("Order (first")][0]
    assert str(order_box.value) == str(order_box.options[0])
    assert any("Correct" in m.value or "Missed" in m.value or "False alarm" in m.value for m in at.markdown)
    assert any("Current filter" in m.value for m in at.markdown)


def test_real_orders_confusion_summary_matches_flags():
    from ui import data
    from ui.pages import real_orders
    sc, thr = data.test_scores(), data.thr()
    cm = real_orders.confusion(sc["cal"] >= thr["tier_cutoffs"]["high_min"], sc["y"])
    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(sc["y"])


# ------------------------------------------------------------------ PERFORMANCE
def test_performance_comparison_chart_renders_and_lift_note():
    at = page_app("performance").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 6                                    # comparison, curves x2, calibration, lift, gain, tiers
    text = all_text(at)
    assert "equals the maximum possible" in text and "PR-AUC vs prevalence" in text and "ECE (final model)" in text


# ------------------------------------------------------------------ EXPLAINABILITY / ROBUSTNESS
def test_explainability_percentage_is_computed_and_has_howto():
    from ui import data
    at = page_app("explainability").run()
    text = all_text(at)
    ship = sum(g["share"] for g in data.grouped_importance() if g["factor"] in ("Shipping mode", "Scheduled shipping days"))
    assert f"{ship * 100:.1f}%" in text and "How to read this" in text


def test_robustness_covers_both_modes_and_sanity_tab():
    at = page_app("robustness").run()
    text = all_text(at)
    assert "Standard and Second Class" in text and "🟠 limited" in text and "🟢 informative" in text
    assert "Label-shuffle test" in text and "TRANSFER" in text and "Trivial-policy comparison" in text


# ------------------------------------------------------------------ IMPACT
def test_impact_has_mode_multiselect_sensitivity_and_clean_money():
    at = page_app("impact").run()
    assert any(m.label == "Shipping modes where intervention can work" for m in at.multiselect)
    text = all_text(at)
    assert "Break-Even Success Rate" in text and "$-" not in text
    assert "Assumptions, not measured" in text
    assert len(at.get("plotly_chart")) == 2
    # selecting Flag all must not duplicate the bar
    radio = [r for r in at.radio if r.label.startswith("Select triage")][0]
    radio.set_value("Flag all").run()
    assert not at.exception


def test_impact_break_even_math():
    import numpy as np
    from ui.pages import impact
    y = np.array([1, 1, 0, 0, 1, 0])
    mask = np.ones(6, bool)
    inside = np.array([1, 1, 1, 1, 0, 0], bool)
    be = impact.break_even(mask, y, inside, late_cost=10.0, int_cost=1.0, r_out=0.0)
    assert abs(be - (6 * 1.0) / (10.0 * 2)) < 1e-12                           # 6 contacts / (2 rescuable late x $10)
    assert abs(impact.economics(mask, y, inside, 10.0, 1.0, be, 0.0)["net"]) < 1e-9


# ------------------------------------------------------------------ BATCH
def test_batch_labels_missing_columns_and_template():
    from ui.pages import batch
    df = pd.DataFrame({"Shipping_Mode": ["Standard Class"], "order_value": [10.0]})
    req, opt = batch.missing_columns(df)
    assert req == ["Type", "n_items"] and "order_hour" in opt
    assert batch.outcome_labels(pd.Series([1, 0, 1])).tolist() == ["Late", "On-time", "Late"]
    at = page_app("batch").run()
    assert "Required: Shipping_Mode, Type, order_value, n_items" in all_text(at)


def test_batch_labelled_sample_shows_warning_late_labels_and_tier_by_mode():
    extra = "from ui import data\n_, test_df = data.datasets()\nsample = test_df.head(20)"
    at = page_app("batch", extra, "batch.render(sample)").run()
    assert not at.exception and not empty_containers(at.main)
    assert any("Small sample" in w.value for w in at.warning)
    text = all_text(at)
    assert "Late" in text and "On-time" in text
    assert "Tier counts by shipping mode" in text


# ------------------------------------------------------------------ METHODOLOGY / LIMITATIONS
def test_methodology_has_all_sections():
    at = page_app("methodology").run()
    text = all_text(at)
    for needle in ("1. Pipeline", "Allowed vs forbidden features", "Leakage guards", "Split dates", "The proxy target", "History rule"):
        assert needle in text, needle
    assert len(at.get("graphviz_chart")) == 1


def test_limitations_uses_near_rule_based_wording():
    text = all_text(page_app("limitations").run())
    assert "near rule-based labels" in text and "synthetic" not in text.lower()
    assert "Standard and Second Class" in text
