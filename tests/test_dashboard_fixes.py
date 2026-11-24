"""Regression tests for the dashboard fixes: preset state, dynamic Impact caption, per-mode flagging totals, batch rules,
chart label room, display-only country names and the toolbar setting."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from test_app_pages import all_text, page_app  # noqa: E402

FM = json.loads((ROOT / "outputs" / "final_metrics.json").read_text(encoding="utf-8"))
THR = json.loads((ROOT / "models" / "thresholds.json").read_text(encoding="utf-8"))


def _test_orders() -> pd.DataFrame:
    o = pd.read_parquet(ROOT / "data" / "processed" / "orders.parquet").sort_values("order_date").reset_index(drop=True)
    return o.iloc[int(np.floor(len(o) * 0.85)):]


# ---------------------------------------------------------------- 2. Score an Order: preset state ---------------------------
def _state(at, name):
    return at.session_state["so_w_" + name]


def test_presets_load_their_own_row_and_clear_custom_state():
    from ui import data
    reps = data.representative_orders()
    at = page_app("score_order").run()
    assert not at.exception
    next(s for s in at.selectbox if s.label == "Payment type").set_value("TRANSFER")
    at.run()
    assert _state(at, "type") == "TRANSFER"
    for tier in ("High", "Medium", "Low"):
        radio = at.radio[0]
        radio.set_value([o for o in radio.options if f"Real {tier.lower()}-risk" in o][0])
        at.run()
        row = reps[tier]
        assert _state(at, "type") == row["Type"], f"{tier}: payment type shows stale state"
        assert _state(at, "mode") == row["Shipping_Mode"]
        assert _state(at, "market") == row["Market"] and _state(at, "dept") == row["Department_Name"]
        assert _state(at, "seg") == row["Customer_Segment"]
        assert _state(at, "hour") == int(row["order_hour"]) and _state(at, "month") == int(row["order_month"])
        assert _state(at, "items") == int(row["n_items"])
    # back to custom: defaults again, not the last preset and not the earlier custom TRANSFER
    radio = at.radio[0]
    radio.set_value("Custom input")
    at.run()
    assert _state(at, "type") != "TRANSFER"


def test_high_and_medium_presets_have_different_payment_types():
    """#67041 is DEBIT and #67047 is TRANSFER in the data: the form must show exactly that, in both directions."""
    from ui import data
    reps = data.representative_orders()
    at = page_app("score_order").run()
    for tier in ("Medium", "High", "Medium"):
        radio = at.radio[0]
        radio.set_value([o for o in radio.options if f"Real {tier.lower()}-risk" in o][0])
        at.run()
        assert next(s for s in at.selectbox if s.label == "Payment type").value == reps[tier]["Type"]


def test_untouched_preset_scores_like_the_real_row():
    import predict
    from ui import data
    reps = data.representative_orders()
    at = page_app("score_order").run()
    radio = at.radio[0]
    radio.set_value([o for o in radio.options if "medium-risk" in o][0])
    at.run()
    expect = predict.score_order(reps["Medium"])["risk_score"]
    assert any(f"{expect * 100:.1f}%" in str(m.value) for m in at.markdown), "preset score differs from the real row's score"


def test_risk_score_tooltip_text():
    from ui import components as C
    assert "Isotonic calibration maps many raw scores to the same probability" in C.RISK_SCORE_HELP
    assert "different orders can show identical percentages" in C.RISK_SCORE_HELP


# ---------------------------------------------------------------- 1. Impact caption ------------------------------------------
def test_impact_caption_is_dynamic_and_first_class_warns():
    at = page_app("impact").run()
    assert not at.exception
    assert not [w for w in at.warning if "First Class lateness" in str(w.value)], "default selection must not warn"
    text = all_text(at)
    assert "First Class" in text and "success" in text
    r_out = next(s for s in at.slider if s.label.startswith("Success rate in all other modes"))
    r_out.set_value(5)
    at.run()
    assert "5.0%" in all_text(at), "caption must follow the slider value"
    ms = at.multiselect[0]
    ms.set_value(list(ms.options))                       # include First Class
    at.run()
    assert any("First Class lateness comes from the carrier's 1-day promise; including it inflates the savings." in str(w.value)
               for w in at.warning)


# ---------------------------------------------------------------- 3. per-mode flagging ---------------------------------------
def test_per_mode_flagging_totals_equal_test_totals():
    rows = FM["per_mode_flagging"]
    te = _test_orders()
    high = next(p for p in FM["policies"] if p["policy"].startswith("High tier"))
    assert sum(r["orders"] for r in rows) == FM["test"]["n"] == len(te)
    assert sum(r["late"] for r in rows) == int(te["Late_delivery_risk"].sum())
    assert sum(r["flagged"] for r in rows) == high["flagged_orders"]
    assert sum(r["caught"] for r in rows) == round(high["recall"] * sum(r["late"] for r in rows))
    assert sum(r["missed"] for r in rows) == sum(r["late"] for r in rows) - sum(r["caught"] for r in rows)
    assert sum(r["share_of_missed"] for r in rows) == pytest.approx(1.0)
    for r in rows:
        assert r["caught"] <= r["late"] and r["caught"] <= r["flagged"]


def test_performance_page_shows_the_table_and_data_built_sentence():
    at = page_app("performance").run()
    assert not at.exception
    text = all_text(at)
    worst = max(FM["per_mode_flagging"], key=lambda r: r["share_of_missed"])
    assert worst["mode"] in text and "capped by design" in text


# ---------------------------------------------------------------- 4. batch -------------------------------------------------
def test_batch_demo_caption_and_small_tier_chart_hidden():
    at = page_app("batch").run()
    next(b for b in at.button if "with target" in b.label).click()
    at.run()
    assert not at.exception
    text = all_text(at)
    assert "Demo sample, not a result; the headline is the held-out test set." in text
    big = [r for r in __import__("predict").evaluate_batch(  # noqa: E501
        _test_orders().sample(50, random_state=42)["Late_delivery_risk"].values,
        np.linspace(0, 1, 50), np.linspace(0, 1, 50), np.array(["Low"] * 20 + ["Medium"] * 20 + ["High"] * 10))["tier_calibration"]
        if not r["suppressed"]]
    assert len(big) < 2
    assert "Calibration chart hidden" in text


def test_predicted_vs_actual_greys_suppressed_tiers_and_shows_n():
    from ui import charts
    rows = [{"tier": "Low", "n": 45, "mean_predicted": .3, "actual_late_rate": .35, "suppressed": False},
            {"tier": "Medium", "n": 12, "mean_predicted": .5, "actual_late_rate": .4, "suppressed": True},
            {"tier": "High", "n": 60, "mean_predicted": .8, "actual_late_rate": .75, "suppressed": False}]
    fig = charts.predicted_vs_actual(rows)
    xs = list(fig.data[0].x)
    assert "n = 12" in xs[1] and "suppressed" in xs[1] and "suppressed" not in xs[0]
    assert fig.data[0].marker.color[1] != fig.data[0].marker.color[0]


# ---------------------------------------------------------------- 5. chart label room --------------------------------------
def test_outside_labels_are_never_clipped_and_have_room():
    from ui import charts
    grouped = [{"factor": f"F{i}", "share": s} for i, s in enumerate([0.54, 0.19, 0.13, 0.09])]
    fig = charts.importance_grouped(grouped)
    assert fig.data[0].cliponaxis is False
    assert fig.layout.xaxis.range[1] > 0.54 * 1.15
    fig = charts.shap_waterfall(["A", "B", "C"], np.array([0.5, -0.18, 0.02]), ["1", "4", "x"], base_value=-0.2)
    assert fig.data[0].cliponaxis is False
    lo, hi = fig.layout.xaxis.range
    # running totals: -0.2 -> 0.3 -> 0.12 -> 0.14; the axis must extend beyond both ends so outside labels sit inside the plot
    assert lo < -0.2 - 0.05 and hi > 0.3 + 0.05
    for builder in (lambda: charts.lift_bar(pd.DataFrame({"decile": [1, 2], "lift": [1.8, 0.4]})),
                    lambda: charts.tier_counts_bar({"Low": 5, "Medium": 6, "High": 9}, {"Low": .2, "Medium": .3, "High": .5},
                                                   {"Low": "L", "Medium": "M", "High": "H"})):
        f = builder()
        assert all(t.cliponaxis is False for t in f.data if t.type == "bar" and t.text is not None)


# ---------------------------------------------------------------- 7. countries ---------------------------------------------
def test_country_names_map_to_english_for_display_only():
    from ui import data
    countries = pd.read_parquet(ROOT / "data" / "processed" / "orders.parquet")["Order_Country"].unique()
    english = set(data.COUNTRY_EN.values())
    unmapped = [c for c in countries if data.en_country(c) not in english]
    assert not unmapped, f"countries without an English display name: {unmapped}"
    assert data.en_country("Reino Unido") == "United Kingdom" and data.en_country("Alemania") == "Germany"
    assert data.en_text("Country: Reino Unido raises risk") == "Country: United Kingdom raises risk"
    assert data.en_country("Canada") == "Canada" and data.en_country(None) is None
    # model inputs are untouched: the parquet still holds the DataCo names
    assert "Reino Unido" in set(countries)


# ---------------------------------------------------------------- 8. toolbar -----------------------------------------------
def test_toolbar_mode_hides_deploy_button():
    import tomllib
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["toolbarMode"] == "viewer"
