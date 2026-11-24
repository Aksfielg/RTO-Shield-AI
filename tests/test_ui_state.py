"""
Streamlit AppTest regression tests for widget STATE:
  * Score an Order: every preset loads every widget from its own holdout row, payment-type reasons agree with the widget,
    and Custom -> preset -> Custom never carries a stale value over.
  * Impact Simulator: default selection, dynamic caption, First Class warning, break-even note quoting the slider.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from test_app_pages import all_text, page_app  # noqa: E402

PRESET_IDS = {"High": 67041, "Medium": 67047, "Low": 67036}      # the real holdout orders the presets are built from


# ----------------------------------------------------------------------------------------------- helpers
def _sel(at, label):
    return next(s for s in at.selectbox if s.label == label)


def _num(at, label):
    return next(s for s in at.number_input if s.label == label)


def _sld(at, label):
    return next(s for s in at.slider if s.label == label)


def widgets(at) -> dict:
    return {"mode": _sel(at, "Shipping mode").value, "type": _sel(at, "Payment type").value, "market": _sel(at, "Market").value,
            "dept": _sel(at, "Department").value, "seg": _sel(at, "Customer segment").value,
            "hour": _sld(at, "Order hour (0-23)").value, "month": _sld(at, "Order month").value,
            "disc": _sld(at, "Discount rate").value, "value": _num(at, "Order value ($)").value, "items": _num(at, "Number of items").value}


def expected_from_row(row: dict, opts: dict) -> dict:
    lo, hi = opts["value_range"]
    return {"mode": row["Shipping_Mode"], "type": row["Type"], "market": row["Market"], "dept": row["Department_Name"],
            "seg": row["Customer_Segment"], "hour": int(row["order_hour"]), "month": int(row["order_month"]),
            "disc": min(max(row["avg_discount_rate"], 0.0), 0.5), "value": min(max(row["order_value"], lo), hi),
            "items": int(row["n_items"])}


def assert_same(actual: dict, expected: dict, where: str):
    for k in ("mode", "type", "market", "dept", "seg", "hour", "month", "items"):
        assert actual[k] == expected[k], f"{where}: {k} shows {actual[k]!r}, expected {expected[k]!r}"
    assert actual["disc"] == pytest.approx(expected["disc"], abs=0.006), f"{where}: discount"
    assert actual["value"] == pytest.approx(expected["value"], abs=0.011), f"{where}: order value"


def pick_preset(at, tier):
    radio = at.radio[0]
    radio.set_value(next(o for o in radio.options if f"Real {tier.lower()}-risk" in o))
    at.run()


def payment_reason_types(at) -> list[str]:
    out = []
    for m in at.markdown:
        out += re.findall(r"Payment type: (\w+) (?:raises|lowers) risk", str(m.value))
    return out


@pytest.fixture(scope="module")
def reps_opts():
    from ui import data
    return data.representative_orders(), data.option_lists()


# ----------------------------------------------------------------------------------------------- Score an Order
def test_presets_are_the_documented_holdout_orders(reps_opts):
    reps, _ = reps_opts
    assert {t: reps[t]["Order Id"] for t in PRESET_IDS} == PRESET_IDS


@pytest.mark.parametrize("tier", ["High", "Medium", "Low"])
def test_each_preset_loads_every_widget_from_its_own_holdout_row(tier, reps_opts):
    reps, opts = reps_opts
    at = page_app("score_order").run()
    assert not at.exception
    pick_preset(at, tier)
    assert not at.exception
    assert_same(widgets(at), expected_from_row(reps[tier], opts), f"{tier} preset #{PRESET_IDS[tier]}")
    for t in payment_reason_types(at):                       # any payment-type reason names the widget's own Type
        assert t == _sel(at, "Payment type").value, f"reason says {t}, widget says {_sel(at, 'Payment type').value}"


def test_payment_reason_codes_use_the_same_type_as_the_widget_in_every_preset(reps_opts):
    reps, _ = reps_opts
    at = page_app("score_order").run()
    seen = 0
    for tier in ("High", "Medium", "Low", "High", "Low", "Medium"):          # includes every transition direction
        pick_preset(at, tier)
        for t in payment_reason_types(at):
            seen += 1
            assert t == reps[tier]["Type"] == _sel(at, "Payment type").value
    assert seen > 0, "no preset produced a payment-type reason, so the check never ran"


def test_custom_then_preset_then_custom_carries_no_stale_value(reps_opts):
    reps, opts = reps_opts
    from ui.pages.score_order import default_values
    custom_defaults = default_values({}, opts)
    at = page_app("score_order").run()
    assert_same(widgets(at), {"mode": custom_defaults["Shipping_Mode"], "type": custom_defaults["Type"], "market": custom_defaults["Market"],
                              "dept": custom_defaults["Department_Name"], "seg": custom_defaults["Customer_Segment"],
                              "hour": custom_defaults["order_hour"], "month": custom_defaults["order_month"],
                              "disc": custom_defaults["avg_discount_rate"], "value": custom_defaults["order_value"],
                              "items": custom_defaults["n_items"]}, "fresh Custom input")
    # edit several Custom widgets, so a stale carry-over would be visible on every field
    _sel(at, "Payment type").set_value("TRANSFER")
    other_market = next(m for m in opts["markets"] if m != reps["Medium"]["Market"] and m != custom_defaults["Market"])
    _sel(at, "Market").set_value(other_market)
    _sld(at, "Order hour (0-23)").set_value(3)
    _num(at, "Number of items").set_value(9)
    at.run()
    edited = widgets(at)
    assert (edited["type"], edited["market"], edited["hour"], edited["items"]) == ("TRANSFER", other_market, 3, 9)

    pick_preset(at, "Medium")                                              # -> #67047: every widget from ITS row
    assert_same(widgets(at), expected_from_row(reps["Medium"], opts), "Medium preset after Custom edits")

    radio = at.radio[0]
    radio.set_value("Custom input")
    at.run()                                                               # -> Custom again: defaults, nothing stale
    back = widgets(at)
    assert (back["type"], back["market"], back["hour"], back["items"]) == (
        custom_defaults["Type"], custom_defaults["Market"], custom_defaults["order_hour"], custom_defaults["n_items"])
    assert back["market"] != other_market and back["hour"] != 3 and back["items"] != 9
    med = expected_from_row(reps["Medium"], opts)
    assert (back["mode"], back["dept"], back["seg"]) != (med["mode"], med["dept"], med["seg"]), "preset values leaked into Custom"
    assert not at.exception


# ----------------------------------------------------------------------------------------------- Impact Simulator
def _impact():
    at = page_app("impact").run()
    assert not at.exception
    return at


def _captions(at):
    return " ".join(str(c.value) for c in at.caption)


def test_impact_default_selection_excludes_first_class():
    at = _impact()
    ms = at.multiselect[0]
    assert "First Class" in ms.options and "First Class" not in ms.value
    assert set(ms.value) == set(ms.options) - {"First Class"}
    assert not [w for w in at.warning if "First Class lateness" in str(w.value)]


@pytest.mark.parametrize("pct", [0, 7, 23])
def test_impact_caption_and_breakeven_quote_the_current_other_modes_slider(pct):
    at = _impact()
    r_out = next(s for s in at.slider if s.label == "Success rate in all other modes (%)")
    r_out.set_value(pct)
    at.run()
    caps = _captions(at)
    shown = f"{pct:.1f}%"
    assert re.search(rf"get {re.escape(shown)} success", caps), f"caption does not quote the slider ({shown}): {caps[:300]}"
    m = re.search(r"with ([\d.]+)% success outside them", caps)
    assert m, "break-even note not found"
    assert float(m.group(1)) == float(pct), "break-even note quotes a different rate than the slider"


def test_impact_selecting_first_class_shows_the_warning_and_removing_it_clears_it():
    at = _impact()
    ms = at.multiselect[0]
    ms.set_value(list(ms.options))
    at.run()
    warn = [str(w.value) for w in at.warning]
    assert any("First Class lateness comes from the carrier's 1-day promise; including it inflates the savings." in w for w in warn)
    ms = at.multiselect[0]
    ms.set_value([o for o in ms.options if o != "First Class"])
    at.run()
    assert not [w for w in at.warning if "First Class lateness" in str(w.value)]
    assert "First Class" in _captions(at)                                    # now listed as an excluded mode again
