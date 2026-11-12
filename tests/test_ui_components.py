"""
tests/test_ui_components.py
===========================
Unit tests for the shared UI primitives (formatters, wording, tables) and structural guards on the UI code:
no split <div> card patterns, no matplotlib/PNG charts, one score display path, tier from the unrounded score.
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from ui import components as C  # noqa: E402

UI_FILES = [p for p in (ROOT / "ui").rglob("*.py")] + [ROOT / "app.py"]


# ------------------------------------------------------------------ formatters
def test_fmt_pct_clips_extremes_only_when_asked():
    assert C.fmt_pct(0.004) == "<1%"
    assert C.fmt_pct(0.995) == ">99%"
    assert C.fmt_pct(0.5) == "50.0%"
    assert C.fmt_pct(0.387) == "38.7%"
    assert C.fmt_pct(0.004, clip=False) == "0.4%"
    assert C.fmt_pct(1.0, clip=False) == "100.0%"
    assert C.fmt_pct(None) == "n/a" and C.fmt_pct(float("nan")) == "n/a"


def test_fmt_money_sign_before_symbol():
    assert C.fmt_money(-222.5) == "-$222.50"
    assert C.fmt_money(4731.5) == "$4,731.50"
    assert C.fmt_money(4954.0, signed=True) == "+$4,954.00"
    assert C.fmt_money(0.0) == "$0.00"


def test_readable_headers():
    assert C.readable("roc_auc") == "ROC-AUC"
    assert C.readable("pr_auc_lift") == "PR-AUC Lift"
    assert C.readable("flagged_orders") == "Flagged Orders"
    assert C.readable("late_in_decile") == "Late In Decile"
    assert C.readable("Shipping Mode") == "Shipping Mode"          # already readable


# ------------------------------------------------------------------ wording
@pytest.mark.parametrize("tier,needle", [("Low", "lower-risk"), ("Medium", "elevated risk"), ("High", "highest-risk")])
def test_plain_sentence_wording_follows_tier(tier, needle):
    s = C.plain_sentence(0.45, tier, "Shipping mode", True)
    assert needle in s
    assert "low probability" not in s.lower()
    assert C.fmt_pct(0.45) in s


def test_plain_sentence_handles_driver_direction():
    assert "supported by favourable" in C.plain_sentence(0.2, "Low", "Shipping mode", False)
    assert "even though" in C.plain_sentence(0.2, "Low", "Shipping mode", True)
    assert "lowers the score" in C.plain_sentence(0.9, "High", "Shipping mode", False)


def test_badge_gauge_and_text_use_the_same_display():
    from ui import charts
    score = 0.3871234
    html = C.risk_badge_html("Low", score)
    gauge = charts.gauge(score, 0.4657, 0.7423)
    text = C.plain_sentence(score, "Low", "x", False)
    shown = C.fmt_pct(score)
    assert shown in html and shown in text and shown in gauge.layout.annotations[0].text
    assert "&lt;1%" in C.risk_badge_html("Low", 0.0049) or "<1%" in C.risk_badge_html("Low", 0.0049)   # clipped everywhere


def test_tier_is_computed_from_the_unrounded_score():
    import predict
    predict._load_artifacts()
    cut = predict._CACHE["high_cut"]
    just_below = cut - 1e-9                      # rounds to the cutoff at 6 dp, but is still Medium
    assert round(just_below, 6) == round(cut, 6)
    assert predict._tier(just_below) == "Medium"
    assert predict._tier(cut) == "High"


# ------------------------------------------------------------------ styled_table
def test_styled_table_hides_index_renames_and_formats(monkeypatch):
    import streamlit as st
    seen = {}
    monkeypatch.setattr(st, "dataframe", lambda data, **kw: seen.update(data=data, kw=kw))
    df = pd.DataFrame({"roc_auc": [0.123456, 0.5], "n_features": [3, 4], "late_rate": [float("nan"), 0.25]},
                      index=[7, 8])
    C.styled_table(df, formats={"late_rate": lambda v: C.fmt_pct(v, clip=False)})
    assert seen["kw"]["hide_index"] is True
    view = seen["data"].data                                   # Styler -> underlying frame
    assert list(view.columns) == ["ROC-AUC", "Features (#)", "Late Rate"]
    assert list(view.index) == [0, 1]                          # index reset (and hidden)
    html = seen["data"].to_html()
    assert "0.123" in html and "0.123456" not in html          # 3 decimals
    assert "25.0%" in html and "–" in html                     # custom format; NaN shown as a dash, never None/nan


# ------------------------------------------------------------------ structural guards
def test_no_split_div_card_patterns():
    bad = []
    for p in UI_FILES:
        text = p.read_text(encoding="utf-8")
        if re.search(r"st\.markdown\(\s*f?['\"]</div>['\"]", text):
            bad.append((p.name, "standalone closing </div>"))
        if re.search(r"rto-card(?!-title)", text):
            bad.append((p.name, "legacy rto-card card class"))
    assert not bad, bad


def test_no_matplotlib_or_png_charts_in_ui():
    for p in UI_FILES:
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+matplotlib", text, re.M), p.name
        assert "st.pyplot" not in text and "plt." not in text, p.name
        assert "st.image" not in text and ".png" not in text, p.name


def test_plotly_only_rendered_through_shared_show():
    for p in UI_FILES:
        if p.name == "components.py":
            continue
        assert "st.plotly_chart" not in p.read_text(encoding="utf-8"), p.name


def test_tables_only_via_styled_table_and_cards_via_container():
    for p in UI_FILES:
        if p.name == "components.py":
            continue
        text = p.read_text(encoding="utf-8")
        assert "st.dataframe" not in text and "st.table" not in text, p.name


def test_money_is_only_formatted_through_fmt_money():
    """Hand-built "$" strings would render negatives as "$-222.50"; everything must go through C.fmt_money."""
    for p in UI_FILES:
        if p.name == "components.py":
            continue
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"""f?["']\$\{""", text), f"{p.name}: build money with C.fmt_money"
        assert not re.search(r"""["']\$["']\s*\+""", text), p.name


def test_fmt_pp_never_shows_negative_zero():
    assert C.fmt_pp(-0.0000155) == "+0.0 pp" and C.fmt_pp(-0.0913) == "-9.1 pp" and C.fmt_pp(0.012) == "+1.2 pp"
