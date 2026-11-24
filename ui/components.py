"""
ui/components.py
================
Reusable UI building blocks.  Cards are ALWAYS ``st.container(border=True)`` -
no split ``<div>`` open/close markdown (which renders empty rounded boxes).

Formatters (one place, used by every page)
    fmt_pct(x)      model scores / rates with <1% and >99% clipping
    fmt_money(x)    -$222.50 style
Widgets
    page_header, proxy_badge, kpi_card, risk_badge, callout, styled_table, show
Text
    plain_sentence  tier-correct one-sentence explanation of a score
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reasons import PAYMENT_LABEL, TYPE_FOOTNOTE, has_payment_reason, reason_text
from ui import theme

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def fmt_pct(x: float | None, decimals: int = 1, clip: bool = True) -> str:
    """
    Format a probability / rate in [0, 1] as a percentage.

    With ``clip=True`` (default, used for MODEL SCORES) values below 1% show as
    "<1%" and above 99% as ">99%", so the app never claims false precision at
    the extremes.  Pass ``clip=False`` for statistics that must read exactly
    (shares, prevalences, flagged volume).
    """
    if x is None or pd.isna(x):
        return "n/a"
    x = float(x)
    if clip:
        if x < 0.01:
            return "<1%"
        if x > 0.99:
            return ">99%"
    return f"{x * 100:.{decimals}f}%"


def fmt_pp(x: float | None, decimals: int = 1) -> str:
    """Percentage-point difference (input is a fraction), never showing a negative zero ('-0.0 pp')."""
    if x is None or pd.isna(x):
        return "n/a"
    v = round(float(x) * 100, decimals)
    return f"{(0.0 if v == 0 else v):+.{decimals}f} pp"


def fmt_money(x: float | None, signed: bool = False) -> str:
    """-$222.50 style (sign before the currency symbol); ``signed`` adds "+" for positives."""
    if x is None or pd.isna(x):
        return "n/a"
    x = float(x)
    if x < 0:
        return f"-${abs(x):,.2f}"
    return f"{'+' if signed and x > 0 else ''}${x:,.2f}"


# ---------------------------------------------------------------------------
# Readable headers
# ---------------------------------------------------------------------------
_HEADER_OVERRIDES = {
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "pr_auc_lift": "PR-AUC Lift", "f1": "F1", "mcc": "MCC", "ece": "ECE",
    "psi": "PSI", "brier": "Brier", "n": "n", "orders": "Orders", "n_features": "Features (#)",
    "balanced_accuracy": "Balanced Acc.", "precision_at_top10pct": "Top-10% Precision",
    "roc_auc_ci_lo": "ROC-AUC CI Low", "roc_auc_ci_hi": "ROC-AUC CI High", "order_id": "Order ID",
    "risk_score": "Calibrated Score", "raw_score": "Raw Score", "late_rate": "Late Rate", "cum_gain_pct": "Cum. Gain %",
    "flagged_share_pct": "Flagged Share (%)", "sched_days": "Scheduled Days", "train_n": "Train N", "test_n": "Test N",
}
_UPPER_WORDS = {"id", "roc", "auc", "pr", "mcc", "ece", "psi", "sla", "shap", "cv"}


def readable(col: str) -> str:
    """snake_case -> Readable Header."""
    if col in _HEADER_OVERRIDES:
        return _HEADER_OVERRIDES[col]
    if " " in col or col != col.lower():
        return col
    return " ".join(w.upper() if w in _UPPER_WORDS else w.capitalize() for w in col.split("_"))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
    html = f'<div class="rto-title">{title}</div>'
    if subtitle:
        html += f'<div class="rto-sub">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)


def card_title(text: str) -> None:
    """Title inside a bordered card."""
    st.markdown(f'<div class="rto-card-title">{text}</div>', unsafe_allow_html=True)


def section_title(text: str) -> None:
    """Page-level section heading."""
    st.markdown(f'<div class="rto-section">{text}</div>', unsafe_allow_html=True)


def metric(label: str, value: str, sub: str | None = None) -> None:
    """st.metric with a plain caption underneath (no misleading delta arrow, no truncation)."""
    st.metric(label, value)
    if sub:
        st.caption(sub)


def proxy_badge() -> None:
    """Persistent proxy-target reminder (rendered once by app.py above every page)."""
    st.markdown(
        '<div class="rto-proxy"><span>⚠️</span><span><b>Proxy target:</b> late delivery &bull; '
        'DataCo public data &bull; not Indian COD.</span></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str | None = None) -> None:
    """A bordered KPI card (label, big value, optional sub-line)."""
    html = f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
    if sub:
        html += f'<div class="kpi-sub">{sub}</div>'
    with st.container(border=True):
        st.markdown(html, unsafe_allow_html=True)


def risk_badge_html(tier: str, score: float) -> str:
    return (f'<span class="rto-badge rto-badge-{tier.lower()}">{theme.TIER_ICON[tier]} {tier.upper()} RISK '
            f'({fmt_pct(score)})</span>')


RISK_SCORE_HELP = ("Isotonic calibration maps many raw scores to the same probability, so different orders can show "
                   "identical percentages.")


def risk_badge(tier: str, score: float) -> None:
    """Tier badge showing the score through ``fmt_pct`` (tier comes from the UNROUNDED score upstream)."""
    st.markdown(risk_badge_html(tier, score), unsafe_allow_html=True, help=RISK_SCORE_HELP)


_CALLOUTS = {"info": st.info, "note": st.info, "warning": st.warning, "danger": st.error, "success": st.success}
_CALLOUT_ICONS = {"info": "ℹ️", "note": "📌", "warning": "⚠️", "danger": "🚫", "success": "✅"}


def callout(kind: str, title: str, body: str = "") -> None:
    """Native Streamlit alert with a bold title; kind in info | note | warning | danger | success."""
    text = f"**{title}**" + (f"\n\n{body}" if body else "")
    _CALLOUTS[kind](text, icon=_CALLOUT_ICONS[kind])


# ---------------------------------------------------------------------------
# Payment-type proxy disclosure (every number comes from final_metrics.json -> type_audit)
# ---------------------------------------------------------------------------
def reason_footnote(reasons) -> None:
    """Show the payment-type footnote under a list of reasons (only when one of them carries the dagger)."""
    if has_payment_reason(reasons):
        st.caption(TYPE_FOOTNOTE)


def chart_footnote(labels=None, values=None, top_k: int = 6) -> None:
    """Footnote under a SHAP chart; with labels/values given, only when 'Payment type' is among the plotted bars."""
    if labels is not None:
        top = [labels[i] for i in np.argsort(np.abs(values))[::-1][:top_k]]
        if PAYMENT_LABEL not in top:
            return
    st.caption(TYPE_FOOTNOTE)


def _signed_ci(d: dict) -> str:
    return f"{d['point']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"


def payment_caption(fm: dict) -> str:
    ta = fm["type_audit"]
    a, c = ta["all_orders"]["delta_roc_auc"], ta["cleaned_orders"]["delta_roc_auc"]
    return (f"Payment type adds {a['point']:+.4f} on all orders but only {_signed_ci(c)} on cleaned orders "
            "(cancelled/fraud removed).")


def payment_ablation_box(fm: dict) -> None:
    """Info box for Robustness > Feature ablation: all-orders vs cleaned-orders payment-type deltas + equal coverage."""
    ta = fm["type_audit"]
    al, cl = ta["all_orders"], ta["cleaned_orders"]
    ea, ec = ta["equal_coverage"]["all_orders"], ta["equal_coverage"]["cleaned_orders"]
    zero = cl["delta_roc_auc"]["ci"][0] <= 0 <= cl["delta_roc_auc"]["ci"][1]
    verdict = ("The cleaned-order interval includes 0, so the gain from payment type is **consistent with a labelling artifact**: "
               "every cancelled / suspected-fraud order is a TRANSFER payment that is labelled on time."
               if zero else
               "The gain from payment type persists on cleaned orders (its interval excludes 0), so it is not explained by the "
               "cancelled / fraud labelling alone.")
    body = "\n".join([
        f"- ROC-AUC change from payment type (final model minus no-payment model): **{_signed_ci(al['delta_roc_auc'])}** on all "
        f"{al['n']:,} test orders, **{_signed_ci(cl['delta_roc_auc'])}** on cleaned orders ({cl['removed_orders']:,} cancelled/fraud removed).",
        f"- Without payment features the model scores **{al['model_b']['roc_auc']:.3f}** on all orders and "
        f"**{cl['model_b']['roc_auc']:.3f}** on cleaned orders (with them: {al['model_a']['roc_auc']:.3f} and {cl['model_a']['roc_auc']:.3f}).",
        f"- Equal coverage (flagging the same {ea['k']:,} orders, {fmt_pct(ea['flagged_share'], clip=False)}): High-tier precision "
        f"{fmt_pct(ea['model_a']['precision'], clip=False)} with payment features vs {fmt_pct(ea['model_b']['precision'], clip=False)} "
        f"without (difference {_signed_ci(ea['delta_precision'])}); on cleaned orders ({ec['k']:,} flagged) "
        f"{fmt_pct(ec['model_a']['precision'], clip=False)} vs {fmt_pct(ec['model_b']['precision'], clip=False)} "
        f"(difference {_signed_ci(ec['delta_precision'])}).",
        f"- {verdict}"])
    callout("info", "Payment type is a partly artifactual signal", body)


def show(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure with the shared template (theme=None stops Streamlit overriding it)."""
    st.plotly_chart(fig, theme=None, width="stretch", key=key, config={"displayModeBar": False})


def styled_table(
    df: pd.DataFrame,
    rename: dict[str, str] | None = None,
    formats: dict[str, str | Callable] | None = None,
    highlight: Callable[[pd.Series], str | None] | None = None,
    height: int | None = None,
    wrap: bool = False,
) -> None:
    """
    The ONLY way tables are shown in the app.

    * index always hidden
    * snake_case headers renamed to readable ones (``rename`` overrides)
    * float columns shown to 3 decimals unless ``formats`` says otherwise
    * ``highlight(row) -> css`` (e.g. "background-color: #FEE2E2") colours whole rows;
      the row passed in uses the ORIGINAL column names
    * ``wrap=True`` renders a static table whose long text cells wrap (st.dataframe cannot wrap)
    """
    d = df.copy().reset_index(drop=True)
    name_map = {c: (rename or {}).get(c, readable(str(c))) for c in d.columns}
    fmts: dict = {c: "{:.3f}" for c in d.columns if pd.api.types.is_float_dtype(d[c])}
    fmts.update({c: f for c, f in (formats or {}).items() if c in d.columns})
    # NaN / None cells: pre-format those columns as text so they read "–" (Styler.na_rep is ignored by st.dataframe)
    for c in list(fmts):
        if d[c].isna().any():
            f = fmts.pop(c)
            d[c] = d[c].map(lambda v, f=f: "–" if pd.isna(v) else (f(v) if callable(f) else f.format(v)))
    for c in d.columns:
        if d[c].dtype == object and d[c].isna().any():
            d[c] = d[c].fillna("–")
    colors = [highlight(r) if highlight else None for _, r in d.iterrows()]

    view = d.rename(columns=name_map)
    styler = view.style.format({name_map[c]: f for c, f in fmts.items()}, na_rep="–")
    if highlight:
        styler = styler.apply(lambda r: [colors[r.name] or ""] * len(r), axis=1)
    if wrap:
        # short columns (scores, ids, flags) never wrap or clip and are right-aligned; long text columns wrap
        short = [name_map[c] for c in d.columns if d[c].astype(str).str.len().max() <= 16]
        if short:
            styler = styler.set_properties(subset=short, **{"white-space": "nowrap", "min-width": "5.5rem", "text-align": "right"})
        st.table(styler.hide(axis="index"))
        return
    kwargs = {"height": height} if height else {}
    st.dataframe(styler, hide_index=True, width="stretch", **kwargs)


def signal_chip(flag: str) -> str:
    """Text chip for the per-mode signal flag."""
    return {"informative": "🟢 informative", "limited": "🟠 limited"}.get(flag, flag)


MIN_REASON_SHAP = 0.05


def filter_reasons(labels, values, val_strs, min_abs: float = MIN_REASON_SHAP, k: int = 3) -> list[str]:
    """
    Plain-English reasons that are material (|SHAP| >= min_abs) and never contradict the score: only factors that
    push in the SAME direction as the order's net SHAP (above-average risk -> reasons that raise it; below-average
    -> reasons that lower it).
    """
    values = np.asarray(values, dtype=float)
    direction = 1.0 if values.sum() > 0 else -1.0
    out = []
    for i in np.argsort(-np.abs(values)):
        if abs(values[i]) >= min_abs and np.sign(values[i]) == direction:
            out.append(reason_text(labels[i], val_strs[i], bool(values[i] > 0)))
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------
def plain_sentence(score: float, tier: str, top_driver: str, driver_raises: bool | None = None) -> str:
    """
    One tier-correct sentence.  The wording follows the TIER (computed from the unrounded score), never a
    separate 0.5 rule, so a Medium order is never described as "low probability".
    """
    p = fmt_pct(score)
    d = top_driver.lower()
    head = f"**{p} chance of arriving late**"
    if tier == "High":
        if driver_raises is False:
            return f"{head}: highest-risk tier. The largest single factor ({d}) lowers the score but is outweighed by the others."
        return f"{head}: highest-risk tier, driven primarily by **{d}**."
    if tier == "Medium":
        if driver_raises is False:
            return f"{head}: elevated risk. The largest single factor ({d}) lowers the score, but other factors keep it above the lower tier."
        return f"{head}: elevated risk, with **{d}** the largest driver."
    if driver_raises:
        return f"{head}: lower-risk tier, even though **{d}** pushes the score up."
    return f"{head}: lower-risk tier, supported by favourable **{d}**."
