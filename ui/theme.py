"""
ui/theme.py
===========
Single source of truth for the dashboard's look: colour palette, CSS and the
shared Plotly template.  Every chart in the app uses the "rto" template and
this palette:

    low risk    green  #10B981
    medium risk amber  #F59E0B
    high risk   red    #EF4444
    neutral     indigo #4F46E5 (+ slate greys)
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INDIGO = "#4F46E5"
INDIGO_DARK = "#4338CA"
INDIGO_LIGHT = "#C7D2FE"
GREEN = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"
SLATE = "#94A3B8"
SLATE_LIGHT = "#CBD5E1"
INK = "#0F172A"
TEXT = "#1E293B"
MUTED = "#64748B"
GRID = "#F1F5F9"

TIER_COLORS = {"Low": GREEN, "Medium": AMBER, "High": RED}
TIER_BG = {"Low": "#D1FAE5", "Medium": "#FEF3C7", "High": "#FEE2E2"}
TIER_ICON = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

# neutral multi-series colourway (indigo first, then greys / light indigo)
COLORWAY = [INDIGO, SLATE, INDIGO_LIGHT, SLATE_LIGHT, INDIGO_DARK, MUTED]

# ---------------------------------------------------------------------------
# Shared Plotly template
# ---------------------------------------------------------------------------
_axis = dict(gridcolor=GRID, linecolor=SLATE_LIGHT, zerolinecolor=SLATE_LIGHT, ticks="outside",
             tickcolor=SLATE_LIGHT, title=dict(font=dict(size=12, color=MUTED)))

RTO_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, size=12, color=TEXT),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=COLORWAY,
        title=dict(font=dict(size=14, color=TEXT), x=0.0, xanchor="left", y=0.97, yanchor="top"),
        xaxis=_axis,
        yaxis=_axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=10, r=10, t=62, b=30),
        hoverlabel=dict(font=dict(family=FONT)),
    )
)
pio.templates["rto"] = RTO_TEMPLATE
pio.templates.default = "rto"


def style(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    """Apply the shared template + standard sizing to any figure."""
    fig.update_layout(template="rto", height=height)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if title is not None:
        fig.update_layout(title=dict(text=f"<b>{title}</b>"))
    return fig


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = f"""
<style>
    html, body, .stApp {{ font-family: {FONT}; color: {TEXT}; }}
    .block-container {{ padding-top: 4rem; padding-bottom: 3rem; max-width: 1280px; }}

    /* st.container(border=True) is the only card primitive in this app (radius/border come from .streamlit/config.toml) */

    /* titles inside cards / sections (smaller than st.subheader) */
    .rto-card-title {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 0 0 .5rem 0; }}
    .rto-section {{ font-size: 1.35rem; font-weight: 700; color: {TEXT}; margin: 1rem 0 .5rem 0; }}

    /* page header */
    .rto-title {{ font-size: 2.0rem; font-weight: 800; color: {INK}; letter-spacing: -0.02em; margin: 0 0 .15rem 0; }}
    .rto-sub {{ font-size: 1.05rem; color: #475569; margin: 0 0 1.1rem 0; }}

    /* persistent proxy-target badge */
    .rto-proxy {{
        display: inline-flex; align-items: center; gap: 8px; background: #FFFBEB; border: 1px solid #FDE68A;
        color: #92400E; padding: 6px 14px; border-radius: 8px; font-size: .84rem; font-weight: 500; margin-bottom: 1rem;
    }}

    /* KPI card text */
    .kpi-label {{ font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; color: {MUTED}; }}
    .kpi-value {{ font-size: 1.85rem; font-weight: 700; color: {INK}; line-height: 1.25; }}
    .kpi-sub {{ font-size: .8rem; font-weight: 500; color: {INDIGO}; }}

    /* risk badges */
    .rto-badge {{
        display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 9999px;
        font-weight: 700; font-size: .85rem; letter-spacing: .02em; border: 1px solid transparent;
    }}
    .rto-badge-low {{ background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }}
    .rto-badge-medium {{ background: #FFFBEB; color: #92400E; border-color: #FDE68A; }}
    .rto-badge-high {{ background: #FEF2F2; color: #991B1B; border-color: #FECACA; }}
    .rto-badge-neutral {{ background: #EEF2FF; color: #3730A3; border-color: {INDIGO_LIGHT}; }}

    /* primary buttons */
    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {{
        background-color: {INDIGO}; border: none; border-radius: 8px; font-weight: 600;
    }}
    div.stButton > button[kind="primary"]:hover {{ background-color: {INDIGO_DARK}; }}

    /* sidebar */
    section[data-testid="stSidebar"] {{ background: #F8FAFC; border-right: 1px solid #E2E8F0; }}
    .rto-side-brand {{ font-size: 1.25rem; font-weight: 800; color: {INK}; margin: 0; }}
    .rto-side-meta {{ font-size: .78rem; color: {MUTED}; line-height: 1.45; margin: .2rem 0 0 0; }}
    .rto-side-cite {{ font-size: .72rem; color: {MUTED}; line-height: 1.4; }}
</style>
"""


def inject_css() -> None:
    """Inject the global CSS (call once per script run, after set_page_config)."""
    st.markdown(CSS, unsafe_allow_html=True)
