"""
app.py
======
RTO Shield AI - Model 1: Streamlit entry point.

Structure
---------
  app.py                 page config, grouped navigation, sidebar chrome (this file)
  ui/theme.py            palette, CSS, shared Plotly template
  ui/components.py       kpi_card, risk_badge, callout, styled_table, fmt_pct, fmt_money, plain_sentence
  ui/charts.py           every chart (Plotly only)
  ui/data.py             cached loaders (st.cache_resource / st.cache_data)
  ui/pages/*.py          one module per page, each exposing render()

Every number on every page comes from outputs/final_metrics.json, models/thresholds.json or the real
holdout orders; nothing is typed into the UI code.

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

# set_page_config must be the first Streamlit call
st.set_page_config(page_title="RTO Shield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

from ui import components as C  # noqa: E402
from ui import data, theme  # noqa: E402
from ui.pages import (  # noqa: E402
    batch, explainability, home, impact, limitations, methodology, performance, real_orders, robustness, score_order,
)

theme.inject_css()

PAGES = {
    "Overview": [
        st.Page(home.render, title="Home", icon="🏠", url_path="home", default=True),
    ],
    "Decisioning & Operations": [
        st.Page(score_order.render, title="Score an Order", icon="🛡️", url_path="score-order"),
        st.Page(real_orders.render, title="Real Test Orders", icon="📦", url_path="real-orders"),
        st.Page(impact.render, title="Impact Simulator", icon="💡", url_path="impact"),
        st.Page(batch.render, title="Batch Scoring", icon="📁", url_path="batch"),
    ],
    "Model Evaluation & Governance": [
        st.Page(performance.render, title="Performance", icon="📊", url_path="performance"),
        st.Page(explainability.render, title="Explainability", icon="🔍", url_path="explainability"),
        st.Page(robustness.render, title="Robustness", icon="⚙️", url_path="robustness"),
        st.Page(limitations.render, title="Limitations", icon="⚠️", url_path="limitations"),
        st.Page(methodology.render, title="Methodology", icon="🧭", url_path="methodology"),
    ],
}

nav = st.navigation(PAGES)

# ---------------------------------------------------------------------------
# Sidebar chrome: model version, data source, citation
# ---------------------------------------------------------------------------
_fm = data.fm()
with st.sidebar:
    st.divider()
    st.markdown(
        '<p class="rto-side-brand">🛡️ RTO Shield AI</p>'
        f'<p class="rto-side-meta"><b>Model 1</b> &middot; {_fm["model"]["type"]}<br>'
        f'Metrics generated {_fm["meta"]["generated"][:10]}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="rto-side-meta"><b>Data:</b> DataCo Smart Supply Chain (public) &middot; '
        f'{_fm["dataset"]["n_orders"]:,} orders &middot; {_fm["dataset"]["date_min"]} to {_fm["dataset"]["date_max"]}</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<p class="rto-side-cite">Constante, F., Silva, F., &amp; Pereira, A. (2019). <i>DataCo Smart Supply Chain for Big Data '
        'Analysis</i>. Mendeley Data, V5. doi:10.17632/8gx2fvg2k6.5 &middot; CC BY 4.0</p>',
        unsafe_allow_html=True,
    )

C.proxy_badge()   # persistent on every page
data.start_warmup()   # load the model in the background while the first page renders
nav.run()
