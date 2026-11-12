"""Explainability: SHAP-computed grouped importance, beeswarm and continuous dependence (all Plotly)."""

from __future__ import annotations

import streamlit as st

from ui import charts
from ui import components as C
from ui import data

DEPENDENCE = [
    ("order_hour", "Order hour (0-23)", "Sharp step for Same Day orders placed from 12:00."),
    ("Days_for_shipment_scheduled", "Scheduled shipping days", "Contracted shipping window set by the shipping mode."),
    ("order_value", "Order value ($)", "Total basket value."),
    ("value_per_item", "Value per item ($)", "Basket density: value per unit."),
    ("customer_prev_late_rate", "Customer historical late rate", "Share of the customer's earlier (already known) orders that were late."),
]


def render() -> None:
    fm = data.fm()
    C.page_header("Model Explainability", "Global feature attributions and continuous SHAP dependence.")
    C.callout("info", "How to read this",
              "SHAP splits each order's risk into contributions from its features. A **positive** value pushes the order toward "
              "\"late\", a **negative** value toward \"on time\"; the size is the strength (in log-odds). Averaging the absolute "
              "values over many orders shows which factors the model relies on most. SHAP describes what the **model** uses, not what "
              "causes lateness: a strong factor can still be an artefact of how the labels were built.")

    tab_glob, tab_bee, tab_dep = st.tabs(["🌐 Global importance", "🐝 Beeswarm", "📈 SHAP dependence (5 continuous features)"])
    grouped = data.grouped_importance()
    ship = sum(g["share"] for g in grouped if g["factor"] in ("Shipping mode", "Scheduled shipping days"))
    n_sample = fm["shap"]["sample_n"]

    with tab_glob:
        with st.container(border=True):
            C.card_title("Global importance by business factor")
            st.write(f"One-hot columns are grouped into their parent features. Shipping mode and scheduled days together account for "
                     f"**{C.fmt_pct(ship, clip=False)}** of total attribution (computed from SHAP on a {n_sample}-order test sample).")
            a, b = st.columns(2)
            with a:
                C.show(charts.importance_grouped(grouped[:8]), key="exp_grouped")
            with b:
                C.show(charts.importance_encoded(fm["shap"]["top_features"][:8]), key="exp_encoded")

    with tab_bee, st.container(border=True):
        C.card_title("Beeswarm distribution")
        st.write("Each dot is an order; the horizontal position is its SHAP impact and the shade shows whether the feature value was low or high.")
        C.show(charts.beeswarm(data.shap_sample()), key="exp_bee")

    with tab_dep, st.container(border=True):
        C.card_title("SHAP dependence on key continuous features")
        sample = data.shap_sample()
        tabs = st.tabs([lbl for _, lbl, _ in DEPENDENCE])
        for tab, (col, label, note) in zip(tabs, DEPENDENCE):
            with tab:
                st.markdown(f"**{label}**: {note}")
                j = sample["feat_names"].index(f"num__{col}")
                C.show(charts.dependence(sample["rows"][col].values, sample["sv"][:, j], label), key=f"exp_dep_{col}")
                st.caption("X axis: the order's actual (unscaled) value. Y axis: that value's SHAP contribution to the risk score.")
