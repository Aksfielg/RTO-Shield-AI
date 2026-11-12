"""Home: headline KPIs, what works / what does not / next, data timeline, and critical context."""

from __future__ import annotations

import streamlit as st

from ui import charts
from ui import components as C
from ui import data


def render() -> None:
    fm = data.fm()
    t, ds = fm["test"], fm["dataset"]
    high = next(p for p in fm["policies"] if p["policy"].startswith("High tier"))
    std, sec = data.mode_stats("Standard Class"), data.mode_stats("Second Class")
    fc, sd = data.mode_stats("First Class"), data.mode_stats("Same Day")
    limited_share = std["share"] + sec["share"]

    C.page_header("RTO Shield AI - Model 1", "Pre-dispatch delivery-failure risk scoring (public-data proxy)")

    c1, c2, c3 = st.columns(3)
    with c1:
        C.kpi_card("ROC-AUC (95% CI)", f"{t['roc_auc']:.3f}", f"[{t['roc_auc_ci'][0]:.3f}, {t['roc_auc_ci'][1]:.3f}] on {t['n']:,} holdout orders")
    with c2:
        C.kpi_card("High-tier precision", C.fmt_pct(high["precision"], clip=False),
                   f"flags {C.fmt_pct(high['flagged_share'], clip=False)} of orders, catches {C.fmt_pct(high['recall'], clip=False)} of late ones")
    with c3:
        C.kpi_card("Standard + Second Class AUC", f"{std['roc_auc']:.2f} / {sec['roc_auc']:.2f}",
                   f"near chance on {C.fmt_pct(limited_share, clip=False)} of volume")

    w1, w2, w3 = st.columns(3)
    with w1, st.container(border=True, height=330):
        C.card_title("What works")
        st.markdown(
            f"- Ranks **Same Day** ({sd['roc_auc']:.2f}) and **First Class** ({fc['roc_auc']:.2f}) orders well\n"
            f"- Calibrated: ECE {t['ece_raw']:.3f} to {t['ece_calibrated']:.3f}\n"
            f"- Stable over time: backtest AUC {fm['backtest']['roc_auc_mean']:.3f} ± {fm['backtest']['roc_auc_std']:.3f}\n"
            f"- High tier is {C.fmt_pct(high['precision'], clip=False)} late vs {C.fmt_pct(t['base_rate'], clip=False)} overall")
    with w2, st.container(border=True, height=330):
        C.card_title("What does not")
        st.markdown(
            f"- **Standard / Second Class** ({C.fmt_pct(limited_share, clip=False)} of volume): cannot rank orders within the mode\n"
            "- **Near rule-based labels:** much of the signal is the dispatch rule, not courier behaviour\n"
            "- **Payment type** looks predictive only because of a labelling artifact\n"
            "- Not validated on Indian COD / D2C data")
    with w3, st.container(border=True, height=330):
        C.card_title("Next: Model 2 (uplift)")
        st.markdown(
            "- Estimate the **causal effect** of an intervention (call, address check, priority dispatch), not just risk\n"
            "- Train on real courier / NDR / return outcomes\n"
            "- Target orders where intervention changes the outcome\n"
            "- Replace the user-set success rates in the Impact Simulator with measured ones")

    with st.container(border=True):
        C.card_title("Data and chronological split")
        C.show(charts.split_timeline(fm["splits"]), key="home_timeline")
        st.caption(f"{ds['n_orders']:,} orders from {ds['date_min']} to {ds['date_max']}. Models are trained on the earliest 70%, "
                   "tuned and calibrated on the next 15%, and every reported number comes from the final 15% (holdout).")

    ver, noon = fm["verification"], fm["same_day_noon_rule"]
    C.section_title("Critical context & bounds")
    C.callout("warning", "What this model does NOT show", "\n".join([
        "- **Not Indian doorstep RTO:** the target is `Late_delivery_risk` (SLA delay), not buyer refusal, fake door delivery or COD default.",
        f"- **Near rule-based labels:** Same Day orders placed from 12:00 are late {C.fmt_pct(noon['late_rate_from_noon'], clip=False)} "
        f"of the time vs {C.fmt_pct(noon['late_rate_before_noon'], clip=False)} before noon.",
        f"- **Payment type is a label artifact:** all {ver['cancel_fraud_orders']:,} CANCELED / SUSPECTED_FRAUD orders are TRANSFER payments "
        "and all are labelled on-time.",
        "- **Advisory only:** built for human fulfilment prioritisation, never for autonomous order cancellation.",
    ]))
