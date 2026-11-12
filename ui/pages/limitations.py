"""Limitations & governance (all figures read from final_metrics.json)."""

from __future__ import annotations

import streamlit as st

from ui import components as C
from ui import data


def render() -> None:
    fm = data.fm()
    C.page_header("Limitations & Governance", "Constraints, dataset context and ethical guidelines (see MODEL_CARD.md).")
    std, sec = data.mode_stats("Standard Class"), data.mode_stats("Second Class")
    top = max(fm["score_clusters"]["top_decile"]["rows"], key=lambda r: r["top10_share_pct"])
    ver, noon = fm["verification"], fm["same_day_noon_rule"]
    pc = lambda v: C.fmt_pct(v, clip=False)

    with st.container(border=True):
        C.card_title("1. The proxy-target reality")
        st.write("`Late_delivery_risk` is an **upstream fulfilment proxy**, NOT Return-To-Origin (RTO).")
        st.markdown(
            "- In Indian D2C e-commerce, RTO occurs when a buyer refuses delivery, cancels at the doorstep or commits COD fraud.\n"
            "- In the DataCo dataset the label reflects whether delivery duration exceeded the contracted carrier SLA.\n"
            "- Delays can raise doorstep-refusal propensity, but this model predicts delay, not refusal.")

    with st.container(border=True):
        C.card_title("2. Near rule-based label construction")
        st.markdown(
            f"- Same Day orders placed from 12:00 are late {pc(noon['late_rate_from_noon'])} of the time vs "
            f"{pc(noon['late_rate_before_noon'])} before noon ({noon['orders']:,} Same Day orders).\n"
            f"- The top decile of scores is **{top['top10_share_pct']:.2f}% {top['shipping_mode']}** orders.\n"
            f"- {pc(fm['test']['precision_at_top10pct'])} precision in the top decile is an artefact of the near rule-based labels, "
            "not model insight.")

    with st.container(border=True):
        C.card_title("3. Payment type is a label artifact")
        st.write(
            f"All **{ver['cancel_fraud_orders']:,}** CANCELED / SUSPECTED_FRAUD orders are TRANSFER payments "
            f"({pc(ver['cancel_fraud_share_transfer'])} of TRANSFER orders vs {pc(ver['cancel_fraud_share_other_types'])} of other types) "
            f"and every one is labelled on time (late rate {pc(ver['late_rate_canceled'])} / {pc(ver['late_rate_suspected_fraud'])} vs "
            f"{pc(ver['late_rate_other_statuses'])} for other statuses). Controlling for shipping mode, the TRANSFER late-rate difference is "
            f"{C.fmt_pp(ver['transfer_gap_within_mode_all'])}, and {C.fmt_pp(ver['transfer_gap_within_mode_clean'])} once those orders "
            "are removed. TRANSFER is not a real payment-risk signal.")

    with st.container(border=True):
        C.card_title("4. Standard and Second Class blind spot")
        st.write(
            f"Standard Class ({pc(std['share'])}) and Second Class ({pc(sec['share'])}) together are **{pc(std['share'] + sec['share'])}** of "
            f"test volume. ROC-AUC is **{std['roc_auc']:.4f}** for Standard Class (95% CI {std['roc_auc_ci_lo']:.3f} to "
            f"{std['roc_auc_ci_hi']:.3f}) and **{sec['roc_auc']:.4f}** for Second Class ({sec['roc_auc_ci_lo']:.3f} to {sec['roc_auc_ci_hi']:.3f}), and "
            f"{pc(fm['score_clusters']['largest_standard_cluster_share_of_standard'])} of Standard Class orders share one calibrated score. "
            "Checkout-time signals cannot predict unobservable middle- and last-mile carrier delays.")

    with st.container(border=True):
        C.card_title("5. Geographic scope & public-data constraints")
        st.markdown(
            "- DataCo comes from Western cross-border retail (North America, LATAM, Europe).\n"
            "- It does not capture PIN-code delivery tiers, NDR/IVR confirmation or doorstep COD refusal.\n"
            "- Do not deploy without retraining on real courier and return telemetry.")

    with st.container(border=True):
        C.card_title("6. Advisory governance & ethical guardrails")
        st.markdown(
            "- **Human in the loop:** decision support only; never autonomous order cancellation or payment decline.\n"
            "- **Fairness:** monitor regional and customer-history late rates so they do not compound regional delivery disadvantages.")
