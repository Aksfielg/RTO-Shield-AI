"""Real Test Orders: inspect unseen holdout orders; cascading filters, confusion summary, CSV export."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import TARGET_COL
from ui import charts
from ui import components as C
from ui import data

import predict

CHIP = {"success": "🟢", "warning": "🟠", "danger": "🔴"}


def verdict(flagged: bool, actual: int) -> tuple[str, str, str]:
    """(kind, chip label, explanation) for a flag decision (flagged = High tier) vs the actual outcome."""
    if flagged and actual == 1:
        return "success", "Correct", "true positive: a late order was caught"
    if not flagged and actual == 0:
        return "success", "Correct", "true negative: an on-time order was cleared"
    if not flagged and actual == 1:
        return "warning", "Missed", "false negative: a late order slipped through"
    return "danger", "False alarm", "false positive: an on-time order was flagged (unnecessary friction)"


def confusion(flagged, y) -> dict:
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}


def reasons_text(row: dict) -> str:
    ex = data.explain_order(row)
    return "; ".join(C.filter_reasons(ex["labels"], ex["values"], ex["val_strs"])) or "No material single driver"


def render(test_df: pd.DataFrame | None = None) -> None:
    fm, thr = data.fm(), data.thr()
    if test_df is None:
        _, test_df = data.datasets()
    sc = data.test_scores()
    high_min = thr["tier_cutoffs"]["high_min"]
    sp = fm["splits"]["test"]
    C.page_header("Real Test Orders",
                  f"Inspect unseen orders from the holdout partition ({sp['date_min']} to {sp['date_max']}) with model evaluation badges.")

    demo = data.demo_scores()
    with st.container(border=True):
        C.card_title("1. Benchmark demo orders (real holdout cases)")
        if demo is not None:
            rows = []
            for _, d in demo.iterrows():
                match = test_df[test_df["Order Id"] == d["order_id"]]
                top = reasons_text(match.iloc[0].to_dict()) if len(match) else ""
                rows.append({"order_id": int(d["order_id"]), "score": d["risk_score"], "tier": d["tier"],
                             "flagged": "Yes" if d["flagged"] else "No", "actual": d["actual"], "shipping_mode": d["shipping_mode"],
                             "payment": d["payment_type"], "region": d["region"], "top_reasons": top})
            C.styled_table(pd.DataFrame(rows), rename={"score": "Risk Score", "actual": "Actual Delivery", "top_reasons": "Top reasons"},
                           formats={"score": lambda v: C.fmt_pct(v)}, wrap=True)
        else:
            st.info("Run src/07_predict_demo.py to generate the benchmark table.")

    with st.container(border=True):
        C.card_title("2. Interactive order inspection")
        f1, f2 = st.columns(2)
        with f1:
            mode_flt = st.selectbox("Shipping mode", ["All"] + sorted(test_df["Shipping_Mode"].dropna().unique().tolist()))
        with f2:
            outcome_flt = st.selectbox("Actual outcome", ["All", "Late only", "On-time only"])
        pool = test_df
        if mode_flt != "All":
            pool = pool[pool["Shipping_Mode"] == mode_flt]
        if outcome_flt != "All":
            pool = pool[pool[TARGET_COL] == (1 if outcome_flt == "Late only" else 0)]
        if pool.empty:
            st.warning("No orders match these filters. Change the shipping mode or outcome.")
            return

        # confusion summary for the CURRENT filter (flagged = High tier)
        pos = pool.index.values
        flagged_all, y_all = sc["cal"][pos] >= high_min, sc["y"][pos]
        cm = confusion(flagged_all, y_all)
        st.markdown(f"**Current filter: {len(pool):,} orders** (flagged = High tier)")
        cmdf = pd.DataFrame({"": ["Actually late", "Actually on time"], "Flagged": [cm["tp"], cm["fp"]], "Not flagged": [cm["fn"], cm["tn"]]})
        a, b = st.columns([1, 1.6])
        with a:
            C.styled_table(cmdf, formats={"Flagged": "{:,}", "Not flagged": "{:,}"})
        with b:
            prec = "n/a" if cm["precision"] is None else C.fmt_pct(cm["precision"], clip=False)
            rec = "n/a" if cm["recall"] is None else C.fmt_pct(cm["recall"], clip=False)
            st.markdown(f"- **Caught** {cm['tp']:,} late orders; **missed** {cm['fn']:,}\n"
                        f"- **False alarms:** {cm['fp']:,}\n"
                        f"- Precision {prec}, recall {rec} within this filter")

        export = pool[["Order Id", "order_date", "Shipping_Mode", "Type", "Market", "order_value", TARGET_COL]].copy()
        export["risk_score"], export["flagged_high_tier"] = sc["cal"][pos], flagged_all
        export["actual"] = export.pop(TARGET_COL).map({1: "Late", 0: "On-time"})
        st.download_button("📥 Download this filter as CSV", export.to_csv(index=False).encode("utf-8"),
                           "rto_shield_filtered_orders.csv", "text/csv")

        # selection resets to the first matching order whenever a filter changes (the key depends on the filters)
        options = [int(i) for i in pool["Order Id"].head(200)]
        oid = st.selectbox(f"Order (first {len(options)} matching)", options, key=f"order_{mode_flt}_{outcome_flt}")

        row = test_df[test_df["Order Id"] == oid].iloc[0].to_dict()
        actual = int(row[TARGET_COL])
        res = predict.score_order(row)
        kind, chip, expl = verdict(res["flagged"], actual)

        st.divider()
        h1, h2 = st.columns([1, 1.4])
        with h1:
            st.markdown(f"#### Order #{oid}")
        with h2:
            C.risk_badge(res["tier"], res["risk_score"])
        st.markdown(f"**{CHIP[kind]} {chip}**: {expl}")

        m1, m2, m3 = st.columns(3)
        with m1:
            C.metric("Model risk score", C.fmt_pct(res["risk_score"]), f"Tier: {res['tier']}")
        with m2:
            C.metric("Flag status", "Flagged" if res["flagged"] else "Not flagged", "High tier: priority review" if res["flagged"] else "Only the High tier is flagged")
        with m3:
            C.metric("Actual delivery", "Late" if actual == 1 else "On time")
        st.caption(res["tier_description"])

        left, right = st.columns([1, 1.3], gap="medium")
        with left:
            ex = data.explain_order(row)
            st.markdown("**Top reasons** (|SHAP| ≥ 0.05, same direction as the score)")
            reasons = C.filter_reasons(ex["labels"], ex["values"], ex["val_strs"])
            for r in reasons or ["No single factor reaches the threshold in this direction."]:
                st.markdown(f"{'🔺' if 'raises' in r else '🔻'} {r}")
            st.markdown("**Key order attributes**")
            attrs = pd.DataFrame({"Attribute": ["Shipping mode", "Scheduled days", "Payment method", "Order value", "Items",
                                                "Placement hour", "Market", "Customer segment"],
                                  "Value": [row.get("Shipping_Mode"), row.get("Days_for_shipment_scheduled"), row.get("Type"),
                                            C.fmt_money(row.get("order_value")), row.get("n_items"),
                                            f"{int(row.get('order_hour', 0)):02d}:00", row.get("Market"), row.get("Customer_Segment")]})
            attrs["Value"] = attrs["Value"].astype(str)
            C.styled_table(attrs)
        with right:
            C.show(charts.shap_waterfall(ex["labels"], ex["values"], ex["val_strs"], ex["base_value"]), key="real_waterfall")
