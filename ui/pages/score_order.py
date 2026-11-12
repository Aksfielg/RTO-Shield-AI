"""Score an Order: live what-if scoring. Presets are ACTUAL holdout rows (incl. history features)."""

from __future__ import annotations

import streamlit as st

from ui import charts
from ui import components as C
from ui import data

import predict

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _clamp(v, lo, hi):
    return min(max(v, lo), hi)


def _pick(options: list, wanted):
    return options.index(wanted) if wanted in options else 0


def default_values(base: dict, opts: dict) -> dict:
    """Widget defaults for a (possibly empty) real order; the rounding/clamping here is undone for untouched fields."""
    return {
        "Shipping_Mode": base.get("Shipping_Mode", "Standard Class" if "Standard Class" in opts["modes"] else opts["modes"][0]),
        "Type": base.get("Type", opts["types"][0]),
        "order_value": round(_clamp(base.get("order_value", 150.0), *opts["value_range"]), 2),
        "n_items": int(_clamp(base.get("n_items", 2), *opts["items_range"])),
        "avg_discount_rate": round(_clamp(base.get("avg_discount_rate", 0.05), 0.0, 0.5), 2),
        "Market": base.get("Market", opts["markets"][0]),
        "Department_Name": base.get("Department_Name", opts["departments"][0]),
        "Customer_Segment": base.get("Customer_Segment", opts["segments"][0]),
        "order_hour": int(base.get("order_hour", 14)),
        "order_dow": int(base.get("order_dow", 2)),
        "order_month": int(base.get("order_month", 6)),
    }


def build_order(base: dict, vals: dict, defaults: dict, mode_days: dict) -> dict:
    """
    Merge the form values over a real order.  A field the user did not change keeps the real order's EXACT value, so an
    untouched preset is the actual test row (all history features included) and scores exactly like the Real Test Orders
    page.  Derived fields are recomputed only when their inputs changed.
    """
    out = dict(base)
    for key, v in vals.items():
        out[key] = base[key] if (key in base and defaults.get(key) == v) else v
    if "Days_for_shipment_scheduled" not in base or out["Shipping_Mode"] != base.get("Shipping_Mode"):
        out["Days_for_shipment_scheduled"] = mode_days.get(out["Shipping_Mode"], 4)
    same_value = "value_per_item" in base and out["order_value"] == base["order_value"] and out["n_items"] == base["n_items"]
    if not same_value:
        out["value_per_item"] = out["order_value"] / max(out["n_items"], 1)
    if "is_cash" not in base or out["Type"] != base.get("Type"):
        out["is_cash"] = int(out["Type"] == "CASH")
    if "is_weekend" not in base or out["order_dow"] != base.get("order_dow"):
        out["is_weekend"] = int(out["order_dow"] in (5, 6))
    if "is_night_order" not in base or out["order_hour"] != base.get("order_hour"):
        out["is_night_order"] = int(out["order_hour"] >= 22 or out["order_hour"] < 6)
    if "high_discount_flag" not in base or out["avg_discount_rate"] != base.get("avg_discount_rate"):
        out["high_discount_flag"] = int(out["avg_discount_rate"] > 0.15)
    out.setdefault("n_distinct_categories", 1)
    return out


def information_level(mode: str, per_mode: list[dict]) -> tuple[str, str, str]:
    """(callout kind, title, body) describing how much the model can say about this shipping mode."""
    row = next(r for r in per_mode if r["mode"] == mode)
    if row["flag"] == "limited":
        lim = [r["roc_auc"] for r in per_mode if r["flag"] == "limited"]
        return ("warning", "Information level: Limited",
                f"Model cannot rank orders within this shipping mode (AUC ~{min(lim):.2f}-{max(lim):.2f}); the score mostly "
                f"reflects the mode's base rate ({C.fmt_pct(row['prevalence'], clip=False)} late).")
    return ("success", "Information level: Informative",
            f"Within {mode} the model ranks orders well (ROC-AUC {row['roc_auc']:.2f}, 95% CI "
            f"{row['roc_auc_ci_lo']:.2f} to {row['roc_auc_ci_hi']:.2f}).")


def render() -> None:
    fm, thr, opts, reps = data.fm(), data.thr(), data.option_lists(), data.representative_orders()
    low_cut, high_cut = thr["tier_cutoffs"]["low_max"], thr["tier_cutoffs"]["high_min"]
    C.page_header("Score an Order", "Simulate pre-dispatch risk evaluation and explainable SHAP drivers for a single order.")

    labels = {"Custom input": None}
    for tier in ["High", "Medium", "Low"]:
        if tier in reps:
            r = reps[tier]
            labels[f"Real {tier.lower()}-risk order #{r['Order Id']} ({r['Shipping_Mode']})"] = tier
    choice = st.radio("Start from an actual holdout order (by order ID), or enter your own:", list(labels), horizontal=True)
    tier_pick = labels[choice]
    base = reps[tier_pick] if tier_pick else {}
    ck = tier_pick or "custom"
    d = default_values(base, opts)

    col_form, col_res = st.columns([1.05, 1.25], gap="large")
    with col_form, st.container(border=True):
        C.card_title("Order parameters")
        mode = st.selectbox("Shipping mode", opts["modes"], index=_pick(opts["modes"], d["Shipping_Mode"]), key=f"mode_{ck}")
        st.caption(f"Contracted shipping days: **{opts['mode_days'][mode]}** (fixed by shipping mode in the data)")
        a, b = st.columns(2)
        with a:
            ptype = st.selectbox("Payment type", opts["types"], index=_pick(opts["types"], d["Type"]), key=f"type_{ck}")
            value = st.number_input("Order value ($)", min_value=float(opts["value_range"][0]), max_value=float(opts["value_range"][1]),
                                    value=float(d["order_value"]), step=10.0, key=f"value_{ck}")
            items = st.number_input("Number of items", min_value=opts["items_range"][0], max_value=opts["items_range"][1],
                                    value=d["n_items"], step=1, key=f"items_{ck}")
        with b:
            disc = st.slider("Discount rate", 0.0, 0.5, float(d["avg_discount_rate"]), 0.01, key=f"disc_{ck}")
            market = st.selectbox("Market", opts["markets"], index=_pick(opts["markets"], d["Market"]), key=f"market_{ck}")
            dept = st.selectbox("Department", opts["departments"], index=_pick(opts["departments"], d["Department_Name"]), key=f"dept_{ck}")
        c, e = st.columns(2)
        with c:
            seg = st.selectbox("Customer segment", opts["segments"], index=_pick(opts["segments"], d["Customer_Segment"]), key=f"seg_{ck}")
            hour = st.slider("Order hour (0-23)", 0, 23, d["order_hour"], key=f"hour_{ck}")
        with e:
            dow_name = st.selectbox("Day of week", DOW_NAMES, index=d["order_dow"], key=f"dow_{ck}")
            month = st.slider("Order month", 1, 12, d["order_month"], key=f"month_{ck}")
        if tier_pick:
            st.caption(f"Loaded the actual holdout row for order #{base['Order Id']}, including its customer and country history features. "
                       "Untouched, it scores exactly like the Real Test Orders page.")
        else:
            st.caption("Note: customer and country history features (previous orders, previous late rates, days since last order), "
                       "region, country and product category are not entered here, so they are **imputed** by the preprocessor.")

    vals = {"Shipping_Mode": mode, "Type": ptype, "order_value": value, "n_items": int(items), "avg_discount_rate": disc,
            "Market": market, "Department_Name": dept, "Customer_Segment": seg, "order_hour": hour,
            "order_dow": DOW_NAMES.index(dow_name), "order_month": month}
    order = build_order(base, vals, d, opts["mode_days"])

    result = predict.score_order(order)          # tier is computed from the UNROUNDED score
    score, tier = result["risk_score"], result["tier"]
    ex = data.explain_order(order)
    reasons = C.filter_reasons(ex["labels"], ex["values"], ex["val_strs"])
    kind, title, body = information_level(mode, fm["per_mode"])

    with col_res:
        with st.container(border=True):
            h1, h2 = st.columns([1.2, 1])
            with h1:
                C.card_title("Risk assessment")
            with h2:
                C.risk_badge(tier, score)
            C.show(charts.gauge(score, low_cut, high_cut), key="score_gauge")
            st.markdown(C.plain_sentence(score, tier, ex["top_driver"], ex["driver_raises"]))
            st.caption(thr["tier_descriptions"][tier]["description"])
            st.markdown(f"**Flag:** {'🚩 FLAGGED (High tier)' if tier == 'High' else 'Not flagged'}  \n"
                        f"<span style='color:#64748B;font-size:.85rem'>Only the High tier is flagged.</span>", unsafe_allow_html=True)
            C.callout(kind, title, body)
            st.markdown(f"**Top reasons** (|SHAP| ≥ {C.MIN_REASON_SHAP}, same direction as the score)")
            if reasons:
                for r in reasons:
                    st.markdown(f"{'🔺' if 'raises' in r else '🔻'} {r}")
            else:
                st.caption(f"No single factor reaches |SHAP| ≥ {C.MIN_REASON_SHAP} in the direction of this score.")
            C.show(charts.grouped_shap_bar(ex["labels"], ex["values"], ex["val_strs"]), key="score_shap_bar")
        if tier_pick:
            actual = "late" if base.get("_actual") == 1 else "on time"
            st.caption(f"This real holdout order was actually delivered **{actual}** (shown for reference, never used for scoring).")
        with st.expander("Technical SHAP waterfall"):
            C.show(charts.shap_waterfall(ex["labels"], ex["values"], ex["val_strs"], ex["base_value"]), key="score_waterfall")
            st.caption("Additive shift from the average log-odds to this order's final log-odds (2 decimals).")
