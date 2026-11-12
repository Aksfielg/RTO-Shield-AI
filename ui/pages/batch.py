"""Batch Scoring: upload (or sample) a batch, score it, triage it by tier and shipping mode, audit it if labelled."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
from ui import charts
from ui import components as C
from ui import data

import predict

COL_RENAMES = {
    "Shipping Mode": "Shipping_Mode",
    "Days for shipment (scheduled)": "Days_for_shipment_scheduled",
    "Customer Segment": "Customer_Segment",
    "Department Name": "Department_Name",
    "Order Region": "Order_Region",
    "Order Country": "Order_Country",
    "order date (DateOrders)": "order_date",
}
OUTCOME_CANDIDATES = ["Late_delivery_risk", "late_delivery_risk", "actual", "outcome", "target", "is_late", "late"]
REQUIRED = ["Shipping_Mode", "Type", "order_value", "n_items"]          # without these the score is meaningless
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
OUTCOME_LABEL = {1: "Late", 0: "On-time"}


def harmonize(df: pd.DataFrame, mode_days: dict) -> tuple[pd.DataFrame, str | None]:
    """Rename known aliases, derive helper fields and detect the outcome column."""
    d = df.copy()
    for old, new in COL_RENAMES.items():
        if old in d.columns and new not in d.columns:
            d[new] = d[old]
    if "Days_for_shipment_scheduled" not in d.columns and "Shipping_Mode" in d.columns:
        d["Days_for_shipment_scheduled"] = d["Shipping_Mode"].map(mode_days)
    if "value_per_item" not in d.columns and {"order_value", "n_items"} <= set(d.columns):
        d["value_per_item"] = (d["order_value"] / d["n_items"].replace(0, np.nan)).fillna(d["order_value"])
    if "is_cash" not in d.columns and "Type" in d.columns:
        d["is_cash"] = (d["Type"].astype(str).str.upper() == "CASH").astype(int)
    return d, next((c for c in OUTCOME_CANDIDATES if c in d.columns), None)


def missing_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(required columns that are missing, optional model features that will be imputed)."""
    required = [c for c in REQUIRED if c not in df.columns]
    optional = [c for c in MODEL_FEATURES if c not in df.columns and c not in required]
    return required, optional


def outcome_labels(series: pd.Series) -> pd.Series:
    return series.astype(float).map(OUTCOME_LABEL)


def render(sample_override: pd.DataFrame | None = None) -> None:
    thr, opts = data.thr(), data.option_lists()
    low_cut, high_cut = thr["tier_cutoffs"]["low_max"], thr["tier_cutoffs"]["high_min"]
    C.page_header("Batch Scoring", "Upload an order batch to run pre-dispatch inference, SHAP reason codes and operational triage.")
    _, test_df = data.datasets()

    with st.container(border=True):
        C.card_title("1. Upload or select batch data")
        left, right = st.columns([1.5, 1], gap="medium")
        with left:
            upload = st.file_uploader("Upload orders CSV", type=["csv"],
                                      help="CSV with pre-dispatch order features. Required: " + ", ".join(REQUIRED) +
                                           ". Other features are imputed when missing.")
        with right:
            st.markdown("**Or load a sample batch of real holdout orders:**")
            if st.button("⚡ 50 test orders (with target)", help="Real holdout orders with the actual outcome, for an audit."):
                st.session_state["batch_sample"] = "labeled"
            if st.button("⚡ 50 test orders (unlabeled)", help="Same kind of orders without the target, to mimic checkout scoring."):
                st.session_state["batch_sample"] = "unlabeled"

    with st.container(border=True):
        C.card_title("CSV template: exact columns, with real holdout orders as examples")
        example = test_df[["Order Id"] + MODEL_FEATURES].head(3).reset_index(drop=True)
        st.caption(f"Columns: Order Id + {len(MODEL_FEATURES)} model features. Required: {', '.join(REQUIRED)}. "
                   f"Optional: add `{TARGET_COL}` (1 = late, 0 = on time) to audit the batch. The three example rows are real holdout orders.")
        C.styled_table(example, formats={c: "{:.3f}" for c in example.columns if pd.api.types.is_float_dtype(example[c])})
        st.download_button("📥 Download this template (CSV)", example.to_csv(index=False).encode("utf-8"),
                           "rto_shield_batch_template.csv", "text/csv")

    df, source = None, ""
    if upload is not None:
        try:
            df, source = pd.read_csv(upload), f"Uploaded file: {upload.name}"
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to read CSV: {exc}")
    elif sample_override is not None:
        df, source = sample_override.copy().reset_index(drop=True), f"Automated test batch ({len(sample_override)} orders)"
    elif st.session_state.get("batch_sample") == "labeled":
        df, source = test_df.sample(50, random_state=42).reset_index(drop=True), "50 sample holdout orders (labelled)"
    elif st.session_state.get("batch_sample") == "unlabeled":
        df = test_df.sample(50, random_state=123).drop(columns=[TARGET_COL]).reset_index(drop=True)
        source = "50 sample holdout orders (unlabelled)"

    if df is None:
        C.callout("info", "Upload a CSV or load a sample batch to begin",
                  "Scoring real Indian D2C orders would first require retraining the model on real courier and return outcomes; this prototype is trained on public DataCo data only.")
        return

    df_clean, outcome_col = harmonize(df, opts["mode_days"])
    required_missing, optional_missing = missing_columns(df_clean)
    with st.container(border=True):
        C.card_title(f"2. Schema validation ({source})")
        if required_missing:
            st.error("Schema error: missing required column(s): " + ", ".join(f"`{c}`" for c in required_missing)
                     + ". Add them (see the template above) and upload again.")
            return
        recognised = len(MODEL_FEATURES) - len(optional_missing)
        st.write(f"Validated **{len(df_clean):,} orders**; recognised **{recognised}/{len(MODEL_FEATURES)}** model features.")
        if optional_missing:
            with st.expander(f"{len(optional_missing)} optional feature(s) missing, imputed by the preprocessor"):
                st.write(", ".join(f"`{c}`" for c in optional_missing))
        unknown = sorted(set(df_clean["Shipping_Mode"].dropna().astype(str)) - set(opts["modes"]))
        if unknown:
            st.warning("Unrecognised Shipping_Mode value(s): " + ", ".join(unknown) + ". They are scored as an unseen category.")
        if outcome_col:
            st.success(f"Outcome column detected: `{outcome_col}`; performance evaluation and calibration will be computed.")
        else:
            st.info("No ground-truth outcome column detected; running inference and tier triage only.")

    with st.spinner("Scoring orders and generating SHAP explanations..."):
        scored = predict.score_batch(df_clean)

    n = len(scored)
    counts = {t: int((scored["tier"] == t).sum()) for t in ["Low", "Medium", "High"]}
    shares = {t: counts[t] / n if n else 0.0 for t in counts}
    lbl = {"Low": f"Low (< {C.fmt_pct(low_cut, clip=False)})",
           "Medium": f"Medium ({C.fmt_pct(low_cut, clip=False)} to {C.fmt_pct(high_cut, clip=False)})",
           "High": f"High (>= {C.fmt_pct(high_cut, clip=False)})"}
    with st.container(border=True):
        C.card_title("3. Triage & risk-tier allocation")
        k = st.columns(4)
        with k[0]:
            C.kpi_card("Total batch volume", f"{n:,}", "Orders processed")
        for col, t, sub in zip(k[1:], ["Low", "Medium", "High"], ["Standard flow", "Monitor SLA", "Flagged for action"]):
            with col:
                C.kpi_card(lbl[t], f"{counts[t]:,}", f"{C.fmt_pct(shares[t], clip=False)} - {sub}")
        a, b = st.columns(2)
        with a:
            C.show(charts.tier_counts_bar(counts, shares, lbl), key="batch_tiers")
        with b:
            ct = pd.crosstab(df_clean["Shipping_Mode"].astype(str).values, scored["tier"].values)
            for t in ["Low", "Medium", "High"]:
                if t not in ct.columns:
                    ct[t] = 0
            ct = ct[["Low", "Medium", "High"]]
            C.show(charts.tier_by_mode(ct), key="batch_by_mode")
        st.markdown("**Tier counts by shipping mode**")
        ct_view = ct.copy()
        ct_view.insert(0, "shipping_mode", ct_view.index)
        ct_view["total"] = ct_view[["Low", "Medium", "High"]].sum(axis=1)
        C.styled_table(ct_view, rename={"shipping_mode": "Shipping Mode"}, formats={"Low": "{:,}", "Medium": "{:,}", "High": "{:,}", "total": "{:,}"})

    if outcome_col:
        labelled = df_clean[outcome_col].notna().values
        y_true = df_clean.loc[labelled, outcome_col].astype(int).values
        sub = scored[labelled]
        ev = predict.evaluate_batch(y_true, sub["raw_score"].values, sub["risk_score"].values, sub["tier"].values)
        with st.container(border=True):
            C.card_title("4. Empirical evaluation & calibration audit")
            if ev["small_sample"]:
                C.callout("warning", "Small sample", ev["warning"])

            def ci_txt(ci):
                return None if ci[0] is None else f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"

            e = st.columns(4)
            with e[0]:
                C.metric("Batch ROC-AUC (raw scores)", "n/a" if ev["roc_auc"] is None else f"{ev['roc_auc']:.3f}", ci_txt(ev["roc_auc_ci"]))
            with e[1]:
                C.metric("Batch PR-AUC (raw scores)", "n/a" if ev["pr_auc"] is None else f"{ev['pr_auc']:.3f}", ci_txt(ev["pr_auc_ci"]))
            with e[2]:
                C.metric("Batch Brier score", "n/a" if ev["brier"] is None else f"{ev['brier']:.3f}", ci_txt(ev["brier_ci"]))
            with e[3]:
                C.metric("Batch late rate", C.fmt_pct(ev["prevalence"], clip=False), f"n = {ev['n']:,} labelled rows")

            st.markdown("**Calibration by risk tier**")
            tab = pd.DataFrame([{
                "tier": r["tier"], "orders": r["n"], "share": r["share"], "mean_predicted": r["mean_predicted"],
                "actual_late_rate": r["actual_late_rate"],
                "calibration_gap": (f"n < {predict.MIN_TIER_N}: suppressed" if r["suppressed"] else f"{r['gap']:+.3f}"),
            } for r in ev["tier_calibration"]])
            C.styled_table(tab, rename={"mean_predicted": "Mean Predicted Score", "actual_late_rate": "Actual Late Rate"},
                           formats={"share": lambda v: C.fmt_pct(v, clip=False), "actual_late_rate": lambda v: C.fmt_pct(v, clip=False)})
            ok = [r for r in ev["tier_calibration"] if not r["suppressed"]]
            if len(ok) < 3:
                st.caption(f"Calibration gaps are suppressed for tiers with fewer than {predict.MIN_TIER_N} orders (too few to be meaningful).")
            if ok:
                C.show(charts.predicted_vs_actual(ok), key="batch_cal")

    with st.container(border=True):
        C.card_title("5. Download & inspect scored results")
        out = scored.copy()
        if outcome_col:
            out.insert(4, "actual", outcome_labels(df_clean[outcome_col]).values)
        st.download_button("📥 Download scored batch (CSV)", out.to_csv(index=False).encode("utf-8"), "rto_shield_batch_scored.csv", "text/csv")
        view = out[[c for c in ["order_id", "risk_score", "tier", "flagged", "actual", "reason_1", "reason_2", "reason_3"] if c in out.columns]].head(20)
        st.markdown("**Preview: first 20 scored orders**")
        C.styled_table(view, rename={"risk_score": "Risk Score", "actual": "Actual Outcome", "reason_1": "Reason 1", "reason_2": "Reason 2",
                                     "reason_3": "Reason 3"}, formats={"risk_score": lambda v: C.fmt_pct(v)})
