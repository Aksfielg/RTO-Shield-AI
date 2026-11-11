"""
app.py
======
Streamlit Dashboard for RTO Shield AI - Model 1.

Comprehensive multi-page interactive web application:
  1. Overview: Project goal, dataset citation, and proxy-target caveat.
  2. Score an order: Interactive pre-dispatch order scoring form with calibrated gauge,
     colored risk tier, flagged status, raw-feature SHAP reasons, and SHAP waterfall.
  3. Real test orders: Curated & searchable real test-set orders with score, tier,
     actual outcome, and feature attribution breakdown.
  4. Model performance: 5-model comparison, ROC/PR curves, calibration, lift/gain,
     decile table, risk tiers, and bootstrap confidence intervals.
  5. Explainability: Global SHAP importance, beeswarm summary, partial dependence,
     and case study waterfalls.
  6. Robustness: Feature ablations, per-shipping-mode evaluation (highlighting Standard Class
     AUC ~0.53 limitation), rolling backtest, and PSI drift check.
  7. Limitations: Honest, comprehensive limitations and ethical governance from MODEL_CARD.md.

Run:
  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup & Matplotlib cache configuration
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

mpl_dir = ROOT_DIR / ".mpl_cache"
mpl_dir.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from config import (
    MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR,
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    TARGET_COL, TRAIN_FRAC, VAL_FRAC,
)
from features import _select_feature_cols
from reasons import get_reason_codes, group_shap_values, _NUMERIC_LABELS
import predict

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RTO Shield AI - Model 1",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for polished styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .caveat-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    .limitation-box {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .tier-badge-low {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        background-color: #d4edda;
        color: #155724;
        font-weight: 700;
        border-radius: 20px;
        font-size: 1.1rem;
    }
    .tier-badge-medium {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        background-color: #fff3cd;
        color: #856404;
        font-weight: 700;
        border-radius: 20px;
        font-size: 1.1rem;
    }
    .tier-badge-high {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 700;
        border-radius: 20px;
        font-size: 1.1rem;
    }
    .flagged-yes {
        color: #dc3545;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .flagged-no {
        color: #28a745;
        font-weight: 700;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Cached Resource Loaders
# ===========================================================================

@st.cache_resource
def load_model_artifacts():
    """Load model, preprocessor, calibrator, thresholds, and SHAP explainer."""
    model_path = MODELS_DIR / "final_model.joblib"
    prep_path  = MODELS_DIR / "preprocessor.joblib"
    cal_path   = MODELS_DIR / "final_calibrator.joblib"
    if not cal_path.exists():
        cal_path = MODELS_DIR / "calibrator.joblib"
    thr_path   = MODELS_DIR / "thresholds.json"

    model = joblib.load(model_path)
    prep  = joblib.load(prep_path)
    cal   = joblib.load(cal_path)
    thr   = json.loads(thr_path.read_text(encoding="utf-8"))

    # Initialise SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    feat_names = list(prep.get_feature_names_out())

    return {
        "model": model,
        "prep": prep,
        "cal": cal,
        "thresholds": thr,
        "explainer": explainer,
        "feat_names": feat_names,
    }


@st.cache_data
def load_datasets():
    """Load processed orders and pre-sliced test partition."""
    parquet_path = PROCESSED_DIR / "orders.parquet"
    df = pd.read_parquet(parquet_path).sort_values("order_date").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (TRAIN_FRAC + VAL_FRAC))
    test_df = df.iloc[test_start:].copy().reset_index(drop=True)
    return df, test_df


@st.cache_data
def load_output_tables():
    """Load summary metrics and validation tables."""
    tables = {}
    csv_files = {
        "model_comparison": OUTPUTS_DIR / "metrics_model_comparison.csv",
        "final_results":    OUTPUTS_DIR / "final_results.csv",
        "ablation":         OUTPUTS_DIR / "ablation_results.csv",
        "backtest":         OUTPUTS_DIR / "backtest_results.csv",
        "decile":           OUTPUTS_DIR / "decile_table.csv",
        "psi":              OUTPUTS_DIR / "psi_results.csv",
        "demo_scores":      OUTPUTS_DIR / "demo_scores.csv",
    }
    for key, path in csv_files.items():
        if path.exists():
            tables[key] = pd.read_csv(path)

    meta_path = OUTPUTS_DIR / "final_results_meta.json"
    if meta_path.exists():
        tables["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))

    return tables


# ===========================================================================
# Visual Component Helpers
# ===========================================================================

def render_risk_gauge(score: float, low_cut: float, high_cut: float, flag_cut: float) -> go.Figure:
    """Render a Plotly gauge showing calibrated risk probability and tier zones."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Calibrated Risk Score", 'font': {'size': 20, 'color': '#212529'}},
        number={'suffix': "%", 'valueformat': ".2f", 'font': {'size': 38, 'color': '#212529'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#495057"},
            'bar': {'color': "#2b5c8f", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#ced4da",
            'steps': [
                {'range': [0, low_cut * 100], 'color': "#d4edda"},
                {'range': [low_cut * 100, high_cut * 100], 'color': "#fff3cd"},
                {'range': [high_cut * 100, 100], 'color': "#f8d7da"},
            ],
            'threshold': {
                'line': {'color': "#dc3545", 'width': 4},
                'thickness': 0.8,
                'value': flag_cut * 100
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif"}
    )
    return fig


def render_shap_waterfall(
    order_dict: dict,
    prep,
    explainer,
    feat_names: list[str],
    max_display: int = 10,
) -> plt.Figure:
    """Generate a clean SHAP waterfall figure using grouped parent features and raw unscaled data."""
    df_row = predict._dict_to_df(order_dict)
    X_raw  = _select_feature_cols(df_row, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    X_enc  = prep.transform(X_raw)

    sv = explainer(X_enc)
    sv_flat = sv.values[0]

    labels, values, val_strs = group_shap_values(sv_flat, feat_names, order_dict)

    clean_exp = shap.Explanation(
        values=values,
        base_values=sv.base_values[0],
        data=val_strs,
        feature_names=labels,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    shap.plots.waterfall(clean_exp, max_display=max_display, show=False)
    plt.title("SHAP Feature Contributions (Grouped Business Factors)", fontsize=12, pad=12)
    plt.tight_layout()
    return fig


# ===========================================================================
# Page 1: Overview
# ===========================================================================

def page_overview():
    st.markdown('<div class="main-title">RTO Shield AI - Model 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Pre-Dispatch Early Risk Prediction & Friction Mitigation</div>', unsafe_allow_html=True)

    # Mandatory visible caveat box
    st.markdown("""
    <div class="caveat-box">
        <h4 style="margin-top:0; color:#856404;">CRITICAL CAVEAT: PROXY TARGET & DATA CONTEXT</h4>
        <p style="margin-bottom:0.4rem; font-size:1.05rem;">
            <strong>Late delivery is a proxy for RTO. DataCo is not Indian COD data.</strong>
        </p>
        <ul style="margin-bottom:0; font-size:0.95rem;">
            <li><strong>Proxy Target</strong>: The model predicts <code>Late_delivery_risk</code> (upstream fulfillment friction) strictly using features available at or before order placement. It does <em>not</em> represent courier refusal, Cash-on-Delivery (COD) failure, or warehouse Return-To-Origin.</li>
            <li><strong>Data Provenance</strong>: Evaluated on the DataCo Smart Supply Chain dataset (Constante et al., 2019), representing Western cross-border retail across Americas and Europe, not Indian domestic D2C logistics.</li>
            <li><strong>Decision Authority</strong>: Advisory only. The model flags high-risk orders for human fulfillment review; it does not autonomously cancel orders.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#6c757d;">Total Order Records</div>
            <div style="font-size:1.8rem; font-weight:700; color:#212529;">65,752</div>
            <div style="font-size:0.8rem; color:#28a745;">Deduplicated at order level</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#6c757d;">Final Model Test ROC-AUC</div>
            <div style="font-size:1.8rem; font-weight:700; color:#2b5c8f;">0.7713</div>
            <div style="font-size:0.8rem; color:#6c757d;">95% CI: [0.762, 0.780]</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#6c757d;">Test PR-AUC</div>
            <div style="font-size:1.8rem; font-weight:700; color:#2b5c8f;">0.8134</div>
            <div style="font-size:0.8rem; color:#6c757d;">95% CI: [0.805, 0.822]</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:0.85rem; color:#6c757d;">Post-Calib Brier / ECE</div>
            <div style="font-size:1.8rem; font-weight:700; color:#155724;">0.1831 / 0.0067</div>
            <div style="font-size:0.8rem; color:#28a745;">Isotonic Regression</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.subheader("Project Goal & Pipeline Architecture")
        st.write("""
        E-commerce sellers face severe profitability erosion when orders fail to deliver on time or are returned.
        **RTO Shield AI - Model 1** investigates whether operational risk can be diagnosed **at checkout**
        before handing parcels over to shipping carriers.

        **Key Architectural Highlights:**
        - **Zero Post-Dispatch Leakage**: Real shipping duration (`Days_for_shipping_real`), delivery status, and tracking dates are strictly forbidden.
        - **Strict Chronological Splitting**: 70% Train (2015-01-01 to 2017-03-16), 15% Validation (2017-03-16 to 2017-09-05), and 15% Test (2017-09-05 to 2018-01-31). Zero temporal overlap.
        - **Leakage-Safe Customer & Regional History**: Historical late rates computed strictly using prior orders (`order_date < current`).
        - **Isotonic Probability Calibration**: Raw tree margins transformed into reliable empirical probabilities (ECE reduced from 0.0495 to 0.0067).
        - **Transparent Decisioning**: 3-tier risk classification coupled with exact SHAP local reason codes expressed in raw business units.
        """)

    with col_right:
        st.subheader("Dataset Citation & Academic Credit")
        st.markdown("""
        > **Primary Academic Source:**  
        > Constante, F., Silva, F., & Pereira, A. (2019).  
        > *DataCo Smart Supply Chain for Big Data Analysis*.  
        > **Mendeley Data**, V5.  
        > **DOI**: [10.17632/8gx2fvg2k6.5](https://doi.org/10.17632/8gx2fvg2k6.5)  
        > **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0).
        """)

        st.subheader("Decisioning Policy (Thresholds)")
        st.markdown("""
        - **Operational Flag Threshold (High Tier)**: `0.6988` (Flagged = High tier, Precision = 88.4%, flags top ~33% of orders).
        - **Precision >= 80% Threshold**: `0.3994` (Precision = 85.1%, Recall = 59.0%, flags top 38.2% of orders).
        - **Max F1 Threshold**: `0.3538` (maximises F1 score on validation set, Recall = 95.5%).
        - **Risk Tier Cutoffs** (Validation 33rd & 67th percentiles of calibrated scores):
          - **Low Risk**: Score < 0.3749 (Standard dispatch, no friction, empirical late rate: 28.9%)
          - **Medium Risk**: 0.3749 ≤ Score < 0.6988 (Monitor carrier pickup SLA, empirical late rate: 42.2%)
          - **High Risk**: Score ≥ 0.6988 (Flagged for proactive intervention, empirical late rate: 88.0%)
        """)


# ===========================================================================
# Page 2: Score an Order
# ===========================================================================

def page_score_order(artifacts):
    st.markdown('<div class="main-title">Score an Order</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluate pre-dispatch risk for a single order in real time</div>', unsafe_allow_html=True)

    # Preset selector
    # Preset selector
    preset_choice = st.radio(
        "Load an Order Profile Preset:",
        [
            "Custom Input",
            "Preset: High Risk (Real Test Order #77188 - First Class)",
            "Preset: Medium Risk (Real Test Order #72160 - Standard Class)",
            "Preset: Low Risk (Real Test Order #67097 - Same Day)",
        ],
        horizontal=True
    )

    defaults = {
        "shipping_mode": "Standard Class",
        "sched_days": 4,
        "payment_type": "DEBIT",
        "order_value": 150.00,
        "n_items": 2,
        "discount_rate": 0.05,
        "market": "Europe",
        "department": "Apparel",
        "segment": "Consumer",
        "order_hour": 14,
        "order_dow": "Wednesday",
        "order_month": 6,
    }

    if preset_choice == "Preset: High Risk (Real Test Order #77188 - First Class)":
        defaults.update({
            "shipping_mode": "First Class", "sched_days": 1, "payment_type": "CASH",
            "order_value": 187.76, "n_items": 2, "discount_rate": 0.16,
            "market": "LATAM", "department": "Footwear", "segment": "Consumer",
            "order_hour": 18, "order_dow": "Friday", "order_month": 12
        })
    elif preset_choice == "Preset: Medium Risk (Real Test Order #72160 - Standard Class)":
        defaults.update({
            "shipping_mode": "Standard Class", "sched_days": 4, "payment_type": "PAYMENT",
            "order_value": 276.92, "n_items": 4, "discount_rate": 0.07,
            "market": "Europe", "department": "Apparel", "segment": "Consumer",
            "order_hour": 14, "order_dow": "Friday", "order_month": 10
        })
    elif preset_choice == "Preset: Low Risk (Real Test Order #67097 - Same Day)":
        defaults.update({
            "shipping_mode": "Same Day", "sched_days": 0, "payment_type": "DEBIT",
            "order_value": 179.99, "n_items": 1, "discount_rate": 0.10,
            "market": "Europe", "department": "Apparel", "segment": "Consumer",
            "order_hour": 11, "order_dow": "Thursday", "order_month": 8
        })

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    mode_day_map = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}

    with st.form("score_order_form"):
        st.subheader("Order Specification")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            shipping_mode = st.selectbox(
                "Shipping Mode",
                ["Standard Class", "First Class", "Second Class", "Same Day"],
                index=["Standard Class", "First Class", "Second Class", "Same Day"].index(defaults["shipping_mode"])
            )
            # Dynamic default scheduled days locked based on shipping mode
            default_sched = mode_day_map.get(shipping_mode, defaults["sched_days"])
            scheduled_days = st.number_input(
                "Scheduled Shipping Days (Locked)",
                min_value=0, max_value=6,
                value=int(default_sched),
                disabled=True,
                help="Contracted delivery SLA days are fixed by shipping mode (Same Day: 0, First Class: 1, Second Class: 2, Standard Class: 4)."
            )
            payment_type = st.selectbox(
                "Payment Type",
                ["DEBIT", "TRANSFER", "PAYMENT", "CASH"],
                index=["DEBIT", "TRANSFER", "PAYMENT", "CASH"].index(defaults["payment_type"])
            )

        with c2:
            order_value = st.number_input("Order Value ($)", min_value=5.0, max_value=5000.0, value=float(defaults["order_value"]), step=10.0)
            n_items = st.number_input("Number of Items", min_value=1, max_value=25, value=int(defaults["n_items"]), step=1)
            avg_discount = st.slider("Average Discount Rate", min_value=0.0, max_value=0.50, value=float(defaults["discount_rate"]), step=0.01)

        with c3:
            market = st.selectbox(
                "Geographic Market",
                ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"],
                index=["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"].index(defaults["market"])
            )
            department = st.selectbox(
                "Product Department",
                ["Apparel", "Fan Shop", "Footwear", "Golf", "Outdoors", "Fitness", "Technology", "Book Shop", "Discs Shop", "Health and Beauty ", "Pet Shop"],
                index=["Apparel", "Fan Shop", "Footwear", "Golf", "Outdoors", "Fitness", "Technology", "Book Shop", "Discs Shop", "Health and Beauty ", "Pet Shop"].index(defaults["department"])
            )
            segment = st.selectbox(
                "Customer Segment",
                ["Consumer", "Corporate", "Home Office"],
                index=["Consumer", "Corporate", "Home Office"].index(defaults["segment"])
            )

        with c4:
            order_hour = st.slider("Order Hour (0-23)", min_value=0, max_value=23, value=int(defaults["order_hour"]))
            order_dow_str = st.selectbox("Day of Week", dow_names, index=dow_names.index(defaults["order_dow"]))
            order_dow = dow_names.index(order_dow_str)
            order_month = st.slider("Order Month", min_value=1, max_value=12, value=int(defaults["order_month"]))

        submitted = st.form_submit_button("🛡️ Compute Risk Score", width="stretch")

    if submitted or preset_choice != "Custom Input":
        # Construct pre-dispatch feature dictionary (lock scheduled_days to shipping mode mode)
        final_sched = mode_day_map.get(shipping_mode, default_sched)
        order_dict = {
            "Shipping_Mode": shipping_mode,
            "Days_for_shipment_scheduled": final_sched,
            "Type": payment_type,
            "order_value": order_value,
            "n_items": n_items,
            "avg_discount_rate": avg_discount,
            "value_per_item": order_value / max(n_items, 1),
            "Market": market,
            "Department_Name": department,
            "Customer_Segment": segment,
            "order_hour": order_hour,
            "order_dow": order_dow,
            "order_month": order_month,
            "is_cash": int(payment_type == "CASH"),
            "is_weekend": int(order_dow in [5, 6]),
            "is_night_order": int(order_hour >= 22 or order_hour <= 5),
            "high_discount_flag": int(avg_discount > 0.15),
            "n_distinct_categories": 1,
        }

        with st.spinner("Executing model scoring and SHAP attribution..."):
            result = predict.score_order(order_dict)

        st.markdown("### Risk Evaluation Results")
        res_col1, res_col2, res_col3 = st.columns([1.1, 1, 1.2])

        thr = artifacts["thresholds"]
        low_cut  = thr["tier_cutoffs"]["low_max"]   # 0.3749
        high_cut = thr["tier_cutoffs"]["high_min"]  # 0.6988
        flag_cut = high_cut                         # Flagged defined as High tier (score >= 0.6988)
        p80_cut  = thr["thresholds"].get("precision_80", 0.3994)
        disp_score = result.get("display_score", f"{result['risk_score']*100:.1f}%")

        with res_col1:
            gauge_fig = render_risk_gauge(result["risk_score"], low_cut, high_cut, flag_cut)
            st.plotly_chart(gauge_fig, width="stretch")
            st.markdown(f"<div style='text-align:center; font-weight:700; font-size:1.1rem; color:#212529;'>Calibrated Score: {disp_score}</div>", unsafe_allow_html=True)

        with res_col2:
            st.write("**Risk Tier Classification**")
            tier = result["tier"]
            if tier == "Low":
                st.markdown(f'<div class="tier-badge-low">LOW RISK ({disp_score})</div>', unsafe_allow_html=True)
                st.caption(f"Score < {low_cut:.4f}. Normal automated processing.")
            elif tier == "Medium":
                st.markdown(f'<div class="tier-badge-medium">MEDIUM RISK ({disp_score})</div>', unsafe_allow_html=True)
                st.caption(f"{low_cut:.4f} <= Score < {high_cut:.4f}. Monitor fulfillment SLA.")
            else:
                st.markdown(f'<div class="tier-badge-high">HIGH RISK ({disp_score})</div>', unsafe_allow_html=True)
                st.caption(f"Score >= {high_cut:.4f}. High probability of late delivery.")

            st.write("**Operational Flag Status**")
            if result["flagged"]:
                st.markdown(f'<div class="flagged-yes">FLAGGED (High Tier: Score >= {high_cut:.4f})</div>', unsafe_allow_html=True)
                st.write("Fulfillment specialist intervention recommended.")
            else:
                st.markdown(f'<div class="flagged-no">NOT FLAGGED (Score < {high_cut:.4f})</div>', unsafe_allow_html=True)
                st.write("Order cleared for standard automated dispatch.")
            st.caption(f"Validation Precision >= 80% threshold: {p80_cut:.4f}.")

        with res_col3:
            st.write("**Top 3 Explainable Reasons (SHAP)**")
            st.markdown("*Attributions expressed in RAW feature values:*")
            for i, reason in enumerate(result["top_reasons"], 1):
                icon = "▲" if "raises risk" in reason else "▼"
                color = "#dc3545" if "raises risk" in reason else "#28a745"
                st.markdown(f"<span style='color:{color}; font-weight:700;'>{icon}</span> {reason}", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Local SHAP Feature Attribution Waterfall")
        st.write("Visual breakdown of how each feature shifts the log-odds margin from the baseline expectation:")
        fig = render_shap_waterfall(
            order_dict,
            artifacts["prep"],
            artifacts["explainer"],
            artifacts["feat_names"]
        )
        st.pyplot(fig)
        plt.close(fig)


# ===========================================================================
# Page 3: Real Test Orders
# ===========================================================================

def page_real_test_orders(artifacts, test_df, tables):
    st.markdown('<div class="main-title">Real Test Orders</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect real unseen orders from the 2017-2018 holdout test partition</div>', unsafe_allow_html=True)

    demo_df = tables.get("demo_scores")

    st.subheader("1. Curated Benchmark Cases (5 Diverse Test Orders)")
    st.write("These 5 orders represent high, medium, and low risk profiles with verified actual outcomes:")

    if demo_df is not None:
        styled_demo = demo_df.copy()
        st.dataframe(styled_demo, width="stretch")

    st.markdown("---")
    st.subheader("2. Interactive Order Inspection")

    selection_mode = st.radio(
        "Choose order source:",
        ["Pick from 5 Curated Demo Orders", "Search / Select Any Order from Full Test Set (9,863 orders)"],
        horizontal=True
    )

    selected_order_dict = None
    actual_outcome = None

    if selection_mode == "Pick from 5 Curated Demo Orders" and demo_df is not None:
        demo_order_ids = demo_df["order_id"].tolist()
        demo_labels = [f"Order #{row['order_id']} | Tier: {row['tier']} | Mode: {row['shipping_mode']} | Actual: {row['actual']}"
                       for _, row in demo_df.iterrows()]
        idx_choice = st.selectbox("Select Benchmark Order:", range(len(demo_labels)), format_func=lambda i: demo_labels[i])
        target_order_id = demo_order_ids[idx_choice]

        matched = test_df[test_df["Order Id"] == target_order_id]
        if not matched.empty:
            selected_order_dict = matched.iloc[0].to_dict()
            actual_outcome = int(matched.iloc[0][TARGET_COL])
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            mode_filter = st.selectbox("Filter by Shipping Mode:", ["All"] + sorted(test_df["Shipping_Mode"].dropna().unique().tolist()))
        with col_f2:
            outcome_filter = st.selectbox("Filter by Actual Outcome:", ["All", "Late Only (1)", "On-time Only (0)"])

        filtered = test_df
        if mode_filter != "All":
            filtered = filtered[filtered["Shipping_Mode"] == mode_filter]
        if outcome_filter == "Late Only (1)":
            filtered = filtered[filtered[TARGET_COL] == 1]
        elif outcome_filter == "On-time Only (0)":
            filtered = filtered[filtered[TARGET_COL] == 0]

        st.caption(f"Showing matching {len(filtered):,} test orders.")
        order_options = filtered["Order Id"].head(200).tolist()
        selected_id = st.selectbox("Select Order ID:", order_options)
        matched = test_df[test_df["Order Id"] == selected_id]
        if not matched.empty:
            selected_order_dict = matched.iloc[0].to_dict()
            actual_outcome = int(matched.iloc[0][TARGET_COL])

    if selected_order_dict:
        st.markdown("---")
        result = predict.score_order(selected_order_dict)

        disp_score = result.get("display_score", f"{result['risk_score']*100:.1f}%")
        c_a, c_b, c_c, c_d = st.columns(4)
        with c_a:
            st.metric("Order ID", str(selected_order_dict.get("Order Id", "N/A")))
        with c_b:
            st.metric("Calibrated Risk Score", disp_score)
        with c_c:
            st.metric("Assigned Tier", result["tier"])
        with c_d:
            act_str = "Late (1)" if actual_outcome == 1 else "On-time (0)"
            match = (result["flagged"] and actual_outcome == 1) or (not result["flagged"] and actual_outcome == 0)
            status_note = "CORRECT" if match else "MISMATCH"
            st.metric("Actual Delivery Outcome", act_str, delta=status_note, delta_color="normal" if match else "inverse")

        col_left, col_right = st.columns([1, 1.2])
        with col_left:
            st.write("**Top 3 SHAP Reason Codes (Raw Units)**")
            for r in result["top_reasons"]:
                st.markdown(f"• `{r}`")

            st.write("**Key Order Characteristics**")
            features_summary = {
                "Shipping Mode": selected_order_dict.get("Shipping_Mode"),
                "Scheduled Days": selected_order_dict.get("Days_for_shipment_scheduled"),
                "Payment Type": selected_order_dict.get("Type"),
                "Order Value ($)": f"${float(selected_order_dict.get('order_value', 0)):.2f}",
                "Number of Items": selected_order_dict.get("n_items"),
                "Order Hour": f"{int(selected_order_dict.get('order_hour', 0)):02d}:00",
                "Market": selected_order_dict.get("Market"),
                "Region": selected_order_dict.get("Order_Region"),
                "Segment": selected_order_dict.get("Customer_Segment"),
            }
            feat_df = pd.DataFrame(list(features_summary.items()), columns=["Attribute", "Value"])
            feat_df["Value"] = feat_df["Value"].astype(str)
            st.table(feat_df)

        with col_right:
            st.write("**SHAP Waterfall for this Test Order**")
            fig = render_shap_waterfall(
                selected_order_dict,
                artifacts["prep"],
                artifacts["explainer"],
                artifacts["feat_names"]
            )
            st.pyplot(fig)
            plt.close(fig)


# ===========================================================================
# Page 4: Model Performance
# ===========================================================================

def page_model_performance(tables):
    st.markdown('<div class="main-title">Model Performance & Calibration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous evaluation on chronological holdout validation and test partitions</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Comparison", "Curves & Calibration", "Deciles & Tiers", "Bootstrap Stability"
    ])

    with tab1:
        st.subheader("Model Benchmark (Validation & Test Sets)")
        st.write("Comparison of candidate models against the heuristic baseline (`Shipping_Mode` + `Days_for_shipment_scheduled`):")

        comp_df = tables.get("model_comparison")
        if comp_df is not None:
            st.dataframe(comp_df.style.highlight_max(subset=["roc_auc", "pr_auc", "f1"], color="#d4edda"), width="stretch")

        st.markdown("""
        **Benchmark Observations:**
        - **LightGBM** achieved the highest test ROC-AUC (0.7678 default, 0.7713 tuned) and PR-AUC (0.8354 default, 0.8134 tuned).
        - **Precision at Top 10%**: 100.0% for all gradient boosted tree models (Random Forest, XGBoost, LightGBM), driven by First Class orders (which constitute 99.7% of the top decile) and tight dispatch scheduling.
        - **Brier Score**: LightGBM achieved the lowest uncalibrated Brier score (0.1833), confirming superior probability ranking.
        """)

    with tab2:
        st.subheader("Diagnostic Curves")
        c1, c2 = st.columns(2)
        with c1:
            roc_img = OUTPUTS_DIR / "roc_curves.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="Figure 1: Receiver Operating Characteristic (ROC) Curves", width="stretch")
        with c2:
            pr_img = OUTPUTS_DIR / "pr_curves.png"
            if pr_img.exists():
                st.image(str(pr_img), caption="Figure 2: Precision-Recall (PR) Curves", width="stretch")

        st.subheader("Probability Calibration & Lift")
        c3, c4 = st.columns(2)
        with c3:
            cal_img = OUTPUTS_DIR / "calibration.png"
            if cal_img.exists():
                st.image(str(cal_img), caption="Figure 3: Reliability Calibration Curves (Before vs After Isotonic Regression)", width="stretch")
        with c4:
            lift_img = OUTPUTS_DIR / "lift_gain.png"
            if lift_img.exists():
                st.image(str(lift_img), caption="Figure 4: Cumulative Gain and Decile Lift Charts", width="stretch")

    with tab3:
        st.subheader("Decile Lift Breakdown (Test Set)")
        st.write("Test set orders partitioned into 10 deciles ranked by predicted risk score:")

        decile_df = tables.get("decile")
        if decile_df is not None:
            st.dataframe(decile_df, width="stretch")

        st.markdown("""
        **Lift Highlights:**
        - **Decile 1 (Top 10% highest risk)**: 100% late rate (986 of 986 orders late), producing a lift of **1.82x** over the 55.05% baseline late rate.
        - **Top 3 Deciles**: Capture **48.3% of all late deliveries** in the test set while flagging only 30% of order volume.
        """)

        st.subheader("Operational Risk Tiers")
        tier_data = [
            {"Tier": "Low", "Score Range": "score < 0.3749", "Test Volume": "1,740 (17.6%)", "Actual Late Rate": "28.91%", "Late Captured": "9.3%", "Policy": "Standard dispatch; automated flow"},
            {"Tier": "Medium", "Score Range": "0.3749 <= score < 0.6988", "Test Volume": "4,850 (49.2%)", "Actual Late Rate": "42.23%", "Late Captured": "37.7%", "Policy": "Monitor carrier pickup SLA"},
            {"Tier": "High", "Score Range": "score >= 0.6988", "Test Volume": "3,273 (33.2%)", "Actual Late Rate": "87.96%", "Late Captured": "53.0%", "Policy": "Flagged: Proactive routing & verification"}
        ]
        st.table(pd.DataFrame(tier_data))

    with tab4:
        st.subheader("Multi-Seed Variance & Bootstrap Confidence Intervals")
        st.write("Evaluating stability across 10 random seeds and 1,000 bootstrap resamples:")

        final_res_df = tables.get("final_results")
        if final_res_df is not None:
            st.dataframe(final_res_df, width="stretch")

        meta = tables.get("meta", {})
        boot = meta.get("bootstrap_cis", {})
        if boot:
            st.markdown("### 95% Bootstrap Confidence Intervals (1,000 resamples)")
            b_data = [
                {"Metric": "Final Model ROC-AUC", "Estimate": "0.7713", "95% CI Lower": f"{boot.get('final_roc_auc', [0,0])[0]:.4f}", "95% CI Upper": f"{boot.get('final_roc_auc', [0,0])[1]:.4f}"},
                {"Metric": "Final Model PR-AUC", "Estimate": "0.8134", "95% CI Lower": f"{boot.get('final_pr_auc', [0,0])[0]:.4f}", "95% CI Upper": f"{boot.get('final_pr_auc', [0,0])[1]:.4f}"},
                {"Metric": "Delta ROC-AUC (Final - Baseline)", "Estimate": "+0.0422", "95% CI Lower": f"{boot.get('delta_roc_auc', [0,0])[0]:.4f}", "95% CI Upper": f"{boot.get('delta_roc_auc', [0,0])[1]:.4f}"},
                {"Metric": "Delta PR-AUC (Final - Baseline)", "Estimate": "+0.0631", "95% CI Lower": f"{boot.get('delta_pr_auc', [0,0])[0]:.4f}", "95% CI Upper": f"{boot.get('delta_pr_auc', [0,0])[1]:.4f}"},
            ]
            st.table(pd.DataFrame(b_data))
            st.caption("Delta CI excludes zero with high significance, confirming the full feature set genuinely outperforms the shipping-only baseline.")


# ===========================================================================
# Page 5: Explainability
# ===========================================================================

def page_explainability():
    st.markdown('<div class="main-title">Model Explainability & SHAP Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Global and local feature attributions via TreeExplainer</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Global Feature Importance", "SHAP Dependence", "Case Study Waterfalls"])

    with tab1:
        st.subheader("Global Importance & Directional Impact")
        col1, col2 = st.columns(2)
        with col1:
            imp_img = OUTPUTS_DIR / "shap_importance.png"
            if imp_img.exists():
                st.image(str(imp_img), caption="Figure 5: Mean |SHAP| Feature Importance", width="stretch")
        with col2:
            sum_img = OUTPUTS_DIR / "shap_summary.png"
            if sum_img.exists():
                st.image(str(sum_img), caption="Figure 6: SHAP Beeswarm Summary Plot", width="stretch")

        st.markdown("""
        **Key Global Insights:**
        1. **`Shipping_Mode` (Standard Class vs First Class)**: Dominant drivers accounting for over 48% of total attribution mass.
        2. **`Days_for_shipment_scheduled`**: Pre-dispatch contracted window (0-4 days).
        3. **`Type` (Payment Method)**: TRANSFER orders exhibit a lower late rate (48.2% vs ~57% for other payment types), reducing risk.
        4. **`order_hour`**: Sharp step function for same-day dispatches (orders placed at/after 12:00 PM are overwhelmingly late).
        """)

    with tab2:
        st.subheader("SHAP Dependence Plots (5 Continuous Features)")
        st.write("Examining non-linear effects and directional SHAP contributions for key continuous predictors:")
        dp1, dp2 = st.columns(2)
        with dp1:
            p1 = OUTPUTS_DIR / "shap_dep_order_hour.png"
            if p1.exists():
                st.image(str(p1), caption="Figure: SHAP Dependence – Order Hour", width="stretch")
        with dp2:
            p2 = OUTPUTS_DIR / "shap_dep_scheduled_days.png"
            if p2.exists():
                st.image(str(p2), caption="Figure: SHAP Dependence – Scheduled Shipping Days", width="stretch")

        dp3, dp4 = st.columns(2)
        with dp3:
            p3 = OUTPUTS_DIR / "shap_dep_order_value.png"
            if p3.exists():
                st.image(str(p3), caption="Figure: SHAP Dependence – Order Value ($)", width="stretch")
        with dp4:
            p4 = OUTPUTS_DIR / "shap_dep_value_per_item.png"
            if p4.exists():
                st.image(str(p4), caption="Figure: SHAP Dependence – Value per Item ($/item)", width="stretch")

        dp5, _ = st.columns(2)
        with dp5:
            p5 = OUTPUTS_DIR / "shap_dep_customer_prev_late_rate.png"
            if p5.exists():
                st.image(str(p5), caption="Figure: SHAP Dependence – Customer Historical Late Rate", width="stretch")

    with tab3:
        st.subheader("Representative Case Study Waterfalls")
        st.write("Three canonical operational scenarios from the test partition:")
        w1, w2, w3 = st.columns(3)
        with w1:
            cw1 = OUTPUTS_DIR / "shap_waterfall_confident_late.png"
            if cw1.exists():
                st.image(str(cw1), caption="Case 1: Confident Late Delivery", width="stretch")
        with w2:
            cw2 = OUTPUTS_DIR / "shap_waterfall_confident_ontime.png"
            if cw2.exists():
                st.image(str(cw2), caption="Case 2: Confident On-Time Delivery", width="stretch")
        with w3:
            cw3 = OUTPUTS_DIR / "shap_waterfall_false_negative.png"
            if cw3.exists():
                st.image(str(cw3), caption="Case 3: False Negative Analysis", width="stretch")


# ===========================================================================
# Page 6: Robustness
# ===========================================================================

def page_robustness(tables):
    st.markdown('<div class="main-title">Robustness & Stress Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ablations, temporal backtesting, per-segment evaluation, and feature drift</div>', unsafe_allow_html=True)

    # CRITICAL WARNING CALLOUT
    st.markdown("""
    <div class="limitation-box">
        <h4 style="margin-top:0; color:#721c24;">CRITICAL SEGMENT LIMITATION: STANDARD CLASS AUC ~0.53</h4>
        <p style="margin-bottom:0.4rem; font-size:1.02rem;">
            <strong>Standard Class constitutes 59.7% of all test volume (5,888 orders), but within this segment, the model's ROC-AUC drops to 0.5296.</strong>
        </p>
        <p style="margin-bottom:0; font-size:0.95rem;">
            While the model demonstrates outstanding predictive discrimination on <em>First Class</em> (ROC-AUC 0.8732) and <em>Same Day</em> (ROC-AUC 0.9914), it provides essentially zero discriminative power inside Standard Class. Unscheduled ground-carrier delays in standard fulfillment cannot be detected from checkout-time signals alone.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Ablation Study", "Per-Shipping-Mode Analysis", "Rolling-Origin Backtest", "Drift Analysis (PSI)"
    ])

    with tab1:
        st.subheader("Feature Ablation Experiments")
        st.write("Evaluating model degradation when specific feature groups are excluded:")

        col1, col2 = st.columns([1, 1.2])
        with col1:
            abl_df = tables.get("ablation")
            if abl_df is not None:
                st.dataframe(abl_df, width="stretch")
        with col2:
            abl_chart = OUTPUTS_DIR / "ablation_chart.png"
            if abl_chart.exists():
                st.image(str(abl_chart), caption="Figure 7: Ablation ROC-AUC vs PR-AUC", width="stretch")

        st.caption("Ablation conclusion: Dropping shipping mode & scheduled days collapses ROC-AUC from 0.7713 to 0.5594. Shipping parameters are the primary structural determinants.")

    with tab2:
        st.subheader("Per-Shipping-Mode Segmentation")
        st.write("Comparison of the full model vs baseline within each shipping mode:")

        mode_data = [
            {"Shipping Mode": "First Class", "Test Orders": 1513, "Share %": "15.3%", "Full Model ROC-AUC": 0.8732, "Full Model PR-AUC": 0.9915, "Baseline ROC-AUC": 0.5000, "Delta ROC-AUC": "+0.3732"},
            {"Shipping Mode": "Same Day", "Test Orders": 539, "Share %": "5.5%", "Full Model ROC-AUC": 0.9914, "Full Model PR-AUC": 0.9844, "Baseline ROC-AUC": 0.5000, "Delta ROC-AUC": "+0.4914"},
            {"Shipping Mode": "Second Class", "Test Orders": 1923, "Share %": "19.5%", "Full Model ROC-AUC": 0.5600, "Full Model PR-AUC": 0.7936, "Baseline ROC-AUC": 0.5000, "Delta ROC-AUC": "+0.0600"},
            {"Shipping Mode": "Standard Class", "Test Orders": 5888, "Share %": "59.7%", "Full Model ROC-AUC": 0.5296, "Full Model PR-AUC": 0.3981, "Baseline ROC-AUC": 0.5000, "Delta ROC-AUC": "+0.0296"},
        ]
        st.dataframe(pd.DataFrame(mode_data).style.apply(
            lambda row: ['background-color: #f8d7da' if row["Shipping Mode"] == "Standard Class" else '' for _ in row],
            axis=1
        ), width="stretch")

    with tab3:
        st.subheader("Rolling-Origin Backtesting (5 Expanding Folds)")
        st.write("Expanding window cross-validation over chronological time periods:")

        col_b1, col_b2 = st.columns([1.1, 1])
        with col_b1:
            back_df = tables.get("backtest")
            if back_df is not None:
                st.dataframe(back_df, width="stretch")
            st.markdown("""
            - **Mean Fold ROC-AUC**: `0.7695 ± 0.0057`
            - **Mean Fold PR-AUC**: `0.8135 ± 0.0040`
            - **Conclusion**: Performance exhibits high temporal consistency across 2016-2017 without seasonal collapse.
            """)
        with col_b2:
            bt_plot = OUTPUTS_DIR / "backtest_plot.png"
            if bt_plot.exists():
                st.image(str(bt_plot), caption="Figure 8: Temporal Stability Across Folds", width="stretch")

    with tab4:
        st.subheader("Population Stability Index (PSI) Drift Check")
        st.write("Quantifying covariate shift between Train and Test sets for top 8 features:")

        psi_df = tables.get("psi")
        if psi_df is not None:
            st.dataframe(psi_df, width="stretch")

        st.caption("All top features exhibit PSI < 0.0001, confirming zero significant distribution drift across the temporal boundary.")


# ===========================================================================
# Page 7: Limitations
# ===========================================================================

def page_limitations():
    st.markdown('<div class="main-title">Honest System Limitations & Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Critical disclosure of architectural constraints and ethical bounds (from MODEL_CARD.md)</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. The Proxy-Target Reality
    `Late_delivery_risk` is an **upstream operational proxy**, NOT Return-To-Origin (RTO).
    - In Indian D2C e-commerce, **RTO occurs when a buyer refuses delivery, cancels at the doorstep, or provides a fake address**.
    - In the DataCo dataset, the label indicates whether a parcel's delivery time exceeded the contracted SLA.
    - While delivery delays correlate with higher cancellation propensity, **this model does not predict package refusal or cash return**.

    ### 2. Near Rule-Based Label Construction
    - In the underlying DataCo database, shipping mode and contracted delivery days heavily govern the ground truth.
    - Same Day orders placed after 12:00 PM are almost deterministically late due to synthesis rules.
    - Precision at the top 10% being 100% is an artefact of this synthetic regularity rather than superhuman artificial intelligence.

    ### 3. The Standard Class Blind Spot
    - **59.7% of all orders** use Standard Class shipping.
    - Within this critical majority, the model's ROC-AUC drops to **0.5296** (near random coin toss).
    - If a company relies on this model exclusively for Standard Class shipping, it will yield negligible predictive advantage.

    ### 4. Geographic & Domain Mismatch
    - DataCo represents cross-border shipments originating primarily in Puerto Rico/US and shipping to Latin America, Europe, and Asia.
    - It contains **zero Indian PIN-code infrastructure, courier aggregation data (e.g., Shiprocket/Delhivery), or COD cash fraud signals**.
    - Retraining on domestic logistics data is an absolute prerequisite for production deployment in India.

    ### 5. Ethical Governance & Fair Fulfillment
    - **No Customer Blacklisting**: The model flags shipments to adjust fulfillment priority or trigger confirmation calls; it must **never** be used to blacklist buyers.
    - **Zero PII**: No customer names, email addresses, phone numbers, or street addresses are processed.
    - **Human-in-the-Loop**: All automated interventions above threshold must be audited by human fulfillment personnel.
    """)


# ===========================================================================
# Main Application Router
# ===========================================================================

def main():
    st.sidebar.title("🛡️ RTO Shield AI")
    st.sidebar.caption("Model 1: Pre-Dispatch Risk Engine")

    pages = [
        "1. Overview",
        "2. Score an order",
        "3. Real test orders",
        "4. Model performance",
        "5. Explainability",
        "6. Robustness",
        "7. Limitations",
    ]

    choice = st.sidebar.radio("Navigation", pages, index=0)

    # Preload cached artifacts
    artifacts = load_model_artifacts()
    _, test_df = load_datasets()
    tables    = load_output_tables()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Model Metadata**
    - Architecture: LightGBM (Tuned)
    - Threshold: `0.3538`
    - Calibration: Isotonic
    - Split: Chronological (70/15/15)
    """)

    if choice == "1. Overview":
        page_overview()
    elif choice == "2. Score an order":
        page_score_order(artifacts)
    elif choice == "3. Real test orders":
        page_real_test_orders(artifacts, test_df, tables)
    elif choice == "4. Model performance":
        page_model_performance(tables)
    elif choice == "5. Explainability":
        page_explainability()
    elif choice == "6. Robustness":
        page_robustness(tables)
    elif choice == "7. Limitations":
        page_limitations()


if __name__ == "__main__":
    main()
