"""
app.py
======
Redesigned Streamlit Dashboard for RTO Shield AI - Model 1.

Clean, professional multi-page application using st.navigation and st.Page:
  Section 1: Overview
    - Home: Hero title, one-line goal, 4 KPI cards, 3-step "How it works" strip,
            and "What this does NOT show" card.
  Section 2: Decisioning & Operations
    - Score an Order: Form on the left, result card on the right (gauge, tier badge,
      plain-English summary, grouped SHAP horizontal bar chart, expander for technical waterfall).
    - Real Test Orders: Filters, styled benchmark table, detail card with Correct/Missed/False Alarm badge.
    - Impact Simulator: Interactive what-if ROI calculator for proactive intervention policies.
  Section 3: Model Evaluation & Governance
    - Performance: Plotly tabs for Comparison, Curves, Calibration, Lift, and Tiers.
    - Explainability: Grouped feature importance, SHAP beeswarm, and 5 continuous dependence tabs.
    - Robustness: Ablation chart, per-shipping-mode evaluation with highlighted Standard Class AUC 0.53 card,
      rolling backtest, and PSI table.
    - Limitations: Comprehensive model governance and constraints from MODEL_CARD.md.

Design System:
  - Clean white canvas with indigo accent (#4F46E5 / #4338CA).
  - Consistent risk palette (Low: Green #10B981, Medium: Amber #F59E0B, High: Red #EF4444).
  - Rounded cards, generous padding, styled tables with 3-decimal formatting.
  - Slim top badge on every page: "Proxy target: late delivery, DataCo public data, not Indian COD."
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
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
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
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RTO Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS Design System
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global Canvas */
    .stApp {
        background-color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1E293B;
    }

    /* Slim Top Proxy Badge */
    .rto-proxy-badge {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 7px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1.25rem;
    }
    .rto-proxy-badge b {
        color: #78350F;
    }

    /* Cards */
    .rto-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
        margin-bottom: 1.25rem;
    }
    .rto-card-highlight {
        background: #FFFFFF;
        border: 1px solid #C7D2FE;
        border-left: 5px solid #4F46E5;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(79,70,229,0.06);
    }
    .rto-card-warning {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #F59E0B;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 1.25rem;
    }
    .rto-card-danger {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #EF4444;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 1.25rem;
    }

    /* KPI Metric Cards */
    .rto-metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 110px;
    }
    .rto-metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .rto-metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .rto-metric-sub {
        font-size: 0.8rem;
        color: #4F46E5;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Step Strip */
    .rto-step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #4F46E5;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        height: 100%;
    }
    .rto-step-num {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4F46E5;
        margin-bottom: 6px;
    }
    .rto-step-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
    }
    .rto-step-desc {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.45;
    }

    /* Risk Tier Badges */
    .rto-badge {
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        letter-spacing: 0.02em;
    }
    .rto-badge-low {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .rto-badge-medium {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .rto-badge-high {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .rto-badge-indigo {
        background: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }

    /* Titles */
    .rto-hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .rto-hero-sub {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }

    /* Section Subheads */
    .rto-section-head {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.55rem 1.25rem;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338CA;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
    }
</style>
"""

# ---------------------------------------------------------------------------
# Caching Data & Artifact Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def get_artifacts():
    """Load LightGBM model, preprocessor, calibrator, thresholds, explainer."""
    model_path = MODELS_DIR / "final_model.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "lightgbm.joblib"
    model = joblib.load(model_path)

    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")

    cal_path = MODELS_DIR / "final_calibrator.joblib"
    if not cal_path.exists():
        cal_path = MODELS_DIR / "calibrator.joblib"
    calibrator = joblib.load(cal_path)

    thr_data = json.loads((MODELS_DIR / "thresholds.json").read_text(encoding="utf-8"))
    feat_names = list(prep.get_feature_names_out())
    explainer = shap.TreeExplainer(model)

    return {
        "model": model,
        "prep": prep,
        "calibrator": calibrator,
        "thresholds": thr_data,
        "feat_names": feat_names,
        "explainer": explainer,
    }

@st.cache_data
def get_datasets():
    """Load orders and holdout test slice."""
    df = pd.read_parquet(PROCESSED_DIR / "orders.parquet")
    df = df.sort_values("order_date").reset_index(drop=True)
    n = len(df)
    test_start = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    test_df = df.iloc[test_start:].copy().reset_index(drop=True)
    return df, test_df

@st.cache_data
def get_tables():
    """Load CSV tables from outputs directory."""
    tables = {}
    csv_map = {
        "model_comparison": OUTPUTS_DIR / "model_comparison.csv",
        "final_results":    OUTPUTS_DIR / "final_results.csv",
        "ablation":         OUTPUTS_DIR / "ablation_results.csv",
        "backtest":         OUTPUTS_DIR / "backtest_results.csv",
        "decile":           OUTPUTS_DIR / "decile_table.csv",
        "psi":              OUTPUTS_DIR / "psi_results.csv",
        "demo_scores":      OUTPUTS_DIR / "demo_scores.csv",
    }
    for k, p in csv_map.items():
        if p.exists():
            tables[k] = pd.read_csv(p)

    meta_p = OUTPUTS_DIR / "final_results_meta.json"
    if meta_p.exists():
        tables["meta"] = json.loads(meta_p.read_text(encoding="utf-8"))

    return tables

# Backwards-compatible aliases
load_model_artifacts = get_artifacts
load_datasets = get_datasets
load_output_tables = get_tables

@st.cache_data
def get_test_predictions():
    """Compute calibrated scores on full test set once for interactive tools."""
    artifacts = get_artifacts()
    _, test_df = get_datasets()
    X_raw = _select_feature_cols(test_df, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    X_te  = artifacts["prep"].transform(X_raw)
    raw_p = artifacts["model"].predict_proba(X_te)[:, 1]
    cal_p = artifacts["calibrator"].predict(raw_p)
    return cal_p, test_df[TARGET_COL].values

# ---------------------------------------------------------------------------
# Reusable Visual Helpers
# ---------------------------------------------------------------------------
def render_top_badge():
    """Display the required slim top badge on every page."""
    st.markdown("""
    <div class="rto-proxy-badge">
        <span>⚠️</span>
        <span><b>Proxy target:</b> late delivery &bull; DataCo public data &bull; not Indian COD.</span>
    </div>
    """, unsafe_allow_html=True)

def render_risk_gauge(score: float, low_cut: float, high_cut: float, flag_cut: float = 0.6988) -> go.Figure:
    """Render a clean indigo-accented gauge with low/medium/high colored bands."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'valueformat': ".1f", 'font': {'size': 40, 'color': '#0F172A', 'family': 'sans-serif'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#4F46E5", 'thickness': 0.32},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, low_cut * 100], 'color': "#D1FAE5"},
                {'range': [low_cut * 100, high_cut * 100], 'color': "#FEF3C7"},
                {'range': [high_cut * 100, 100], 'color': "#FEE2E2"},
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3.5},
                'thickness': 0.85,
                'value': high_cut * 100,
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=25, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "sans-serif"}
    )
    return fig

def render_grouped_shap_bar(grouped_labels: list[str], grouped_values: np.ndarray, val_strs: list[str], top_k: int = 6) -> go.Figure:
    """Render a horizontal bar chart of grouped parent SHAP attributions."""
    order = np.argsort(np.abs(grouped_values))[::-1][:top_k]
    sel_labels = [grouped_labels[i] for i in order][::-1]
    sel_values = [grouped_values[i] for i in order][::-1]
    sel_vals   = [val_strs[i] for i in order][::-1]

    colors = ["#EF4444" if v > 0 else "#10B981" for v in sel_values]
    hover_texts = [
        f"<b>{lbl}</b>: {val}<br>SHAP log-odds: {v:+.4f}<br>{'▲ Raises risk' if v > 0 else '▼ Lowers risk'}"
        for lbl, val, v in zip(sel_labels, sel_vals, sel_values)
    ]

    fig = go.Figure(go.Bar(
        x=sel_values,
        y=[f"{lbl} ({val})" for lbl, val in zip(sel_labels, sel_vals)],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hoverinfo="text",
        hovertext=hover_texts,
    ))
    fig.update_layout(
        title=dict(text="<b>Key Risk Drivers (Grouped Factors)</b>", font=dict(size=13, color="#1E293B")),
        xaxis=dict(
            title="SHAP Impact (Log-Odds)",
            zeroline=True,
            zerolinecolor="#94A3B8",
            zerolinewidth=1.5,
            gridcolor="#F1F5F9",
            tickfont=dict(size=10)
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color="#334155")),
        margin=dict(l=10, r=20, t=35, b=30),
        height=250,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig

def render_shap_waterfall(
    order_dict: dict,
    prep,
    explainer,
    feat_names: list[str],
    max_display: int = 10,
) -> plt.Figure:
    """Generate a clean SHAP waterfall figure using grouped parent features and raw data."""
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

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    shap.plots.waterfall(clean_exp, max_display=max_display, show=False)
    plt.title("Local SHAP Attribution Waterfall (Grouped Factors)", fontsize=11, pad=10)
    plt.tight_layout()
    return fig

# ===========================================================================
# Page 1: Home
# ===========================================================================
def page_home():
    render_top_badge()

    st.markdown('<div class="rto-hero-title">RTO Shield AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="rto-hero-sub">Pre-dispatch early-warning late delivery risk scoring and friction mitigation at checkout time.</div>', unsafe_allow_html=True)

    # 4 KPI Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="rto-metric-card">
            <div class="rto-metric-label">Dataset Orders</div>
            <div class="rto-metric-val">65,752</div>
            <div class="rto-metric-sub">Deduplicated order records</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="rto-metric-card">
            <div class="rto-metric-label">Final Model ROC-AUC</div>
            <div class="rto-metric-val">0.771</div>
            <div class="rto-metric-sub">95% CI: [0.762, 0.780]</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="rto-metric-card">
            <div class="rto-metric-label">Test PR-AUC</div>
            <div class="rto-metric-val">0.813</div>
            <div class="rto-metric-sub">95% CI: [0.805, 0.822]</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="rto-metric-card">
            <div class="rto-metric-label">Post-Calib ECE</div>
            <div class="rto-metric-val">0.007</div>
            <div class="rto-metric-sub">Brier: 0.183 (Isotonic)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="rto-section-head">How It Works</div>', unsafe_allow_html=True)

    # 3-Step Strip
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="rto-step-card">
            <div class="rto-step-num">Step 01</div>
            <div class="rto-step-title">Order Details</div>
            <div class="rto-step-desc">
                Pre-dispatch checkout signals captured at order placement: shipping mode, contracted SLA days,
                basket value, item count, payment method, geographic destination, and placement hour.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="rto-step-card">
            <div class="rto-step-num">Step 02</div>
            <div class="rto-step-title">Risk Model</div>
            <div class="rto-step-desc">
                A tuned LightGBM classifier evaluates multi-dimensional signals, followed by Isotonic Regression
                to translate raw tree margins into true empirical late-delivery probabilities.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="rto-step-card">
            <div class="rto-step-num">Step 03</div>
            <div class="rto-step-title">Tier + Reasons</div>
            <div class="rto-step-desc">
                Orders are triaged into Low, Medium, or High (Flagged) operational tiers, paired with 3 plain-English
                grouped SHAP reasons to guide targeted routing and fulfillment review.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="rto-section-head">Critical Context & Bounds</div>', unsafe_allow_html=True)

    # "What this does NOT show" Card
    st.markdown("""
    <div class="rto-card-warning">
        <h4 style="margin-top:0; color:#92400E; font-size:1.1rem;">⚠️ What This Model Does NOT Show</h4>
        <ul style="margin-bottom:0; color:#78350F; line-height:1.6; font-size:0.95rem;">
            <li><strong>Not Indian Doorstep RTO</strong>: The target is <code>Late_delivery_risk</code> (upstream fulfillment SLA delay), not buyer refusal, fake door delivery, or COD cash default.</li>
            <li><strong>Near Rule-Based Synthesis</strong>: DataCo dispatch rules heavily dictate outcomes for Same Day and First Class (orders placed after 12:00 PM are almost deterministically late).</li>
            <li><strong>Standard Class Blind Spot</strong>: The model achieves an ROC-AUC of <strong>0.530</strong> (near-random) on Standard Class, which accounts for <strong>59.7% of all order volume</strong>.</li>
            <li><strong>Academic Public Data</strong>: Western Americas/Europe retail orders; not validated against Indian domestic D2C logistics, PIN codes, or courier NDR workflows.</li>
            <li><strong>Advisory Support Only</strong>: Designed for human fulfillment prioritization, never for autonomous order cancellation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Alias for backwards compatibility
page_overview = page_home

# ===========================================================================
# Page 2: Score an Order
# ===========================================================================
def page_score_order(artifacts_arg=None):
    render_top_badge()

    st.markdown('<div class="rto-hero-title">Score an Order</div>', unsafe_allow_html=True)
    st.markdown('<div class="rto-hero-sub">Simulate pre-dispatch risk evaluation and explainable SHAP drivers for a single order.</div>', unsafe_allow_html=True)

    artifacts = artifacts_arg or get_artifacts()
    thr = artifacts["thresholds"]
    low_cut  = thr["tier_cutoffs"]["low_max"]   # 0.3749
    high_cut = thr["tier_cutoffs"]["high_min"]  # 0.6988

    # Preset selection
    preset_choice = st.radio(
        "Load Order Profile Preset:",
        [
            "Custom Input",
            "Preset: High Risk (Order #77188 - First Class)",
            "Preset: Medium Risk (Order #72160 - Standard Class)",
            "Preset: Low Risk (Order #67097 - Same Day)",
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

    if preset_choice == "Preset: High Risk (Order #77188 - First Class)":
        defaults.update({
            "shipping_mode": "First Class", "sched_days": 1, "payment_type": "CASH",
            "order_value": 187.76, "n_items": 2, "discount_rate": 0.16,
            "market": "LATAM", "department": "Footwear", "segment": "Consumer",
            "order_hour": 18, "order_dow": "Friday", "order_month": 12
        })
    elif preset_choice == "Preset: Medium Risk (Order #72160 - Standard Class)":
        defaults.update({
            "shipping_mode": "Standard Class", "sched_days": 4, "payment_type": "PAYMENT",
            "order_value": 276.92, "n_items": 4, "discount_rate": 0.07,
            "market": "Europe", "department": "Apparel", "segment": "Consumer",
            "order_hour": 14, "order_dow": "Friday", "order_month": 10
        })
    elif preset_choice == "Preset: Low Risk (Order #67097 - Same Day)":
        defaults.update({
            "shipping_mode": "Same Day", "sched_days": 0, "payment_type": "DEBIT",
            "order_value": 179.99, "n_items": 1, "discount_rate": 0.10,
            "market": "Europe", "department": "Apparel", "segment": "Consumer",
            "order_hour": 11, "order_dow": "Thursday", "order_month": 8
        })

    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    mode_day_map = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}

    col_form, col_result = st.columns([1.1, 1.2], gap="large")

    with col_form:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Order Parameters')

        with st.form("score_form"):
            shipping_mode = st.selectbox(
                "Shipping Mode",
                ["Standard Class", "First Class", "Second Class", "Same Day"],
                index=["Standard Class", "First Class", "Second Class", "Same Day"].index(defaults["shipping_mode"])
            )
            # Locked contracted SLA days
            sched_days = mode_day_map.get(shipping_mode, defaults["sched_days"])
            st.number_input(
                "Contracted Shipping Days (Locked)",
                min_value=0, max_value=6,
                value=int(sched_days),
                disabled=True,
                help="Fixed by carrier SLA: Same Day=0, First Class=1, Second Class=2, Standard Class=4."
            )

            c_f1, c_f2 = st.columns(2)
            with c_f1:
                payment_type = st.selectbox(
                    "Payment Type",
                    ["DEBIT", "TRANSFER", "PAYMENT", "CASH"],
                    index=["DEBIT", "TRANSFER", "PAYMENT", "CASH"].index(defaults["payment_type"])
                )
                order_value = st.number_input("Order Value ($)", min_value=5.0, max_value=5000.0, value=float(defaults["order_value"]), step=10.0)
                n_items = st.number_input("Number of Items", min_value=1, max_value=25, value=int(defaults["n_items"]), step=1)
            with c_f2:
                avg_discount = st.slider("Discount Rate", min_value=0.0, max_value=0.50, value=float(defaults["discount_rate"]), step=0.01)
                market = st.selectbox(
                    "Market",
                    ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"],
                    index=["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"].index(defaults["market"])
                )
                department = st.selectbox(
                    "Department",
                    ["Apparel", "Fan Shop", "Footwear", "Golf", "Outdoors", "Fitness", "Technology"],
                    index=["Apparel", "Fan Shop", "Footwear", "Golf", "Outdoors", "Fitness", "Technology"].index(defaults["department"]) if defaults["department"] in ["Apparel", "Fan Shop", "Footwear", "Golf", "Outdoors", "Fitness", "Technology"] else 0
                )

            c_f3, c_f4 = st.columns(2)
            with c_f3:
                segment = st.selectbox(
                    "Customer Segment",
                    ["Consumer", "Corporate", "Home Office"],
                    index=["Consumer", "Corporate", "Home Office"].index(defaults["segment"])
                )
                order_hour = st.slider("Order Hour (0-23)", min_value=0, max_value=23, value=int(defaults["order_hour"]))
            with c_f4:
                order_dow_str = st.selectbox("Day of Week", dow_names, index=dow_names.index(defaults["order_dow"]))
                order_dow = dow_names.index(order_dow_str)
                order_month = st.slider("Order Month", min_value=1, max_value=12, value=int(defaults["order_month"]))

            submit_btn = st.form_submit_button("🛡️ Compute Risk Score", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Build feature dict
    order_dict = {
        "Shipping_Mode": shipping_mode,
        "Days_for_shipment_scheduled": sched_days,
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

    result = predict.score_order(order_dict)
    score_val = result["risk_score"]
    disp_score = result.get("display_score", f"{score_val*100:.1f}%")
    tier = result["tier"]

    # Compute grouped SHAP
    df_row = predict._dict_to_df(order_dict)
    X_raw  = _select_feature_cols(df_row, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    X_enc  = artifacts["prep"].transform(X_raw)
    sv = artifacts["explainer"](X_enc)
    grouped_labels, grouped_values, val_strs = group_shap_values(sv.values[0], artifacts["feat_names"], order_dict)

    # Top driver name
    top_driver = grouped_labels[int(np.argmax(np.abs(grouped_values)))]

    with col_result:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)

        # Header with Tier Badge
        c_th1, c_th2 = st.columns([1.2, 1])
        with c_th1:
            st.markdown('<h3 style="margin:0;">Risk Assessment</h3>', unsafe_allow_html=True)
        with c_th2:
            if tier == "Low":
                st.markdown(f'<div style="text-align:right;"><span class="rto-badge rto-badge-low">🟢 LOW RISK ({disp_score})</span></div>', unsafe_allow_html=True)
            elif tier == "Medium":
                st.markdown(f'<div style="text-align:right;"><span class="rto-badge rto-badge-medium">🟠 MEDIUM RISK ({disp_score})</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:right;"><span class="rto-badge rto-badge-high">🔴 HIGH RISK ({disp_score})</span></div>', unsafe_allow_html=True)

        # Large Gauge
        gauge_fig = render_risk_gauge(score_val, low_cut, high_cut)
        st.plotly_chart(gauge_fig, width="stretch")

        # One plain-English summary sentence
        if score_val >= 0.5:
            summary_sentence = f"<b>{disp_score} chance of arriving late</b>, driven primarily by <b>{top_driver.lower()}</b>."
        else:
            summary_sentence = f"<b>{disp_score} chance of arriving late</b> (low probability), supported by favorable <b>{top_driver.lower()}</b>."

        st.markdown(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 16px; margin: 10px 0 16px 0; font-size:0.95rem; color:#1E293B;">
            💬 {summary_sentence}
        </div>
        """, unsafe_allow_html=True)

        # Horizontal bar chart of grouped drivers
        shap_bar_fig = render_grouped_shap_bar(grouped_labels, grouped_values, val_strs, top_k=6)
        st.plotly_chart(shap_bar_fig, width="stretch")

        st.markdown('</div>', unsafe_allow_html=True)

        # Expander for technical waterfall
        with st.expander("🔍 View Technical SHAP Waterfall & Raw Feature Vector"):
            waterfall_fig = render_shap_waterfall(
                order_dict,
                artifacts["prep"],
                artifacts["explainer"],
                artifacts["feat_names"]
            )
            st.pyplot(waterfall_fig)
            plt.close(waterfall_fig)

            st.caption("Waterfall illustrates additive shift from expected base log-odds to final predicted margin.")

# ===========================================================================
# Page 3: Real Test Orders
# ===========================================================================
def page_real_test_orders(artifacts_arg=None, test_df_arg=None, tables_arg=None):
    render_top_badge()

    st.markdown('<div class="rto-hero-title">Real Test Orders</div>', unsafe_allow_html=True)
    st.markdown('<div class="rto-hero-sub">Inspect unseen orders from the 2017-2018 holdout partition with model evaluation badges.</div>', unsafe_allow_html=True)

    artifacts = artifacts_arg or get_artifacts()
    _, test_df = get_datasets() if test_df_arg is None else (None, test_df_arg)
    tables = tables_arg or get_tables()
    demo_df = tables.get("demo_scores")

    st.markdown('<div class="rto-card">', unsafe_allow_html=True)
    st.markdown('### 1. Benchmark Demo Orders (5 Diverse Real Cases)')
    if demo_df is not None:
        styled_demo = demo_df.copy()
        styled_demo = styled_demo.rename(columns={
            "order_id": "Order ID",
            "score": "Calibrated Score",
            "display": "Display Score",
            "tier": "Tier",
            "flagged": "Flagged",
            "actual": "Actual Delivery",
            "shipping_mode": "Shipping Mode",
            "region": "Region",
            "payment": "Payment",
            "reasons": "Top Reason",
        })
        st.dataframe(
            styled_demo.style.format({"Calibrated Score": "{:.3f}"}),
            width="stretch"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="rto-card">', unsafe_allow_html=True)
    st.markdown('### 2. Interactive Order Inspection')

    col_flt1, col_flt2, col_flt3 = st.columns(3)
    with col_flt1:
        mode_flt = st.selectbox("Shipping Mode Filter:", ["All"] + sorted(test_df["Shipping_Mode"].dropna().unique().tolist()))
    with col_flt2:
        outcome_flt = st.selectbox("Actual Outcome:", ["All", "Late Only (1)", "On-time Only (0)"])
    with col_flt3:
        search_id = st.selectbox(
            "Select Benchmark or Filtered Order:",
            [77188, 67039, 72160, 73182, 76954] + test_df["Order Id"].head(50).tolist()
        )

    matched = test_df[test_df["Order Id"] == search_id]
    if not matched.empty:
        sel_row = matched.iloc[0].to_dict()
        actual_val = int(sel_row[TARGET_COL])
        score_info = predict.score_order(sel_row)
        score_val = score_info["risk_score"]
        disp_score = score_info.get("display_score", f"{score_val*100:.1f}%")
        is_flagged = score_info["flagged"]

        # Classification verification badge
        if is_flagged and actual_val == 1:
            badge_html = '<span class="rto-badge rto-badge-low">🟢 CORRECT &bull; True Positive (Late Detected)</span>'
        elif (not is_flagged) and actual_val == 0:
            badge_html = '<span class="rto-badge rto-badge-low">🟢 CORRECT &bull; True Negative (On-Time Cleared)</span>'
        elif (not is_flagged) and actual_val == 1:
            badge_html = '<span class="rto-badge rto-badge-medium">🟠 MISSED &bull; False Negative (Late Slipped Through)</span>'
        else:
            badge_html = '<span class="rto-badge rto-badge-high">🔴 FALSE ALARM &bull; False Positive (Unnecessary Friction)</span>'

        st.markdown("---")
        st.markdown(f"#### Order #{search_id} Diagnostic Card &bull; {badge_html}", unsafe_allow_html=True)

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Model Risk Score", disp_score, f"Tier: {score_info['tier']}")
        with col_d2:
            st.metric("Flagged Status", "FLAGGED (High Risk)" if is_flagged else "CLEARED (Standard Flow)")
        with col_d3:
            st.metric("Actual Delivery Outcome", "Late (1)" if actual_val == 1 else "On-time (0)")

        col_attr1, col_attr2 = st.columns([1, 1.2], gap="medium")
        with col_attr1:
            st.markdown("**Top 3 SHAP Reason Codes:**")
            for r in score_info["top_reasons"]:
                color = "#EF4444" if "raises risk" in r else "#10B981"
                icon = "▲" if "raises risk" in r else "▼"
                st.markdown(f"<span style='color:{color}; font-weight:700;'>{icon}</span> {r}", unsafe_allow_html=True)

            st.markdown("**Key Order Attributes:**")
            attrs = {
                "Shipping Mode": str(sel_row.get("Shipping_Mode")),
                "Scheduled Days": str(sel_row.get("Days_for_shipment_scheduled")),
                "Payment Method": str(sel_row.get("Type")),
                "Order Value ($)": f"${float(sel_row.get('order_value', 0)):.2f}",
                "Items Count": str(sel_row.get("n_items")),
                "Placement Hour": f"{int(sel_row.get('order_hour', 0)):02d}:00",
                "Market": str(sel_row.get("Market")),
                "Customer Segment": str(sel_row.get("Customer_Segment")),
            }
            attr_df = pd.DataFrame(list(attrs.items()), columns=["Attribute", "Value"])
            attr_df["Value"] = attr_df["Value"].astype(str)
            st.table(attr_df)

        with col_attr2:
            st.markdown("**Local SHAP Waterfall:**")
            wf_fig = render_shap_waterfall(sel_row, artifacts["prep"], artifacts["explainer"], artifacts["feat_names"])
            st.pyplot(wf_fig)
            plt.close(wf_fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# Page 4: Performance
# ===========================================================================
def page_performance(tables_arg=None):
    render_top_badge()

    st.markdown('<div class="rto-hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="rto-hero-sub">Empirical validation and test diagnostics across chronological holdout partitions.</div>', unsafe_allow_html=True)

    tables = tables_arg or get_tables()
    cal_p, y_te = get_test_predictions()

    tab_comp, tab_curves, tab_cal, tab_lift, tab_tiers = st.tabs([
        "📊 Model Comparison", "📈 Curves", "🎯 Calibration", "🚀 Lift & Gain", "🏷️ Risk Tiers"
    ])

    with tab_comp:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Model Comparison: ROC-AUC vs Baseline')

        comp_df = tables.get("model_comparison")
        if comp_df is not None:
            # Bar chart of test ROC-AUC
            fig_bar = px.bar(
                comp_df,
                x="model",
                y="roc_auc",
                color="model",
                color_discrete_map={
                    "LightGBM": "#4F46E5",
                    "XGBoost": "#6366F1",
                    "Random Forest": "#818CF8",
                    "Logistic Regression": "#94A3B8",
                    "Baseline LR (shipping+days)": "#CBD5E1",
                },
                labels={"roc_auc": "Test ROC-AUC", "model": "Candidate Model"},
                title="<b>Test Set Discrimination (ROC-AUC) Across Models</b>",
            )
            fig_bar.update_layout(
                yaxis=dict(range=[0.5, 0.85], gridcolor="#F1F5F9"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                height=320,
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, width="stretch")

            # Clean styled table
            st.markdown('#### Detailed Validation & Test Metrics')
            styled_comp = comp_df.rename(columns={
                "model": "Model",
                "split": "Split",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "f1": "F1",
                "recall": "Recall",
                "precision": "Precision",
                "brier": "Brier",
                "ece": "ECE",
                "precision_top10": "Top 10% Precision",
            })
            st.dataframe(
                styled_comp.style.format(precision=3),
                width="stretch"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_curves:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Diagnostic Curves (ROC & Precision-Recall)')
        fpr, tpr, _ = roc_curve(y_te, cal_p)
        prec, rec, _ = precision_recall_curve(y_te, cal_p)
        test_auc = roc_auc_score(y_te, cal_p)
        test_pr_auc = average_precision_score(y_te, cal_p)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Final LightGBM (AUC={test_auc:.3f})", line=dict(color="#4F46E5", width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess (AUC=0.500)", line=dict(color="#94A3B8", dash="dash")))
            fig_roc.update_layout(
                title="<b>ROC Curve (Test Set)</b>",
                xaxis=dict(title="False Positive Rate", gridcolor="#F1F5F9"),
                yaxis=dict(title="True Positive Rate", gridcolor="#F1F5F9"),
                height=340,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_roc, width="stretch")

        with col_c2:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"Final LightGBM (PR-AUC={test_pr_auc:.3f})", line=dict(color="#4F46E5", width=2.5)))
            fig_pr.add_trace(go.Scatter(x=[0, 1], y=[y_te.mean(), y_te.mean()], mode="lines", name=f"Baseline Rate ({y_te.mean():.3f})", line=dict(color="#94A3B8", dash="dash")))
            fig_pr.update_layout(
                title="<b>Precision-Recall Curve (Test Set)</b>",
                xaxis=dict(title="Recall", gridcolor="#F1F5F9"),
                yaxis=dict(title="Precision", gridcolor="#F1F5F9"),
                height=340,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_pr, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_cal:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Probability Calibration')
        st.markdown("""
        Reliability curve before vs after Isotonic Regression. Calibration reduces Expected Calibration Error (ECE)
        from **0.018** to **0.007**, aligning model probability scores with true empirical late rates.
        """)
        cal_img = OUTPUTS_DIR / "calibration.png"
        if cal_img.exists():
            st.image(str(cal_img), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_lift:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Decile Lift & Cumulative Gain')
        decile_df = tables.get("decile")
        if decile_df is not None:
            c_l1, c_l2 = st.columns([1.2, 1])
            with c_l1:
                fig_lift = px.bar(
                    decile_df,
                    x="decile",
                    y="lift",
                    text=decile_df["lift"].apply(lambda v: f"{v:.2f}x"),
                    title="<b>Lift by Predicted Risk Decile</b>",
                    labels={"decile": "Decile (1 = Highest Risk)", "lift": "Lift Factor"}
                )
                fig_lift.update_traces(marker_color="#4F46E5", textposition="outside")
                fig_lift.update_layout(yaxis=dict(gridcolor="#F1F5F9"), height=320, paper_bgcolor="white", plot_bgcolor="white")
                st.plotly_chart(fig_lift, width="stretch")
            with c_l2:
                st.markdown("""
                **Key Lift Highlights:**
                - **Decile 1 (Top 10%)**: Achieves **1.82x lift** over random dispatch (every order in Decile 1 was late).
                - **Top 3 Deciles**: Capture **48.3% of all late deliveries** while flagging only 30% of total volume.
                """)
                st.dataframe(
                    decile_df.rename(columns={
                        "decile": "Decile", "n_orders": "Orders", "late_orders": "Late Orders",
                        "empirical_late_rate": "Late Rate", "lift": "Lift", "cum_gain": "Cum Gain %"
                    }).style.format(precision=3),
                    width="stretch"
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_tiers:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Operational Risk Tiers (Validation p33 / p67 Cutoffs)')
        tier_data = [
            {"Risk Tier": "Low", "Score Cutoff": "score < 0.375", "Test Volume": "1,740 (17.6%)", "Actual Late Rate": "28.9%", "Late Captured": "9.3%", "Policy Action": "Standard automated dispatch; zero friction"},
            {"Risk Tier": "Medium", "Score Cutoff": "0.375 <= score < 0.699", "Test Volume": "4,850 (49.2%)", "Actual Late Rate": "42.2%", "Late Captured": "37.7%", "Policy Action": "Monitor carrier pickup SLA"},
            {"Risk Tier": "High", "Score Cutoff": "score >= 0.699", "Test Volume": "3,273 (33.2%)", "Actual Late Rate": "88.0%", "Late Captured": "53.0%", "Policy Action": "Operational Flag: Priority routing & proactive confirmation"}
        ]
        st.dataframe(pd.DataFrame(tier_data), width="stretch")

        st.markdown("""
        **Operational Threshold Policy Options:**
        - **Operational Flag (High Tier)**: `score >= 0.699` &bull; Precision: **88.4%** &bull; Recall: **52.2%** &bull; Flags 32.5% of volume.
        - **Precision ≥ 80% Strategy**: `score >= 0.399` &bull; Precision: **85.1%** &bull; Recall: **59.0%** &bull; Flags 38.2% of volume.
        - **Max-F1 Strategy**: `score >= 0.354` &bull; Precision: **58.6%** &bull; Recall: **95.5%** &bull; Flags 89.8% of volume.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

page_model_performance = page_performance

# ===========================================================================
# Page 5: Explainability
# ===========================================================================
def page_explainability():
    render_top_badge()

    st.markdown('<div class="rto-hero-title">Model Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="rto-hero-sub">Global feature attributions and continuous SHAP dependence insights.</div>', unsafe_allow_html=True)

    tab_glob, tab_bee, tab_dep = st.tabs([
        "🌐 Global Feature Importance", "🐝 Beeswarm Summary", "📈 SHAP Dependence (5 Continuous Features)"
    ])

    with tab_glob:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Global Importance by Parent Feature')
        st.markdown("""
        One-hot categorical encodings are consolidated into their parent business factors.
        Shipping parameters account for over 48% of total predictive mass.
        """)
        imp_img = OUTPUTS_DIR / "shap_importance.png"
        if imp_img.exists():
            st.image(str(imp_img), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_bee:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Global Beeswarm Distribution')
        st.markdown("Each point represents an order; color indicates whether raw feature value was high (red) or low (blue).")
        bee_img = OUTPUTS_DIR / "shap_summary.png"
        if bee_img.exists():
            st.image(str(bee_img), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_dep:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### SHAP Dependence on Key Continuous Features')

        d1, d2, d3, d4, d5 = st.tabs([
            "Order Hour", "Scheduled Days", "Order Value ($)", "Value per Item ($)", "Customer Late Rate"
        ])

        with d1:
            st.markdown("**Order Hour**: Sharp step-function for same-day dispatches placed at or after 12:00 PM.")
            p = OUTPUTS_DIR / "shap_dep_order_hour.png"
            if p.exists(): st.image(str(p), width="stretch")

        with d2:
            st.markdown("**Scheduled Days**: Contracted shipping SLA duration (0 to 4 days).")
            p = OUTPUTS_DIR / "shap_dep_scheduled_days.png"
            if p.exists(): st.image(str(p), width="stretch")

        with d3:
            st.markdown("**Order Value ($)**: Total basket monetary size.")
            p = OUTPUTS_DIR / "shap_dep_order_value.png"
            if p.exists(): st.image(str(p), width="stretch")

        with d4:
            st.markdown("**Value per Item ($)**: Basket density and item value.")
            p = OUTPUTS_DIR / "shap_dep_value_per_item.png"
            if p.exists(): st.image(str(p), width="stretch")

        with d5:
            st.markdown("**Customer Historical Late Rate**: Historical upstream fulfillment delay frequency for returning buyers.")
            p = OUTPUTS_DIR / "shap_dep_customer_prev_late_rate.png"
            if p.exists(): st.image(str(p), width="stretch")

        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# Page 6: Robustness
# ===========================================================================
def page_robustness(tables_arg=None):
    render_top_badge()

    st.markdown('<div class="rto-hero-title">Robustness & Stress Testing</div>', unsafe_allow_html=True)
    st.markdown('<div class="rto-hero-sub">Feature ablations, shipping-mode segment performance, and temporal stability.</div>', unsafe_allow_html=True)

    tables = tables_arg or get_tables()

    # Highlighted limitation card
    st.markdown("""
    <div class="rto-card-danger">
        <h4 style="margin-top:0; color:#991B1B; font-size:1.1rem;">CRITICAL LIMITATION: STANDARD CLASS AUC 0.53</h4>
        <p style="margin-bottom:0.3rem; font-size:1.02rem; color:#7F1D1D;">
            <strong>Standard Class comprises 59.7% of all order volume (5,888 test orders), but within this segment, ROC-AUC collapses to 0.5296.</strong>
        </p>
        <p style="margin-bottom:0; font-size:0.92rem; color:#991B1B; line-height:1.5;">
            While the model demonstrates strong discrimination on <em>First Class</em> (ROC-AUC 0.873) and <em>Same Day</em> (ROC-AUC 0.991), it offers near-zero predictive power inside Standard Class. Unscheduled ground carrier delays cannot be diagnosed from checkout-time signals alone.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_mode, tab_abl, tab_bt, tab_psi = st.tabs([
        "📦 Per-Shipping-Mode", "🧪 Feature Ablation", "⏳ Rolling Backtest", "🛡️ Drift (PSI)"
    ])

    with tab_mode:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Model Performance Across Shipping Modes')
        mode_data = [
            {"Shipping Mode": "First Class", "Test Orders": 1513, "Volume Share": "15.3%", "Full Model ROC-AUC": 0.873, "Full Model PR-AUC": 0.992, "Baseline ROC-AUC": 0.500, "Delta ROC-AUC": "+0.373"},
            {"Shipping Mode": "Same Day", "Test Orders": 539, "Volume Share": "5.5%", "Full Model ROC-AUC": 0.991, "Full Model PR-AUC": 0.984, "Baseline ROC-AUC": 0.500, "Delta ROC-AUC": "+0.491"},
            {"Shipping Mode": "Second Class", "Test Orders": 1923, "Volume Share": "19.5%", "Full Model ROC-AUC": 0.560, "Full Model PR-AUC": 0.794, "Baseline ROC-AUC": 0.500, "Delta ROC-AUC": "+0.060"},
            {"Shipping Mode": "Standard Class", "Test Orders": 5888, "Volume Share": "59.7%", "Full Model ROC-AUC": 0.530, "Full Model PR-AUC": 0.398, "Baseline ROC-AUC": 0.500, "Delta ROC-AUC": "+0.030"},
        ]
        st.dataframe(
            pd.DataFrame(mode_data).style.apply(
                lambda r: ['background-color: #FEE2E2' if r["Shipping Mode"] == "Standard Class" else '' for _ in r],
                axis=1
            ),
            width="stretch"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_abl:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Feature Ablation Study')
        st.markdown("Quantifying ROC-AUC degradation when individual feature families are excluded:")
        abl_df = tables.get("ablation")
        if abl_df is not None:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.dataframe(
                    abl_df.rename(columns={
                        "feature_set": "Feature Subset", "n_features": "Features (#)",
                        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1",
                        "recall": "Recall", "precision": "Precision", "brier": "Brier"
                    }).style.format(precision=3),
                    width="stretch"
                )
            with c2:
                abl_img = OUTPUTS_DIR / "ablation_chart.png"
                if abl_img.exists():
                    st.image(str(abl_img), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_bt:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Rolling-Origin Temporal Backtest (5 Expanding Folds)')
        back_df = tables.get("backtest")
        if back_df is not None:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.dataframe(
                    back_df.rename(columns={
                        "fold": "Fold", "n_train": "Train N", "n_test": "Test N",
                        "test_start": "Start Date", "test_end": "End Date",
                        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1",
                        "recall": "Recall", "precision": "Precision", "brier": "Brier"
                    }).style.format(precision=3),
                    width="stretch"
                )
                st.markdown("""
                - **Mean Fold ROC-AUC**: `0.770 ± 0.006`
                - **Mean Fold PR-AUC**: `0.814 ± 0.004`
                - Confirms temporal consistency across 2016-2017 without seasonal decay.
                """)
            with c2:
                bt_img = OUTPUTS_DIR / "backtest_plot.png"
                if bt_img.exists():
                    st.image(str(bt_img), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_psi:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Population Stability Index (PSI) Drift Check')
        psi_df = tables.get("psi")
        if psi_df is not None:
            st.dataframe(
                psi_df.rename(columns={"feature": "Feature", "psi": "PSI Value", "status": "Status"}).style.format({"PSI Value": "{:.4f}"}),
                width="stretch"
            )
            st.caption("PSI < 0.1 indicates zero significant covariate shift between training and test distributions.")
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# Page 7: Impact Simulator
# ===========================================================================
def page_impact_simulator(artifacts_arg=None, test_df_arg=None):
    render_top_badge()

    st.markdown('<div class="rto-hero-title">Impact Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="rto-hero-sub">Interactive what-if cost-benefit simulation for fulfillment triage policies.</div>', unsafe_allow_html=True)

    artifacts = artifacts_arg or get_artifacts()
    cal_p, y_te = get_test_predictions()

    col_sim_in, col_sim_out = st.columns([1, 1.25], gap="large")

    with col_sim_in:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown('### Business Inputs')

        monthly_orders = st.number_input("Monthly Order Volume", min_value=1000, max_value=200000, value=10000, step=1000)
        late_cost = st.number_input("Cost of Unresolved Late Delivery ($)", min_value=1.0, max_value=100.0, value=15.0, step=1.0, help="Customer friction, return handling, re-shipping cost.")
        intervention_cost = st.number_input("Proactive Intervention Cost ($)", min_value=0.25, max_value=20.0, value=2.50, step=0.25, help="Cost of SMS/call confirmation or carrier expedited handling.")
        rescue_rate = st.slider("Proactive Rescue Efficiency (%)", min_value=10, max_value=80, value=30, step=5, help="Share of flagged late deliveries saved by proactive action.") / 100.0

        st.markdown('### Triage Policy')
        policy_option = st.selectbox(
            "Select Decision Policy:",
            [
                "High Tier Flag (Score >= 0.699, Recommended)",
                "Precision >= 80% Strategy (Score >= 0.399)",
                "Max-F1 Strategy (Score >= 0.354)",
                "Custom Threshold Slider",
            ]
        )

        if policy_option == "High Tier Flag (Score >= 0.699, Recommended)":
            thresh = 0.6988
        elif policy_option == "Precision >= 80% Strategy (Score >= 0.399)":
            thresh = 0.3994
        elif policy_option == "Max-F1 Strategy (Score >= 0.354)":
            thresh = 0.3538
        else:
            thresh = st.slider("Custom Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.02)

        st.markdown('</div>', unsafe_allow_html=True)

    # Compute empirical rates from test partition at threshold
    pred_flag = (cal_p >= thresh).astype(int)
    flagged_share = float(pred_flag.mean())
    tp_count = int(((pred_flag == 1) & (y_te == 1)).sum())
    fp_count = int(((pred_flag == 1) & (y_te == 0)).sum())
    total_flagged_test = tp_count + fp_count
    precision = tp_count / max(total_flagged_test, 1)
    recall = tp_count / max(int((y_te == 1).sum()), 1)

    # Scale to monthly volume
    scaled_flagged = int(monthly_orders * flagged_share)
    scaled_late_caught = int(scaled_flagged * precision)
    scaled_rescued = int(scaled_late_caught * rescue_rate)
    gross_savings = scaled_rescued * late_cost
    total_intervention_cost = scaled_flagged * intervention_cost
    net_savings = gross_savings - total_intervention_cost
    roi = (net_savings / max(total_intervention_cost, 1)) * 100.0
    status_quo_cost = int(monthly_orders * float(y_te.mean()) * late_cost)
    managed_cost = status_quo_cost - gross_savings + total_intervention_cost

    with col_sim_out:
        st.markdown('<div class="rto-card">', unsafe_allow_html=True)
        st.markdown(f'### Projected Monthly Economics (Threshold: `{thresh:.3f}`)')

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Orders Flagged / Month", f"{scaled_flagged:,}", f"{flagged_share*100:.1f}% of volume")
            st.metric("Gross Friction Saved", f"${gross_savings:,.0f}", f"{scaled_rescued:,} orders rescued")
        with m2:
            st.metric("Total Intervention Cost", f"${total_intervention_cost:,.0f}", f"${intervention_cost:.2f} / flagged order")
            st.metric(
                "Net Monthly Savings",
                f"${net_savings:,.0f}",
                f"ROI: {roi:+.1f}%",
                delta_color="normal" if net_savings > 0 else "inverse"
            )

        # Plotly comparison
        cost_df = pd.DataFrame([
            {"Scenario": "Status Quo (No Model)", "Cost ($)": status_quo_cost, "Category": "Friction Loss"},
            {"Scenario": "With RTO Shield", "Cost ($)": managed_cost, "Category": "Managed Cost"},
        ])
        fig_cost = px.bar(
            cost_df,
            x="Scenario",
            y="Cost ($)",
            color="Scenario",
            color_discrete_map={"Status Quo (No Model)": "#94A3B8", "With RTO Shield": "#4F46E5"},
            text=cost_df["Cost ($)"].apply(lambda v: f"${v:,.0f}"),
            title=f"<b>Monthly Cost Comparison (Net Savings: ${net_savings:,.0f})</b>",
        )
        fig_cost.update_traces(textposition="outside")
        fig_cost.update_layout(yaxis=dict(gridcolor="#F1F5F9"), height=250, paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
        st.plotly_chart(fig_cost, width="stretch")

        st.caption(f"Precision: {precision*100:.1f}% &bull; Recall: {recall*100:.1f}% &bull; Baseline Test Late Rate: {y_te.mean()*100:.1f}%.")
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# Page 8: Limitations
# ===========================================================================
def page_limitations():
    render_top_badge()

    st.markdown('<div class="rto-hero-title">Limitations & Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="rto-hero-sub">Architectural constraints, dataset context, and ethical guidelines from MODEL_CARD.md.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="rto-card">
        <h3 style="margin-top:0; color:#1E293B;">1. The Proxy-Target Reality</h3>
        <p style="color:#475569; line-height:1.6;">
            <code>Late_delivery_risk</code> is an <strong>upstream fulfillment proxy</strong>, NOT Return-To-Origin (RTO).
        </p>
        <ul style="color:#475569; line-height:1.6;">
            <li>In real-world Indian D2C e-commerce, <strong>RTO occurs when a buyer refuses delivery, cancels at the doorstep, or commits COD fraud</strong>.</li>
            <li>In the DataCo dataset, the label reflects whether delivery duration exceeded the contracted carrier SLA.</li>
            <li>While fulfillment delays increase doorstep refusal propensity, this model evaluates shipping delay rather than doorstep refusal.</li>
        </ul>
    </div>

    <div class="rto-card">
        <h3 style="margin-top:0; color:#1E293B;">2. Near Rule-Based Label Construction</h3>
        <p style="color:#475569; line-height:1.6;">
            In the DataCo academic database, shipping mode and contracted delivery days heavily govern the ground truth label.
        </p>
        <ul style="color:#475569; line-height:1.6;">
            <li>Same Day orders placed after 12:00 PM are almost deterministically late due to synthetic dispatch cutoffs (94.4% late vs 0.0% before noon).</li>
            <li>The top decile is <strong>99.70% First Class</strong> orders whose tight 1-day SLA creates extreme label separation.</li>
            <li>100% precision in the top decile is an artefact of synthetic scheduling rules rather than artificial intelligence insight.</li>
        </ul>
    </div>

    <div class="rto-card">
        <h3 style="margin-top:0; color:#1E293B;">3. Standard Class Discrimination Blind Spot</h3>
        <p style="color:#475569; line-height:1.6;">
            Standard Class accounts for <strong>59.7% of all order volume</strong> (5,888 test orders).
            Within this majority segment, the model's ROC-AUC collapses to <strong>0.5296</strong>.
            Checkout-time signals cannot predict unobservable middle-mile and last-mile carrier ground delays.
        </p>
    </div>

    <div class="rto-card">
        <h3 style="margin-top:0; color:#1E293B;">4. Geographic Scope & Public Data Constraints</h3>
        <ul style="color:#475569; line-height:1.6;">
            <li>The DataCo dataset originates from Western cross-border retail across North America, LATAM, and Europe (Constante et al., 2019).</li>
            <li>It does not capture Indian domestic logistics complexities: PIN-code delivery tiers, Non-Delivery Report (NDR) IVR confirmation, or doorstep COD refusal.</li>
            <li>Must not be deployed into production without local retraining on actual courier tracking telemetry.</li>
        </ul>
    </div>

    <div class="rto-card">
        <h3 style="margin-top:0; color:#1E293B;">5. Advisory Governance & Ethical Guardrails</h3>
        <ul style="color:#475569; line-height:1.6;">
            <li><strong>Human-in-the-Loop</strong>: The model is strictly decision-support; it must never autonomously cancel orders or decline customer payments.</li>
            <li><strong>Fairness</strong>: Regional and buyer historical late rates must be monitored periodically to avoid compounding regional delivery disadvantages.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Setup (st.navigation + st.Page)
# ---------------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PAGES_CONFIG = {
    "Overview": [
        st.Page(page_home, title="Home", icon="🏠", default=True),
    ],
    "Decisioning & Operations": [
        st.Page(page_score_order, title="Score an Order", icon="🛡️"),
        st.Page(page_real_test_orders, title="Real Test Orders", icon="📦"),
        st.Page(page_impact_simulator, title="Impact Simulator", icon="💡"),
    ],
    "Model Evaluation & Governance": [
        st.Page(page_performance, title="Performance", icon="📊"),
        st.Page(page_explainability, title="Explainability", icon="🔍"),
        st.Page(page_robustness, title="Robustness", icon="⚙️"),
        st.Page(page_limitations, title="Limitations", icon="⚠️"),
    ],
}

nav = st.navigation(PAGES_CONFIG)
nav.run()
