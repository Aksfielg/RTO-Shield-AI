"""
test_pages_direct.py
====================
Direct unit validation of every page renderer function in app.py.
Verifies that all data structures, artifacts, image paths, tables,
gauges, and SHAP waterfalls execute without exceptions.
"""

import sys
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["USERPROFILE"] = str(ROOT_DIR)
os.environ["STREAMLIT_CONFIG_DIR"] = str(ROOT_DIR / ".streamlit")

import app
import predict


def test_app_pages_and_components():
    """Verify that all 7 dashboard pages and components execute cleanly."""
    artifacts = app.load_model_artifacts()
    df, test_df = app.load_datasets()
    tables = app.load_output_tables()

    # 1. Overview
    app.page_overview()

    # 2. Score an order
    thr = artifacts["thresholds"]
    low_cut = thr["tier_cutoffs"]["low_max"]
    high_cut = thr["tier_cutoffs"]["high_min"]
    flag_cut = thr["thresholds"]["final_tuned_max_f1"]

    sample_order = {
        "Shipping_Mode": "First Class",
        "Days_for_shipment_scheduled": 1,
        "Type": "CASH",
        "order_value": 250.0,
        "n_items": 3,
        "avg_discount_rate": 0.10,
        "value_per_item": 83.33,
        "Market": "Europe",
        "Department_Name": "Apparel",
        "Customer_Segment": "Consumer",
        "order_hour": 16,
        "order_dow": 3,
        "order_month": 7,
        "is_cash": 1,
        "is_weekend": 0,
        "is_night_order": 0,
        "high_discount_flag": 0,
        "n_distinct_categories": 1,
    }
    score_res = predict.score_order(sample_order)
    assert "risk_score" in score_res
    assert "top_reasons" in score_res
    assert len(score_res["top_reasons"]) == 3

    gauge = app.render_risk_gauge(score_res["risk_score"], low_cut, high_cut, flag_cut)
    assert gauge is not None

    waterfall_fig = app.render_shap_waterfall(
        sample_order,
        artifacts["prep"],
        artifacts["explainer"],
        artifacts["feat_names"]
    )
    assert waterfall_fig is not None
    app.plt.close(waterfall_fig)

    app.page_score_order(artifacts)

    # 3. Real test orders
    app.page_real_test_orders(artifacts, test_df, tables)

    # 4. Model performance
    app.page_model_performance(tables)

    # 5. Explainability
    app.page_explainability()

    # 6. Robustness
    app.page_robustness(tables)

    # 7. Impact Simulator
    app.page_impact_simulator()

    # 8. Limitations
    app.page_limitations()

