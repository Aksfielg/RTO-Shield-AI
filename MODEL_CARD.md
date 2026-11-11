# Model Card — RTO Shield AI, Model 1

> Version 1.0 · November 2026  
> Dataset: Constante et al. 2019, Mendeley Data V5, DOI 10.17632/8gx2fvg2k6.5

---

## 1. Intended Use

| | |
|---|---|
| **Primary use** | Academic and exploratory prototype to evaluate whether pre-dispatch order features correlate with `Late_delivery_risk` in a multi-market supply chain. |
| **Research use cases** | Investigating pre-dispatch risk signals, evaluating time-based leakage safeguards, demonstrating explainable risk tiering with SHAP reason codes. |
| **Output** | Calibrated risk score ∈ [0, 1], risk tier (Low / Medium / High), boolean flag, and 3 plain-English reasons. |
| **NOT intended for** | Real-world Indian D2C e-commerce RTO prediction; autonomous order cancellation; individual customer blacklisting; credit/fraud profiling; production deployment without retraining on verified delivery outcomes. |
| **Proxy Target Caveat** | The target is `Late_delivery_risk`, NOT Return-To-Origin (RTO). The model predicts whether an order's delivery will be logged as late, which is an upstream operational proxy that may correlate with delivery friction, but does NOT represent package refusal, COD failure, or return to warehouse. |
| **Decision authority** | Advisory only. All flagged orders must be reviewed by a human fulfilment specialist before any operational change is made. |

---

## 2. Model Details

| | |
|---|---|
| **Model type** | LightGBM Gradient Boosted Trees (tuned) |
| **Framework** | LightGBM 4.7.0, scikit-learn 1.x |
| **Hyperparameters** | n_estimators=200, max_depth=4, learning_rate=0.009, subsample=0.60, colsample_bytree=0.59, min_child_weight=10, reg_lambda=0.50 |
| **Calibration** | Isotonic Regression fitted on validation predictions |
| **Threshold** | 0.3538 (maximises F1 on validation set) |
| **Input** | 27 pre-dispatch features (8 categorical, 19 numeric) → 252 encoded features after one-hot encoding |
| **Output** | Calibrated probability + tier + flag + SHAP reasons |

---

## 3. Training Data

| Property | Value |
|---|---|
| **Dataset** | DataCo Smart Supply Chain for Big Data Analysis |
| **Citation** | Constante, F., Silva, F., & Pereira, A. (2019). Mendeley Data V5. DOI: [10.17632/8gx2fvg2k6.5](https://doi.org/10.17632/8gx2fvg2k6.5) |
| **Licence** | CC BY 4.0 |
| **Raw size** | 180,519 rows × 53 columns |
| **Processed size** | 65,752 order-level rows × 32 columns |
| **Date range** | January 2015 – January 2018 |
| **Geography** | 164 countries; primarily North America, Europe, Latin America |
| **Industry** | Multi-category retail supply chain (electronics, clothing, sports goods) |

**Preprocessing:**
- Duplicate Order IDs aggregated to order level (sum/mean/max)
- NaN in `customer_prev_cash_share` (20,829 rows) → imputed with 0
- NaN in `days_since_prev_order` (20,652 rows) → imputed with median
- Post-dispatch columns dropped before feature engineering

---

## 4. Features

### Categorical (8)
| Feature | Description |
|---|---|
| `Type` | Payment type (CASH, DEBIT, PAYMENT, TRANSFER) |
| `Shipping_Mode` | First Class, Same Day, Second Class, Standard Class |
| `Market` | Geographic market (Africa, Europe, LATAM, Pacific Asia, USCA) |
| `Order_Region` | Sub-region of order |
| `Order_Country` | Destination country |
| `Customer_Segment` | Consumer, Corporate, Home Office |
| `Department_Name` | Product department |
| `main_item_category` | Primary product category |

### Numeric (19)
| Feature | Description |
|---|---|
| `Days_for_shipment_scheduled` | Contracted delivery window (pre-dispatch) |
| `order_value`, `n_items`, `avg_discount_rate`, `value_per_item` | Order size / value signals |
| `is_cash`, `is_weekend`, `is_night_order`, `high_discount_flag`, `n_distinct_categories` | Binary flags |
| `order_month`, `order_dow`, `order_hour` | Temporal features |
| `customer_prev_orders`, `customer_prev_late_rate` | Customer history (leakage-safe: uses only orders with `order_date < current`) |
| `country_prev_orders`, `country_prev_late_rate` | Country history (same safeguard) |
| `customer_prev_cash_share` | Share of customer's prior orders paid in cash |
| `days_since_prev_order` | Days since the customer's previous order |

### Forbidden features (never used)
`Days_for_shipping_real`, `Delivery Status`, `Order Status`, `shipping_date`, `Benefit per order`, `Order Profit Per Order`, `Order Item Profit Ratio`, `aux_target_cancel_fraud`, all PII columns.

---

## 5. Evaluation

### Train / Validation / Test Split

| Split | Rows | Date range | Positive rate |
|---|---|---|---|
| Train (70 %) | 46,026 | 2015-01-01 → 2017-03-16 | 54.85 % |
| Validation (15 %) | 9,863 | 2017-03-16 → 2017-09-05 | 54.50 % |
| Test (15 %) | 9,863 | 2017-09-05 → 2018-01-31 | 55.05 % |

Splits are **strictly chronological** with no overlap (verified by `tests/test_pipeline.py::test_chrono_split_no_overlap`).

### Model Comparison (TEST set, threshold = 0.5)

| Model | ROC-AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|
| Baseline LR (shipping+days) | 0.7291 | 0.7503 | 0.658 | 0.1965 |
| Logistic Regression | 0.7497 | 0.8200 | 0.674 | 0.1949 |
| Random Forest | 0.7653 | 0.8312 | 0.705 | 0.1882 |
| XGBoost | 0.7658 | 0.8351 | 0.708 | 0.1864 |
| **LightGBM (default)** | **0.7678** | **0.8354** | **0.696** | **0.1833** |

### Final Model Metrics (LightGBM tuned, TEST, threshold = 0.3538)

| Metric | TEST | 95 % Bootstrap CI (1000 resamples) |
|---|---|---|
| ROC-AUC | 0.7713 | [0.7622, 0.7801] |
| PR-AUC | 0.8134 | [0.8047, 0.8215] |
| F1 | 0.7259 | — |
| Recall | 0.9549 | — |
| Precision | 0.5855 | — |
| Brier score | 0.1831 | — |
| ECE (after calib.) | 0.0067 | — |

**Δ ROC-AUC vs. baseline, 95 % CI:** [+0.0365, +0.0477] (statistically significant improvement).

**Multi-seed stability (10 random seeds):** ROC-AUC 0.7706 ± 0.0011, PR-AUC 0.8134 ± 0.0034.

---

## 6. Calibration

Isotonic Regression was fitted on validation set predictions to align predicted probabilities with observed empirical late rates.

### Default LightGBM (Step 3 Threshold & Calibration Analysis)
| Metric | Before calibration | After calibration |
|---|---|---|
| Brier score | 0.1833 | 0.1831 |
| ECE | 0.0182 | 0.0042 |

### Final Tuned LightGBM (Step 5 Multi-Seed & Final Model)
| Metric | Before calibration | After calibration |
|---|---|---|
| Brier score | 0.1868 | 0.1831 |
| ECE | 0.0495 | 0.0067 |

Reliability curves saved to `outputs/calibration.png`. Both default and tuned models demonstrate significant calibration improvement following isotonic regression.

---

## 7. Risk Tiers (from thresholds.json)

Tier cutoffs derived from the **33rd and 67th percentiles of calibrated validation scores** of the final tuned LightGBM model:

| Tier | Score range | Recommended action | Test Late Rate | Test Volume Share |
|---|---|---|---|---|
| Low | < 0.3749 | Standard dispatch; automated processing | 28.91% | 17.6% (1,740) |
| Medium | 0.3749 – 0.6988 | Monitor carrier pickup SLA | 42.23% | 49.2% (4,850) |
| High | ≥ 0.6988 | **Operational Flag**: Proactive intervention / route review | 87.96% | 33.2% (3,273) |

> **Threshold Strategy Options:**
> - **Operational Flag (High Tier)**: `score >= 0.6988` (Precision: 88.36%, Recall: 52.15%, F1: 0.6559, Flags: 32.50%).
> - **High Precision Strategy (Precision ≥ 80%)**: `score >= 0.3994` (Precision: 85.06%, Recall: 59.02%, F1: 0.6969, Flags: 38.20%).
> - **Max-F1 Strategy**: `score >= 0.3538` (Precision: 58.55%, Recall: 95.49%, F1: 0.7259, Flags: 89.78%).

---

## 8. Top Features (SHAP, 500-row test sample)

| Rank | Feature | Mean \|SHAP\| | % of total | Directional Note |
|---|---|---|---|---|
| 1 | Shipping_Mode_Standard Class | 0.3920 | ~33.1 % | Lowers risk vs First Class |
| 2 | Days_for_shipment_scheduled | 0.1990 | ~16.8 % | Longer scheduled window lowers risk |
| 3 | Shipping_Mode_First Class | 0.1825 | ~15.4 % | Substantially raises risk |
| 4 | Type_TRANSFER | 0.1119 | ~9.4 % | Lowers late rate (48.2% vs ~57% for other payment types) |
| 5 | order_hour | 0.0892 | ~7.5 % | Step function for Same Day dispatches |

**Key finding:** `order_hour` is near-deterministic for Same Day shipping (see proxy-target caveat below).

---

## 9. Limitations

> [!WARNING]
> **Proxy target** — `Late_delivery_risk` is a pre-computed flag in the dataset, not raw measured delivery outcomes. The model learns to replicate this rule-based flag, not actual late-delivery propensity.

> [!CAUTION]
> **Near rule-based label** — `order_hour` alone explains ~94 % of Same Day late-delivery outcomes. Precision@top10% = 100 % for the top decile, which consists of 99.7% First Class orders with tight 1-day SLAs, an artefact of synthetic label construction rather than genuine courier insight.

| Limitation | Detail |
|---|---|
| Public logistics data | DataCo is a simulated/anonymised academic dataset, not real D2C e-commerce transactions |
| Geographic mismatch | Dataset is North America / LATAM centric; not validated on Indian last-mile logistics |
| Not Indian D2C RTO | RTO in Indian e-commerce involves courier SLAs, PIN-code reachability, COD fraud — none captured here |
| Standard Class discrimination | Per-mode ROC-AUC: Same Day 0.9914, First Class 0.8732, **Standard Class 0.5296** (near-random) |
| Shipping mode concentration | Standard Class = 60 %+ of orders; weakest discrimination in the highest-volume segment |
| Secondary target robustness only | `aux_target_cancel_fraud` (ROC-AUC 0.8794) is an exploratory check, not a production label |
| No real-time features | Live courier tracking, weather, traffic data not incorporated |

---

## 10. Ethical Notes

| Dimension | Detail |
|---|---|
| **PII** | No personally identifiable information used (names, emails, passwords, street addresses dropped in `01_prepare.py`) |
| **Customer ID** | `Customer_Id` is used only internally for history aggregation; never passed as a model feature |
| **Fairness** | No demographic or protected-class attributes are features; model is not evaluated for demographic parity |
| **Transparency** | SHAP explanations provided for every prediction; plain-English reason codes exposed via `get_reason_codes()` |
| **Human oversight** | Model flags orders for human review; it does not autonomously cancel or reroute shipments |
| **Auditability** | All thresholds, calibration parameters, and feature lists are saved in `models/`; full reproducibility via `run_all.py` |

---

## 11. Files

| File | Description |
|---|---|
| `models/final_model.joblib` | Tuned LightGBM model |
| `models/preprocessor.joblib` | Fitted ColumnTransformer (OHE + StandardScaler) |
| `models/calibrator.joblib` | Fitted Isotonic Regression calibrator |
| `models/thresholds.json` | Thresholds and tier cutoffs |
| `outputs/final_results.csv` | Multi-seed metrics |
| `outputs/final_results_meta.json` | Bootstrap CIs and best hyperparameters |
| `outputs/metrics_model_comparison.csv` | 5-model comparison |
| `outputs/ablation_results.csv` | Feature ablation study |
| `outputs/backtest_results.csv` | Rolling-origin backtest |
| `outputs/report_tables.md` | All tables in paste-ready form |

---

*Model Card follows the Google Model Cards framework (Mitchell et al., 2019).*

