# Model Card — RTO Shield AI, Model 1

<!-- GENERATED FILE: edit templates/MODEL_CARD.md.tmpl, then run `python src/11_render_docs.py`. -->

> Metrics generated 2026-11-12 22:09:44 from `outputs/final_metrics.json` and `models/thresholds.json`.
> Dataset: Constante et al. 2019, Mendeley Data V5, DOI 10.17632/8gx2fvg2k6.5

---

## 1. Intended Use

| | |
|---|---|
| **Primary use** | Academic and exploratory prototype to evaluate whether pre-dispatch order features correlate with `Late_delivery_risk` in a multi-market supply chain. |
| **Research use cases** | Investigating pre-dispatch risk signals, evaluating time-based leakage safeguards, demonstrating explainable risk tiering with SHAP reason codes. |
| **Output** | Calibrated risk score in [0, 1], risk tier (Low / Medium / High), boolean flag (= High tier), and 3 plain-English reasons. |
| **NOT intended for** | Real-world Indian D2C e-commerce RTO prediction; autonomous order cancellation; individual customer blacklisting; credit/fraud profiling; production deployment without retraining on verified delivery outcomes. |
| **Proxy target caveat** | The target is `Late_delivery_risk`, NOT Return-To-Origin (RTO). It is an upstream operational proxy and does NOT represent package refusal, COD failure, or return to warehouse. |
| **Claims not made** | No result in this card measures RTO, doorstep-refusal or Indian COD performance; "cash" payment type is not a verified COD indicator. |
| **Decision authority** | Advisory only. All flagged orders must be reviewed by a human fulfilment specialist before any operational change is made. |

---

## 2. Model Details

| | |
|---|---|
| **Model type** | LightGBM gradient boosted trees (tuned) |
| **Framework** | LightGBM, scikit-learn |
| **Hyperparameters** | n_estimators=200, max_depth=4, learning_rate=0.0093, subsample=0.60, colsample_bytree=0.59, min_child_weight=10, reg_lambda=0.50 |
| **Calibration** | Isotonic regression fitted on validation predictions |
| **Tiers / flag** | Low / Medium / High; **flagged = High tier** (score >= 0.7423) |
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
| **Date range** | 2015-01-01 to 2018-01-31 |
| **Geography** | 164 countries; primarily North America, Europe, Latin America |
| **Industry** | Multi-category retail supply chain (electronics, clothing, sports goods) |

**Preprocessing:** duplicate Order IDs aggregated to order level (sum/mean/max); missing customer-history values imputed in the preprocessor (0 for cash share, median for days since previous order); post-dispatch columns dropped before feature engineering.

---

## 4. Features

### Categorical
`Type`, `Shipping_Mode`, `Market`, `Order_Region`, `Order_Country`, `Customer_Segment`, `Department_Name`, `main_item_category`.

### Numeric
`Days_for_shipment_scheduled`; `order_value`, `n_items`, `avg_discount_rate`, `value_per_item`; flags `is_cash`, `is_weekend`, `is_night_order`, `high_discount_flag`, `n_distinct_categories`; temporal `order_month`, `order_dow`, `order_hour`; leakage-safe history `customer_prev_orders`, `customer_prev_late_rate`, `country_prev_orders`, `country_prev_late_rate`, `customer_prev_cash_share`, `days_since_prev_order` (only orders whose outcome was already known before the current order date are counted).

### Forbidden features (never used)
`Days_for_shipping_real`, `Delivery Status`, `Order Status`, `shipping_date`, `Benefit per order`, `Order Profit Per Order`, `Order Item Profit Ratio`, `aux_target_cancel_fraud`, all PII columns.

---

## 5. Evaluation

### Train / Validation / Test split (strictly chronological)

| Split | Orders | Date range | Positive rate |
|---|---|---|---|
| Train | 46,026 | 2015-01-01 -> 2017-03-16 | 54.85% |
| Validation | 9,863 | 2017-03-16 -> 2017-09-05 | 54.50% |
| Test | 9,863 | 2017-09-05 -> 2018-01-31 | 55.05% |

### Model comparison (TEST, threshold = 0.5)

| Model | ROC-AUC | PR-AUC | Brier | Precision | Recall | F1 | Accuracy | Balanced acc. | Specificity | MCC | Prec@Top10% | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline LR (shipping+days) | 0.7291 | 0.7503 | 0.1965 | 0.8490 | 0.5372 | 0.6580 | 0.6926 | 0.7101 | 0.8829 | 0.4386 | 0.9504 | 0.0051 |
| Logistic Regression | 0.7497 | 0.8200 | 0.1949 | 0.8440 | 0.5611 | 0.6741 | 0.7013 | 0.7171 | 0.8730 | 0.4483 | 0.9970 | 0.0190 |
| Random Forest | 0.7653 | 0.8312 | 0.1882 | 0.7939 | 0.6333 | 0.7046 | 0.7076 | 0.7159 | 0.7986 | 0.4329 | 1.0000 | 0.0545 |
| XGBoost | 0.7658 | 0.8351 | 0.1864 | 0.7754 | 0.6510 | 0.7078 | 0.7040 | 0.7100 | 0.7690 | 0.4191 | 1.0000 | 0.0401 |
| LightGBM (default) | 0.7678 | 0.8354 | 0.1833 | 0.8531 | 0.5882 | 0.6963 | 0.7175 | 0.7321 | 0.8759 | 0.4758 | 1.0000 | 0.0182 |

### Final model metrics (TEST)

> **Metric conventions.** ROC-AUC, PR-AUC, lifts and curves use the RAW (uncalibrated) model scores, because isotonic calibration collapses scores into ties and distorts rankings. Calibrated scores are used only for calibration (Brier, ECE), tiers, thresholds and every threshold-based metric (precision, recall, F1, accuracy, balanced accuracy, specificity, MCC).

| Metric | TEST | 95% bootstrap CI |
|---|---|---|
| ROC-AUC (raw scores) | 0.7716 | [0.7625, 0.7804] |
| PR-AUC (raw scores) | 0.8362 | [0.8277, 0.8440] |
| PR-AUC lift over prevalence | 1.519 | - |
| F1 @ max-F1 threshold (0.3538) | 0.7259 | - |
| Precision @ max-F1 threshold | 0.5855 | - |
| Recall @ max-F1 threshold | 0.9549 | - |
| Accuracy @ max-F1 threshold | 0.6031 | - |
| Balanced accuracy @ max-F1 threshold | 0.5635 | - |
| Specificity @ max-F1 threshold | 0.1721 | - |
| MCC @ max-F1 threshold | 0.2086 | - |
| Brier (calibrated) | 0.1831 | - |
| ECE (calibrated) | 0.0067 | - |

**Δ ROC-AUC vs. baseline, 95% CI:** [0.0364, 0.0483].

**Multi-seed stability (10 seeds):** ROC-AUC 0.7712 ± 0.0007, PR-AUC 0.8363 ± 0.0006.

**Label-shuffle sanity check:** | Check | Validation ROC-AUC | Test ROC-AUC | Status |
|---|---|---|---|
| Label shuffle (expect ~0.50) | 0.4950 | 0.4959 | PASS |

### Per shipping mode (TEST)

| Shipping mode | n (test) | Share | Prevalence | ROC-AUC | ROC-AUC 95% CI | PR-AUC | PR-AUC lift | Signal |
|---|---|---|---|---|---|---|---|---|
| Same Day | 539 | 5.5% | 50.3% | 0.9901 | [0.9836, 0.9950] | 0.9896 | 1.97x | informative |
| First Class | 1,513 | 15.3% | 95.0% | 0.8709 | [0.8503, 0.8904] | 0.9929 | 1.04x | informative |
| Second Class | 1,923 | 19.5% | 76.9% | 0.5666 | [0.5347, 0.5968] | 0.8044 | 1.05x | limited |
| Standard Class | 5,888 | 59.7% | 38.1% | 0.5292 | [0.5138, 0.5440] | 0.3983 | 1.05x | limited |

Signal = **limited** when the ROC-AUC 95% bootstrap CI lower bound is below 0.55 (which also covers any CI that includes 0.5); otherwise informative. Limited modes: Second Class, Standard Class. PR-AUC lift = PR-AUC divided by the mode's prevalence (1.0x = no better than the base rate; a mode with very high prevalence cannot show much lift).

### Feature ablation (TEST)

| Feature subset | # feat. | ROC-AUC | ROC-AUC 95% CI | PR-AUC | Brier | Own thr. | F1 | Precision | Recall | Accuracy | Bal. acc. | Spec. | MCC | Delta ROC-AUC vs all [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All features | 27 | 0.7716 | [0.7625, 0.7804] | 0.8362 | 0.1831 | 0.354 | 0.726 | 0.586 | 0.955 | 0.603 | 0.564 | 0.172 | 0.209 | - |
| No shipping mode/days | 25 | 0.5593 | [0.5473, 0.5695] | 0.5949 | 0.2452 | 0.251 | 0.710 | 0.550 | 0.999 | 0.550 | 0.500 | 0.001 | -0.000 | +0.2123 [+0.2012, +0.2236] * |
| No history features | 21 | 0.7708 | [0.7614, 0.7800] | 0.8362 | 0.1847 | 0.337 | 0.725 | 0.583 | 0.958 | 0.599 | 0.559 | 0.160 | 0.200 | +0.0008 [-0.0017, +0.0033] |
| No time features | 22 | 0.7408 | [0.7310, 0.7501] | 0.8153 | 0.1951 | 0.348 | 0.712 | 0.567 | 0.958 | 0.574 | 0.531 | 0.104 | 0.121 | +0.0308 [+0.0271, +0.0347] * |
| No payment/is_cash | 25 | 0.7551 | [0.7453, 0.7644] | 0.8162 | 0.1847 | 0.356 | 0.722 | 0.565 | 1.000 | 0.576 | 0.528 | 0.057 | 0.180 | +0.0165 [+0.0108, +0.0221] * |
| Shipping only (baseline) | 2 | 0.7291 | [0.7206, 0.7374] | 0.7503 | 0.1963 | 0.010 | 0.710 | 0.550 | 1.000 | 0.550 | 0.500 | 0.000 | 0.000 | +0.0425 [+0.0364, +0.0483] * |

Threshold metrics use each ablated model's OWN max-F1 threshold chosen on validation. Delta = ROC-AUC(all features) - ROC-AUC(ablated), paired bootstrap, * = CI excludes 0. History features add NO significant gain: ROC-AUC(all) - ROC-AUC(no history) = +0.0008 (95% paired-bootstrap CI [-0.0017, +0.0033] includes 0 or is negative); PR-AUC is 0.8362 with history vs 0.8362 without.

---

## 6. Calibration (final tuned model)

Isotonic regression was fitted on validation-set predictions of the final tuned model.

| Metric (TEST, final tuned model) | Before calibration (raw) | After isotonic calibration |
|---|---|---|
| Brier score | 0.1860 | 0.1831 |
| ECE | 0.0495 | 0.0067 |

The validation ECE after calibration is ~0 by construction (the calibrator is fitted on that set); the test-set figure above is the honest one. Reliability curves: `outputs/calibration.png`.

---

## 7. Risk Tiers and Policies

Cutoffs are **midpoints of low-density gaps** between calibrated-score clusters (41 distinct scores on test), chosen on validation scores only so that no cluster holding more than 2% of validation orders is split between tiers, then verified on the test set.

| Tier | Score range | Test orders | Share | Late rate | Late captured | Suggested handling |
|---|---|---|---|---|---|---|
| Low | score < 0.4657 | 6,104 | 61.9% | 36.5% | 41.0% | Standard dispatch with routine monitoring. |
| Medium | 0.4657 <= score < 0.7423 | 554 | 5.6% | 66.6% | 6.8% | Monitor the carrier pickup SLA. |
| High | score >= 0.7423 | 3,205 | 32.5% | 88.4% | 52.2% | Operational flag: review routing and confirm proactively. |

- **Low** - Lower risk: 37% of these orders were late in held-out test data (overall: 55%).
- **Medium** - Elevated risk: 67% of these orders were late in held-out test data (overall: 55%).
- **High** - Highest risk: 88% of these orders were late in held-out test data (overall: 55%).

### Policies (TEST)

| Policy | Threshold | Flagged | Precision | Recall | F1 | Accuracy | Balanced acc. | Specificity | MCC |
|---|---|---|---|---|---|---|---|---|---|
| Flag all | - | 100.00% (9,863) | 0.5505 | 1.0000 | 0.7101 | 0.5505 | 0.5000 | 0.0000 | 0.0000 |
| High tier (Operational Flag) | 0.7423 | 32.50% (3,205) | 0.8836 | 0.5215 | 0.6559 | 0.6988 | 0.7187 | 0.9159 | 0.4646 |
| Precision >= 0.80 | 0.4657 | 38.11% (3,759) | 0.8516 | 0.5895 | 0.6967 | 0.7174 | 0.7318 | 0.8741 | 0.4749 |
| Max-F1 | 0.3538 | 89.78% (8,855) | 0.5855 | 0.9549 | 0.7259 | 0.6031 | 0.5635 | 0.1721 | 0.2086 |
| Cost-minimising (FN=2xFP) | 0.3320 | 92.21% (9,095) | 0.5777 | 0.9676 | 0.7234 | 0.5927 | 0.5506 | 0.1335 | 0.1877 |

### Threshold stability (TEST, threshold ±0.01 / ±0.02)

Maximum change in flagged share: High tier: 0.00 pp; Precision >= 0.80: 0.00 pp; Max-F1: 2.46 pp (limit 5 pp).

| Policy | Shift | Threshold | Flagged share | Precision | Recall | F1 | Accuracy | Balanced acc. | Specificity | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| High tier | -0.02 | 0.7223 | 32.50% | 0.8836 | 0.5215 | 0.6559 | 0.6988 | 0.7187 | 0.9159 | 0.4646 |
| High tier | -0.01 | 0.7323 | 32.50% | 0.8836 | 0.5215 | 0.6559 | 0.6988 | 0.7187 | 0.9159 | 0.4646 |
| High tier | +0.00 | 0.7423 | 32.50% | 0.8836 | 0.5215 | 0.6559 | 0.6988 | 0.7187 | 0.9159 | 0.4646 |
| High tier | +0.01 | 0.7523 | 32.50% | 0.8836 | 0.5215 | 0.6559 | 0.6988 | 0.7187 | 0.9159 | 0.4646 |
| High tier | +0.02 | 0.7623 | 32.50% | 0.8836 | 0.5215 | 0.6559 | 0.6988 | 0.7187 | 0.9159 | 0.4646 |
| Precision >= 0.80 | -0.02 | 0.4457 | 38.11% | 0.8516 | 0.5895 | 0.6967 | 0.7174 | 0.7318 | 0.8741 | 0.4749 |
| Precision >= 0.80 | -0.01 | 0.4557 | 38.11% | 0.8516 | 0.5895 | 0.6967 | 0.7174 | 0.7318 | 0.8741 | 0.4749 |
| Precision >= 0.80 | +0.00 | 0.4657 | 38.11% | 0.8516 | 0.5895 | 0.6967 | 0.7174 | 0.7318 | 0.8741 | 0.4749 |
| Precision >= 0.80 | +0.01 | 0.4757 | 38.11% | 0.8516 | 0.5895 | 0.6967 | 0.7174 | 0.7318 | 0.8741 | 0.4749 |
| Precision >= 0.80 | +0.02 | 0.4857 | 38.11% | 0.8516 | 0.5895 | 0.6967 | 0.7174 | 0.7318 | 0.8741 | 0.4749 |
| Max-F1 | -0.02 | 0.3338 | 92.21% | 0.5777 | 0.9676 | 0.7234 | 0.5927 | 0.5506 | 0.1335 | 0.1877 |
| Max-F1 | -0.01 | 0.3438 | 92.01% | 0.5782 | 0.9663 | 0.7235 | 0.5933 | 0.5514 | 0.1365 | 0.1886 |
| Max-F1 | +0.00 | 0.3538 | 89.78% | 0.5855 | 0.9549 | 0.7259 | 0.6031 | 0.5635 | 0.1721 | 0.2086 |
| Max-F1 | +0.01 | 0.3638 | 89.76% | 0.5855 | 0.9545 | 0.7258 | 0.6029 | 0.5633 | 0.1721 | 0.2078 |
| Max-F1 | +0.02 | 0.3738 | 89.75% | 0.5855 | 0.9545 | 0.7258 | 0.6030 | 0.5634 | 0.1723 | 0.2080 |

---

## 8. Top Features (SHAP, 500-row test sample)

| Rank | Feature | Mean |SHAP| | Share of total |
|---|---|---|---|
| 1 | Shipping_Mode_Standard Class | 0.3920 | 33.1% |
| 2 | Days_for_shipment_scheduled | 0.1990 | 16.8% |
| 3 | Shipping_Mode_First Class | 0.1825 | 15.4% |
| 4 | Type_TRANSFER | 0.1119 | 9.4% |
| 5 | order_hour | 0.0892 | 7.5% |

By business factor (one-hot columns grouped):

| Business factor | Mean |SHAP| | Share of total |
|---|---|---|
| Shipping mode | 0.5631 | 54.4% |
| Scheduled shipping days | 0.1990 | 19.2% |
| Payment type | 0.1327 | 12.8% |
| Hour of day | 0.0892 | 8.6% |
| Value per item | 0.0107 | 1.0% |
| Order value | 0.0090 | 0.9% |

Shipping mode and scheduled days together account for 73.6% of total attribution. The order hour acts as a step function for Same Day shipments (late rate 0.0% before 12:00 vs 95.4% from 12:00).

---

## 9. Limitations and Verification Findings

> [!WARNING]
> **Proxy target** — `Late_delivery_risk` is a pre-computed flag in the dataset, not raw measured delivery outcomes. The model learns to replicate this rule-based flag.

> [!CAUTION]
> **TRANSFER is a labelling artifact.** All 2,855 CANCELED / SUSPECTED_FRAUD orders are TRANSFER payments (15.8% of TRANSFER orders vs 0.0% of other types) and all carry `Late_delivery_risk = 0` (CANCELED 0.0%, SUSPECTED_FRAUD 0.0%, other statuses 57.3%). Controlling for shipping mode, the TRANSFER late rate differs from other payment types by -9.1 pp; with cancel/fraud orders removed the difference is -0.0 pp. The final model's test ROC-AUC is 0.7716 on all orders and 0.7750 without them.

> [!CAUTION]
> **Near rule-based label** — the highest decile is dominated by First Class orders (below); precision in the top decile is 100%, an artefact of how the label was constructed rather than genuine courier insight.

| Shipping mode | Top-10% count | Top-10% share | Test count | Overall share | Late rate |
|---|---|---|---|---|---|
| First Class | 986 | 100.00% | 1,513 | 15.34% | 95.0% |
| Same Day | 0 | 0.00% | 539 | 5.46% | 50.3% |
| Second Class | 0 | 0.00% | 1,923 | 19.50% | 76.9% |
| Standard Class | 0 | 0.00% | 5,888 | 59.70% | 38.1% |

| Limitation | Detail |
|---|---|
| Public logistics data | DataCo is a simulated/anonymised academic dataset, not real D2C e-commerce transactions |
| Geographic mismatch | Dataset is North America / LATAM centric; not validated on Indian last-mile logistics |
| Not Indian D2C RTO | RTO in Indian e-commerce involves courier SLAs, PIN-code reachability, COD fraud — none captured here |
| Standard Class discrimination | Standard Class ROC-AUC 0.5292 (95% CI 0.5138 to 0.5440, signal limited) on 59.7% of orders |
| Segment headroom | A segment-only model does not help (below) |
| Clustered scores | 38.0% of Standard Class orders share a single calibrated score, so ranking inside Standard Class is almost arbitrary |
| No real-time features | Live courier tracking, weather, traffic data not incorporated |

Segment-only LightGBM models (trained on one shipping mode, same features):

| Segment | Final model ROC-AUC | Segment-only model ROC-AUC | Final PR-AUC | Segment-only PR-AUC |
|---|---|---|---|---|
| Standard Class | 0.5292 | 0.5278 | 0.3983 | 0.3993 |
| Second Class | 0.5666 | 0.5623 | 0.8044 | 0.7892 |

---

## 10. Ethical Notes

| Dimension | Detail |
|---|---|
| **PII** | No personally identifiable information used (names, emails, passwords, street addresses dropped in `01_prepare.py`) |
| **Customer ID** | `Customer_Id` is used only internally for history aggregation; never passed as a model feature |
| **Fairness** | No demographic or protected-class attributes are features; model is not evaluated for demographic parity |
| **Transparency** | SHAP explanations provided for every prediction; plain-English reason codes via `get_reason_codes()` |
| **Human oversight** | Model flags orders for human review; it does not autonomously cancel or reroute shipments |
| **Auditability** | All thresholds, calibration parameters and metrics are saved in `models/` and `outputs/final_metrics.json`; reproducible via `run_all.py` |

---

## 11. Files

| File | Description |
|---|---|
| `models/final_model.joblib` | Tuned LightGBM model |
| `models/preprocessor.joblib` | Fitted ColumnTransformer (OHE + StandardScaler) |
| `models/final_calibrator.joblib` | Fitted isotonic calibrator |
| `models/thresholds.json` | Thresholds, tier cutoffs, tier descriptions, selection checks |
| `outputs/final_metrics.json` | Every reported number (single source of truth) |
| `outputs/report_tables.md` | All tables in paste-ready form (generated) |
| `outputs/verification_findings.md` | Label-artifact and confound verification |

---

*Model Card follows the Google Model Cards framework (Mitchell et al., 2019).*
