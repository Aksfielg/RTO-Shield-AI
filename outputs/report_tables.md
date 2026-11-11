# RTO Shield AI — Report Tables

All tables are paste-ready. Data sourced from outputs CSVs and console logs.  
See the **Figure File Index** at the bottom for which PNGs accompany each section.

---

## Table 1 — Model Comparison (Validation Set, threshold = 0.5)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Prec@Top10% | ECE |
|---|---|---|---|---|---|---|---|---|
| Baseline LR (shipping+days) | 0.8405 | 0.5384 | 0.6564 | 0.7276 | 0.7445 | 0.1976 | 0.9514 | 0.0107 |
| Logistic Regression | 0.8322 | 0.5647 | 0.6728 | 0.7412 | 0.8120 | 0.1970 | 0.9970 | 0.0283 |
| Random Forest | 0.8269 | 0.6045 | 0.6984 | 0.7626 | 0.8251 | 0.1877 | 1.0000 | 0.0321 |
| XGBoost | 0.8240 | 0.6047 | 0.6975 | 0.7626 | 0.8295 | 0.1862 | 1.0000 | 0.0357 |
| **LightGBM** | **0.8480** | **0.5905** | **0.6962** | **0.7638** | **0.8290** | **0.1839** | **1.0000** | **0.0258** |

## Table 2 — Model Comparison (Test Set, threshold = 0.5)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Prec@Top10% | ECE |
|---|---|---|---|---|---|---|---|---|
| Baseline LR (shipping+days) | 0.8490 | 0.5372 | 0.6580 | 0.7291 | 0.7503 | 0.1965 | 0.9504 | 0.0051 |
| Logistic Regression | 0.8440 | 0.5611 | 0.6741 | 0.7497 | 0.8200 | 0.1949 | 0.9970 | 0.0190 |
| Random Forest | 0.7939 | 0.6333 | 0.7046 | 0.7653 | 0.8312 | 0.1882 | 1.0000 | 0.0545 |
| XGBoost | 0.7754 | 0.6510 | 0.7078 | 0.7658 | 0.8351 | 0.1864 | 1.0000 | 0.0401 |
| **LightGBM** | **0.8531** | **0.5882** | **0.6963** | **0.7678** | **0.8354** | **0.1833** | **1.0000** | **0.0182** |

---

## Table 3 — Calibration: Before vs After (Test Set)

| Metric | Before (raw LightGBM) | After (Isotonic Regression) |
|---|---|---|
| Brier score | 0.1833 | 0.1831 |
| ECE | 0.0182 | 0.0042 |

---

## Table 4 — Threshold Analysis (chosen on Validation, applied to Test)

| Threshold strategy | Threshold | Test Precision | Test Recall | Test F1 | Test Flagged Share |
|---|---|---|---|---|---|
| **Flag by High tier (Operational Flag)** | **0.6988** | **0.8836** | **0.5215** | **0.6559** | **32.50% (3,205 / 9,863)** |
| **Precision ≥ 0.80 strategy** | **0.3994** | **0.8506** | **0.5902** | **0.6969** | **38.20% (3,768 / 9,863)** |
| Maximize F1 (validation) | 0.3538 | 0.5855 | 0.9549 | 0.7259 | 89.78% (8,855 / 9,863) |
| Cost 2:1 (FN=2×FP) | 0.3320 | 0.5828 | 0.9584 | 0.7248 | 90.57% (8,933 / 9,863) |
| Flag-all (trivial baseline) | 0.0000 | 0.5505 | 1.0000 | 0.7101 | 100.00% (9,863 / 9,863) |
| Flag-none (trivial baseline) | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% (0 / 9,863) |

---

## Table 5 — Risk Tiers (Test Set, cutoffs from Validation p33/p67 of Final Tuned Model)

| Tier | Score range | Test Orders | Share % | Actual Late Rate | Late Captured % | Cutoffs Source |
|---|---|---|---|---|---|---|
| Low | score < 0.3749 | 1,740 | 17.6% | 0.2891 | 9.3% | p33 of final tuned calibrated val scores |
| Medium | 0.3749 ≤ score < 0.6988 | 4,850 | 49.2% | 0.4223 | 37.7% | p33 to p67 of final tuned calibrated val scores |
| High | score ≥ 0.6988 | 3,273 | 33.2% | 0.8796 | 53.0% | p67 of final tuned calibrated val scores (Operational Flag) |

---

## Table 6 — Hyperparameter Tuning (Tuned vs Default, Validation)

| Configuration | Val ROC-AUC | Winner |
|---|---|---|
| Default LightGBM | 0.7638 | — |
| Tuned (RandomizedSearchCV, n_iter=30, TSS CV=4) | 0.7747 | ✅ Tuned |

**Best hyperparameters found:**

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 4 |
| learning_rate | 0.0093 |
| subsample | 0.6040 |
| colsample_bytree | 0.5867 |
| min_child_weight | 10 |
| reg_lambda | 0.5019 |

---

## Table 7 — Multi-Seed Stability (10 seeds, Test Set)

| Seed | ROC-AUC | PR-AUC | F1 | Recall | Brier |
|---|---|---|---|---|---|
| 0 | 0.7715 | 0.8133 | 0.7241 | 0.9676 | 0.1826 |
| 1 | 0.7699 | 0.8088 | 0.7260 | 0.9534 | 0.1826 |
| 2 | 0.7714 | 0.8196 | 0.7239 | 0.9654 | 0.1827 |
| 3 | 0.7685 | 0.8107 | 0.7241 | 0.9639 | 0.1830 |
| 4 | 0.7706 | 0.8131 | 0.7239 | 0.9665 | 0.1829 |
| 5 | 0.7718 | 0.8170 | 0.7251 | 0.9523 | 0.1827 |
| 6 | 0.7704 | 0.8115 | 0.7248 | 0.9635 | 0.1828 |
| 7 | 0.7720 | 0.8138 | 0.7244 | 0.9628 | 0.1830 |
| 8 | 0.7703 | 0.8158 | 0.7249 | 0.9626 | 0.1830 |
| 9 | 0.7700 | 0.8099 | 0.7250 | 0.9619 | 0.1829 |
| **Mean** | **0.7706** | **0.8134** | **0.7246** | **0.9620** | **0.1828** |
| **± Std** | **±0.0011** | **±0.0034** | **±0.0007** | **±0.0051** | **±0.0002** |

---

## Table 8 — Bootstrap 95% Confidence Intervals (Test Set, 1000 resamples)

| Metric | Point estimate | 95 % CI lower | 95 % CI upper |
|---|---|---|---|
| Final model ROC-AUC | 0.7713 | 0.7622 | 0.7801 |
| Final model PR-AUC | 0.8134 | 0.8047 | 0.8215 |
| Δ ROC-AUC (final − baseline) | 0.0422 | 0.0365 | 0.0477 |
| Δ PR-AUC (final − baseline) | 0.0631 | 0.0594 | 0.0667 |

---

## Table 9 — Ablation Study (Test Set, threshold = 0.3538)

| Feature set | N features | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier |
|---|---|---|---|---|---|---|---|
| **All features** | **27** | **0.7713** | **0.8134** | **0.7259** | **0.9549** | **0.5855** | **0.1831** |
| No shipping mode/days | 25 | 0.5594 | 0.5918 | 0.7099 | 0.9991 | 0.5505 | 0.2452 |
| No history features | 21 | 0.7710 | 0.8234 | 0.7248 | 0.9584 | 0.5828 | 0.1847 |
| No time features | 22 | 0.7409 | 0.7990 | 0.7123 | 0.9578 | 0.5670 | 0.1951 |
| No payment / is_cash | 25 | 0.7555 | 0.7966 | 0.7221 | 1.0000 | 0.5650 | 0.1847 |
| Shipping only (baseline) | 2 | 0.7291 | 0.7503 | 0.7101 | 1.0000 | 0.5505 | 0.1963 |

> **Key finding:** Removing shipping_mode + scheduled_days collapses ROC-AUC from 0.7713 → 0.5594 — these are the dominant features.

---

## Table 10 — Per-Shipping-Mode Evaluation (Test Set, full model vs baseline)

| Shipping Mode | Test Orders | Full model ROC-AUC | Full model PR-AUC | Baseline ROC-AUC | Baseline PR-AUC | Δ ROC-AUC |
|---|---|---|---|---|---|---|
| First Class | 1,513 | 0.8732 | 0.9915 | 0.5000 | 0.9504 | +0.3732 |
| Same Day | 539 | 0.9914 | 0.9844 | 0.5000 | 0.5028 | +0.4914 |
| Second Class | 1,923 | 0.5600 | 0.7936 | 0.5000 | 0.7691 | +0.0600 |
| Standard Class | 5,888 | 0.5296 | 0.3981 | 0.5000 | 0.3808 | +0.0296 |

> Note: Inside each shipping mode, the baseline LR has ROC-AUC = 0.5000 because its only other feature (`Days_for_shipment_scheduled`) is constant within each mode. The full model adds massive value for First Class (+0.37) and Same Day (+0.49), but exhibits weak discrimination for Standard Class (0.5296), which constitutes 59.7% of all test orders.

---

## Table 11 — Rolling-Origin Backtest (5 Expanding-Window Folds)

| Fold | Train N | Test N | Test start | Test end | ROC-AUC | PR-AUC | F1 | Recall | Precision | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 26,300 | 6,575 | 2016-06-01 | 2016-09-25 | 0.7734 | 0.8191 | 0.7298 | 0.9565 | 0.5900 | 0.1826 |
| 2 | 32,876 | 6,575 | 2016-09-25 | 2017-01-18 | 0.7762 | 0.8143 | 0.7293 | 0.8881 | 0.6187 | 0.1812 |
| 3 | 39,451 | 6,576 | 2017-01-18 | 2017-05-12 | 0.7596 | 0.8087 | 0.7151 | 0.9184 | 0.5855 | 0.1861 |
| 4 | 46,026 | 6,575 | 2017-05-12 | 2017-09-05 | 0.7713 | 0.8096 | 0.7239 | 0.9381 | 0.5894 | 0.1831 |
| 5 | 52,601 | 6,575 | 2017-09-05 | 2017-12-14 | 0.7670 | 0.8157 | 0.7233 | 0.8804 | 0.6138 | 0.1860 |
| **Mean** | — | — | — | — | **0.7695** | **0.8135** | **0.7243** | **0.9163** | **0.5995** | **0.1838** |
| **Std** | — | — | — | — | **±0.0057** | **±0.0040** | **±0.0053** | **±0.0280** | **±0.0132** | **±0.0019** |

---

## Table 12 — Decile Lift Table (Test Set)

| Decile | N | Late in decile | Late rate | Lift | Cum late | Cum lift | Cum gain % |
|---|---|---|---|---|---|---|---|
| 1 (highest risk) | 986 | 986 | 1.0000 | 1.8164 | 986 | 1.8164 | 18.16 |
| 2 | 986 | 866 | 0.8783 | 1.5953 | 1,852 | 1.7059 | 34.11 |
| 3 | 986 | 773 | 0.7840 | 1.4240 | 2,625 | 1.6119 | 48.34 |
| 4 | 986 | 651 | 0.6602 | 1.1993 | 3,276 | 1.5087 | 60.33 |
| 5 | 986 | 394 | 0.3996 | 0.7258 | 3,670 | 1.3522 | 67.59 |
| 6 | 986 | 397 | 0.4026 | 0.7313 | 4,067 | 1.2487 | 74.90 |
| 7 | 986 | 369 | 0.3742 | 0.6798 | 4,436 | 1.1674 | 81.69 |
| 8 | 986 | 378 | 0.3834 | 0.6963 | 4,814 | 1.1085 | 88.66 |
| 9 | 986 | 351 | 0.3560 | 0.6466 | 5,165 | 1.0572 | 95.12 |
| 10 (lowest risk) | 989 | 265 | 0.2679 | 0.4867 | 5,430 | 1.0000 | 100.00 |

> **Top 3 deciles capture 48.3 % of all late orders** while covering only 30 % of test orders.  
> **Decile 1 lift = 1.82×** (every late order in top decile is correctly identified — late rate = 100 %).

---

## Table 13 — Sanity Check: Same Day Late Rate by Order Hour

| Order time | Group | Late rate |
|---|---|---|
| Before 12:00 noon | Same Day, early | ≈ 0.0 % |
| At or after 12:00 noon | Same Day, late | ≈ 94.4 % |

> This near-rule confirms `order_hour` is near-deterministic for Same Day shipping, consistent with the proxy-target caveat.

---

## Table 14 — Secondary Target Robustness (aux_target_cancel_fraud)

| Metric | Value |
|---|---|
| Positive rate (cancel/fraud proxy) | 4.3 % (train), 4.3 % (test) |
| ROC-AUC | 0.8794 |
| PR-AUC | 0.1576 |
| F1 | 0.2722 |
| Recall | 0.9906 |
| Precision | 0.1578 |
| Brier score | 0.0364 |

> This is a **robustness check only** — the model is not intended or validated for fraud detection.

> This is a **robustness check only** — the model is not intended or validated for fraud detection.

---

## Table 15 — Population Stability Index (Train → Test, top 8 SHAP features)

| Feature | PSI | Stability |
|---|---|---|
| Shipping_Mode_Standard_Class | ≈ 0.000 | 🟢 Stable |
| Shipping_Mode_First_Class | ≈ 0.000 | 🟢 Stable |
| Type_TRANSFER | ≈ 0.000 | 🟢 Stable |
| order_hour | ≈ 0.000 | 🟢 Stable |
| Days_for_shipment_scheduled | ≈ 0.000 | 🟢 Stable |
| order_month | ≈ 0.000 | 🟢 Stable |
| customer_prev_late_rate | ≈ 0.000 | 🟢 Stable |
| order_dow | ≈ 0.000 | 🟢 Stable |

> PSI < 0.1 = stable; 0.1–0.2 = slight shift; > 0.2 = significant drift. All features are 🟢.

---

## Figure File Index

| Figure file | Report section | Description |
|---|---|---|
| `outputs/roc_curves.png` | §Model Comparison | ROC curves for all 5 models on test set |
| `outputs/pr_curves.png` | §Model Comparison | PR curves for all 5 models on test set |
| `outputs/calibration.png` | §Calibration | Reliability diagram before vs after isotonic calibration |
| `outputs/lift_gain.png` | §Threshold & Tiers | Decile lift bar chart + cumulative gain curve |
| `outputs/shap_importance.png` | §Explainability | Top-15 mean \|SHAP\| bar chart |
| `outputs/shap_summary.png` | §Explainability | SHAP beeswarm summary plot |
| `outputs/shap_dep_order_hour.png` | §Explainability | SHAP dependence plot: Order Hour |
| `outputs/shap_dep_scheduled_days.png` | §Explainability | SHAP dependence plot: Scheduled Shipping Days |
| `outputs/shap_dep_order_value.png` | §Explainability | SHAP dependence plot: Order Value |
| `outputs/shap_dep_value_per_item.png` | §Explainability | SHAP dependence plot: Value per Item |
| `outputs/shap_dep_customer_prev_late_rate.png` | §Explainability | SHAP dependence plot: Customer Historical Late Rate |
| `outputs/shap_waterfall_confident_late.png` | §Explainability | Waterfall: confident late prediction |
| `outputs/shap_waterfall_confident_ontime.png` | §Explainability | Waterfall: confident on-time prediction |
| `outputs/shap_waterfall_false_negative.png` | §Explainability | Waterfall: false negative case study |
| `outputs/ablation_chart.png` | §Ablation | Bar chart of ROC-AUC per feature subset |
| `outputs/backtest_plot.png` | §Robustness | Rolling-origin ROC-AUC and PR-AUC per fold |
| `outputs/demo_scores.csv` | §Demo | 5 test orders with scores, tiers, reasons |

---

*Generated by RTO Shield AI pipeline. Data: Constante et al. 2019, Mendeley Data V5, DOI 10.17632/8gx2fvg2k6.5, CC BY 4.0.*

