# RTO Shield AI - Model 1: Sanity Checks & Quality Audits

> Generated: 2026-11-12 22:14:41  
> Pipeline: Final Tuned LightGBM + Isotonic Calibration  
> Dataset: Constante et al. 2019 (Mendeley Data V5, DOI 10.17632/8gx2fvg2k6.5)  

---

## 1. Label-Shuffle Test (Signal Integrity Audit)

The target label `y_train` is randomly shuffled (seed=42) while retaining the exact feature matrix `X_train`. The model is retrained from scratch using the final tuned LightGBM hyperparameters and evaluated on the untouched validation and test sets.

| Metric | Expected | Observed | Status |
|---|---|---|---|
| Validation ROC-AUC | ~0.50 [0.47, 0.53] | **0.4950** | PASS |
| Test ROC-AUC | ~0.50 [0.47, 0.53] | **0.4959** | PASS |

**Overall Result:** `PASS`  
- **Interpretation:** When true relationships between features and outcome are destroyed by label permutation, the model achieves near-chance performance (validation 0.4950, test 0.4959). This proves that the 0.7716 test ROC-AUC of the genuine model stems from true signal, not preprocessor leakage or data memorisation.

---

## 2. Trivial Policies Comparison (Test Set, N = 9,863)

Evaluating whether the model offers actionable operational value over simple heuristic rules or uncalibrated default baselines:

| Policy | Precision | Recall | F1 Score | Orders Flagged | Flagged Share (%) |
|---|---|---|---|---|---|
| **Flag-all (all positive)** | 0.5505 | 1.0000 | 0.7101 | 9,863 | 100.00% |
| **Flag-none (all negative)** | 0.0000 | 0.0000 | 0.0000 | 0 | 0.00% |
| **Flag by High tier (score >= 0.7423)** | 0.8836 | 0.5215 | 0.6559 | 3,205 | 32.50% |
| **Flag by Precision >= 80% (score >= 0.4657)** | 0.8516 | 0.5895 | 0.6967 | 3,759 | 38.11% |
| **Flag by final threshold (score >= 0.3538)** | 0.5855 | 0.9549 | 0.7259 | 8,855 | 89.78% |

**Policy Takeaways:**
- **Flag-all**: Yields an F1 of 0.7101 purely because the baseline test positive rate is 55.05%. Flagging 100% of orders is not actionable.
- **Flag by High tier (score >= 0.7423)**: precision **88.36%** while flagging **32.50%** of orders, capturing 52.15% of all late shipments.
- **Flag by final threshold (score >= 0.3538)**: F1 0.7259 and recall **95.49%**, but it flags 89.78% of orders, so it adds little over flag-all.

---

## 3. Top-Decile Composition (Top 10% Scored Test Orders, N = 986)

Analysing the shipping-mode distribution within the highest 10% risk scores compared against the overall test distribution:

| Shipping Mode | Top 10% Count | Top 10% Share (%) | Full Test Count | Overall Share (%) | Empirical Late Rate |
|---|---|---|---|---|---|
| **First Class** | 986 | 100.00% | 1513 | 15.34% | 0.9504 |
| **Same Day** | 0 | 0.00% | 539 | 5.46% | 0.5028 |
| **Second Class** | 0 | 0.00% | 1923 | 19.50% | 0.7691 |
| **Standard Class** | 0 | 0.00% | 5888 | 59.70% | 0.3808 |

**Analytical Insight:**
- **100.00% of the top decile** consists of **First Class** orders (986 of 986).
- First Class has an empirical late rate of 95.0% in the test set.
- The highest scores concentrate on the segment where the label is nearly deterministic (see MODEL_CARD limitations).

---

## 4. Calibration by Shipping Mode (Test Set)

Assessing whether calibrated probabilities align with actual empirical late rates within individual fulfillment segments:

| Shipping Mode | Test Orders | Test Share (%) | Mean Predicted Score | Actual Late Rate | Calibration Error |
|---|---|---|---|---|---|
| **First Class** | 1,513 | 15.34% | 0.9551 | 0.9504 | +0.0047 |
| **Same Day** | 539 | 5.46% | 0.4504 | 0.5028 | -0.0524 |
| **Second Class** | 1,923 | 19.50% | 0.7590 | 0.7691 | -0.0101 |
| **Standard Class** | 5,888 | 59.70% | 0.3792 | 0.3808 | -0.0015 |

**Segment Calibration Insights:**
- **First Class (15.3% of volume)**: mean predicted 95.51% vs 95.04% actual (+0.47 pp).
- **Same Day (5.5% of volume)**: mean predicted 45.04% vs 50.28% actual (-5.24 pp).
- **Second Class (19.5% of volume)**: mean predicted 75.90% vs 76.91% actual (-1.01 pp).
- **Standard Class (59.7% of volume)**: mean predicted 37.92% vs 38.08% actual (-0.15 pp).

---

## 5. Raw Dataset Provenance & Integrity

| Property | Value |
|---|---|
| **File Name** | `DataCoSupplyChainDataset.csv` |
| **File Path** | `C:\Users\risha\Downloads\rto_shield_model1\data\raw\DataCoSupplyChainDataset.csv` |
| **File Size** | 95,910,149 bytes (91.47 MB) |
| **Row Count (excl. header)** | **180,519** |
| **SHA-256 Checksum** | `fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6` |
| **Deduplicated Orders** | 65,752 order-level records |

This SHA-256 checksum uniquely identifies the exact version 5 DataCo benchmark dataset used across all experimental milestones.