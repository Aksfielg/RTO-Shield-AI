# RTO Shield AI — Model 1

> **Early-warning late-delivery risk scoring at order-placement time.**

<!-- GENERATED FILE: edit templates/README.md.tmpl, then run `python src/11_render_docs.py`. -->

---

## Goal

RTO Shield AI - Model 1 is an academic prototype evaluating whether pre-dispatch order features correlate with **late-delivery risk** (`Late_delivery_risk = 1`) at order placement time, before shipment dispatch.

> **Important Distinction:** The target is `Late_delivery_risk` (delivery delay), which is an upstream operational proxy. It is **NOT** a direct measure of Return-To-Origin (RTO), doorstep refusal, or COD cash default. In actual e-commerce operations, late delivery can increase RTO risk, but this model predicts delivery delay rather than return events. **No result in this repository measures RTO, doorstep-refusal or Indian COD performance.**

---

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis**

> Constante, F., Silva, F., & Pereira, A. (2019). *DataCo Smart Supply Chain for Big Data Analysis* [Data set]. Mendeley Data, V5.
> DOI: [10.17632/8gx2fvg2k6.5](https://doi.org/10.17632/8gx2fvg2k6.5)
> Licence: **CC BY 4.0**

| Property | Value |
|---|---|
| Raw rows | 180,519 |
| Processed orders | 65,752 |
| Date range | 2015-01-01 to 2018-01-31 |
| Countries | 164 |
| Positive rate (Late_delivery_risk = 1) | 54.8% |
| Raw file SHA-256 | `fa6d022ed437155e1a2f0378710602848703c8a7f203f7ff5d77805bf8480aa6` |

| Split | Orders | Date range | Positive rate |
|---|---|---|---|
| Train | 46,026 | 2015-01-01 -> 2017-03-16 | 54.85% |
| Validation | 9,863 | 2017-03-16 -> 2017-09-05 | 54.50% |
| Test | 9,863 | 2017-09-05 -> 2018-01-31 | 55.05% |

---

## ⚠️ Critical Proxy-Target & Scope Caveats

1. **Near rule-based labelling.** `Late_delivery_risk` is a pre-computed label in the DataCo dataset that heavily reflects near rule-based dispatch logic rather than real courier variance. For Same Day orders, the order hour alone separates outcomes: orders placed before 12:00 are late 0.0% of the time, orders from 12:00 onward 95.4% (3,571 Same Day orders).
2. **Standard Class has almost no signal.** Standard Class is 59.7% of test orders, and the model's ROC-AUC there is **0.5292** (95% CI 0.5138 to 0.5440; signal: **limited**). The aggregate test ROC-AUC (0.7716) is driven by First Class (0.8709) and Same Day (0.9901). A model trained on Standard Class alone reaches only 0.5278.
3. **The payment-type (TRANSFER) effect is a labelling artifact.** All 2,855 CANCELED / SUSPECTED_FRAUD orders are TRANSFER payments and every one of them carries `Late_delivery_risk = 0`. Within each shipping mode the TRANSFER late rate differs from other payment types by -9.1 pp; with those orders removed the difference is -0.0 pp. Do not read TRANSFER as lower payment risk.
4. **Not validated on Indian D2C.** This dataset represents global supply chain orders (predominantly North America and LATAM). It does not capture Indian D2C last-mile dynamics (NDR verification, fake door attempts, PIN-code delivery tiers, or COD payment fraud). Do not deploy without retraining on actual courier and return telemetry.
5. **Scores are clustered.** The calibrated score takes only 41 distinct values on the test set; 38.0% of Standard Class orders share one score. Cutoffs are therefore placed in low-density gaps between clusters (see Risk tiers).

---

## Setup

```bash
# 1. Clone / download the repo
# 2. Place DataCoSupplyChainDataset.csv in data/raw/

# 3. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows PowerShell
# source venv/bin/activate     # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt
```

**Python:** 3.11+ recommended (tested on 3.13)
**Key packages:** pandas, numpy, scikit-learn, xgboost, lightgbm, shap, joblib, matplotlib, scipy, pyarrow, pytest, streamlit, plotly

---

## How to Run

### Full pipeline (recommended)

```bash
python run_all.py
```

Runs all scripts in dependency order and logs to `outputs/run_log.txt`. Order: `01 → 02 → 05 → 03 → 04 → 06 → 07 → 08 → 09 → 10 → 11` (the tuned model from step 05 must exist before tiers and thresholds are set in step 03).
Full pipeline takes roughly 30–60 minutes (dominated by SHAP and multi-seed tuning).

### Individual scripts

```bash
venv\Scripts\python.exe src/01_prepare.py              # Wrangle & feature-engineer
venv\Scripts\python.exe src/02_train.py                # Train 5 models, compare metrics
venv\Scripts\python.exe src/05_tune_multiseed.py       # Tune final LightGBM, calibrate, stability
venv\Scripts\python.exe src/03_calibrate_threshold.py  # Cluster-aware tiers + thresholds (final model)
venv\Scripts\python.exe src/04_explain.py              # SHAP explanations
venv\Scripts\python.exe src/06_robustness.py           # Ablations, backtest, PSI
venv\Scripts\python.exe src/07_predict_demo.py         # Predict demo (5 orders)
venv\Scripts\python.exe src/08_sanity_checks.py        # Sanity checks suite & provenance
venv\Scripts\python.exe src/09_verification.py         # Label-artifact / confound verification
venv\Scripts\python.exe src/10_finalize.py             # Compute every reported number -> outputs/final_metrics.json
venv\Scripts\python.exe src/11_render_docs.py          # Render README, MODEL_CARD, report_tables from JSON
```

### Single source of truth

`outputs/final_metrics.json` (all metrics) and `models/thresholds.json` (cutoffs and tier text) feed the Streamlit app, `README.md`, `MODEL_CARD.md` and `outputs/report_tables.md`. Those documents are rendered from `templates/` and contain no hand-typed metrics.

### Streamlit Dashboard

```bash
streamlit run app.py
```

Nine pages in three groups (Overview, Decisioning & Operations, Model Evaluation & Governance). Every number comes from
`outputs/final_metrics.json`, `models/thresholds.json` or the real holdout orders; all charts are Plotly; there is no
invented data (the "Score an Order" presets are real holdout orders, one per risk tier).

### CLI prediction

```bash
venv\Scripts\python.exe src/predict.py --json sample_order.json
```

### Tests

```bash
venv\Scripts\pytest.exe tests/ -v -s
```

---

## Results

### Model Comparison (threshold = 0.5, validation and test)

| Model | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Brier |
|---|---|---|---|---|---|
| Baseline LR (shipping+days) | 0.7276 | 0.7291 | 0.7503 | 0.658 | 0.1965 |
| Logistic Regression | 0.7412 | 0.7497 | 0.8200 | 0.674 | 0.1949 |
| Random Forest | 0.7626 | 0.7653 | 0.8312 | 0.705 | 0.1882 |
| XGBoost | 0.7626 | 0.7658 | 0.8351 | 0.708 | 0.1864 |
| LightGBM (default) | 0.7638 | 0.7678 | 0.8354 | 0.696 | 0.1833 |

### Final Model: tuned LightGBM + isotonic calibration

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

**Multi-seed stability (10 seeds):** ROC-AUC 0.7712 ± 0.0007, PR-AUC 0.8363 ± 0.0006

**Δ ROC-AUC vs baseline (bootstrap 95% CI):** [0.0364, 0.0483]

**Calibration (final model, test):**

| Metric (TEST, final tuned model) | Before calibration (raw) | After isotonic calibration |
|---|---|---|
| Brier score | 0.1860 | 0.1831 |
| ECE | 0.0495 | 0.0067 |

### Per-shipping-mode performance (test)

| Shipping mode | n (test) | Share | Prevalence | ROC-AUC | ROC-AUC 95% CI | PR-AUC | PR-AUC lift | Signal |
|---|---|---|---|---|---|---|---|---|
| Same Day | 539 | 5.5% | 50.3% | 0.9901 | [0.9836, 0.9950] | 0.9896 | 1.97x | informative |
| First Class | 1,513 | 15.3% | 95.0% | 0.8709 | [0.8503, 0.8904] | 0.9929 | 1.04x | informative |
| Second Class | 1,923 | 19.5% | 76.9% | 0.5666 | [0.5347, 0.5968] | 0.8044 | 1.05x | limited |
| Standard Class | 5,888 | 59.7% | 38.1% | 0.5292 | [0.5138, 0.5440] | 0.3983 | 1.05x | limited |

Signal = **limited** when the ROC-AUC 95% bootstrap CI lower bound is below 0.55 (which also covers any CI that includes 0.5); otherwise informative. Limited modes: Second Class, Standard Class. PR-AUC lift = PR-AUC divided by the mode's prevalence (1.0x = no better than the base rate; a mode with very high prevalence cannot show much lift).

### Feature ablation (test)

| Feature subset | # feat. | ROC-AUC | ROC-AUC 95% CI | PR-AUC | Brier | Own thr. | F1 | Precision | Recall | Accuracy | Bal. acc. | Spec. | MCC | Delta ROC-AUC vs all [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All features | 27 | 0.7716 | [0.7625, 0.7804] | 0.8362 | 0.1831 | 0.354 | 0.726 | 0.586 | 0.955 | 0.603 | 0.564 | 0.172 | 0.209 | - |
| No shipping mode/days | 25 | 0.5593 | [0.5473, 0.5695] | 0.5949 | 0.2452 | 0.251 | 0.710 | 0.550 | 0.999 | 0.550 | 0.500 | 0.001 | -0.000 | +0.2123 [+0.2012, +0.2236] * |
| No history features | 21 | 0.7708 | [0.7614, 0.7800] | 0.8362 | 0.1847 | 0.337 | 0.725 | 0.583 | 0.958 | 0.599 | 0.559 | 0.160 | 0.200 | +0.0008 [-0.0017, +0.0033] |
| No time features | 22 | 0.7408 | [0.7310, 0.7501] | 0.8153 | 0.1951 | 0.348 | 0.712 | 0.567 | 0.958 | 0.574 | 0.531 | 0.104 | 0.121 | +0.0308 [+0.0271, +0.0347] * |
| No payment/is_cash | 25 | 0.7551 | [0.7453, 0.7644] | 0.8162 | 0.1847 | 0.356 | 0.722 | 0.565 | 1.000 | 0.576 | 0.528 | 0.057 | 0.180 | +0.0165 [+0.0108, +0.0221] * |
| Shipping only (baseline) | 2 | 0.7291 | [0.7206, 0.7374] | 0.7503 | 0.1963 | 0.010 | 0.710 | 0.550 | 1.000 | 0.550 | 0.500 | 0.000 | 0.000 | +0.0425 [+0.0364, +0.0483] * |

Threshold metrics use each ablated model's OWN max-F1 threshold chosen on validation. Delta = ROC-AUC(all features) - ROC-AUC(ablated), paired bootstrap, * = CI excludes 0. History features add NO significant gain: ROC-AUC(all) - ROC-AUC(no history) = +0.0008 (95% paired-bootstrap CI [-0.0017, +0.0033] includes 0 or is negative); PR-AUC is 0.8362 with history vs 0.8362 without.

### Risk tiers

Tier cutoffs (0.4657 and 0.7423) are midpoints of low-density gaps between calibrated-score clusters, selected on validation scores only and verified on test. **Flagged = High tier.**

| Tier | Score range | Test orders | Share | Late rate | Late captured | Suggested handling |
|---|---|---|---|---|---|---|
| Low | score < 0.4657 | 6,104 | 61.9% | 36.5% | 41.0% | Standard dispatch with routine monitoring. |
| Medium | 0.4657 <= score < 0.7423 | 554 | 5.6% | 66.6% | 6.8% | Monitor the carrier pickup SLA. |
| High | score >= 0.7423 | 3,205 | 32.5% | 88.4% | 52.2% | Operational flag: review routing and confirm proactively. |

- **Low** - Lower risk: 37% of these orders were late in held-out test data (overall: 55%).
- **Medium** - Elevated risk: 67% of these orders were late in held-out test data (overall: 55%).
- **High** - Highest risk: 88% of these orders were late in held-out test data (overall: 55%).

### Policies on the test set

| Policy | Threshold | Flagged | Precision | Recall | F1 | Accuracy | Balanced acc. | Specificity | MCC |
|---|---|---|---|---|---|---|---|---|---|
| Flag all | - | 100.00% (9,863) | 0.5505 | 1.0000 | 0.7101 | 0.5505 | 0.5000 | 0.0000 | 0.0000 |
| High tier (Operational Flag) | 0.7423 | 32.50% (3,205) | 0.8836 | 0.5215 | 0.6559 | 0.6988 | 0.7187 | 0.9159 | 0.4646 |
| Precision >= 0.80 | 0.4657 | 38.11% (3,759) | 0.8516 | 0.5895 | 0.6967 | 0.7174 | 0.7318 | 0.8741 | 0.4749 |
| Max-F1 | 0.3538 | 89.78% (8,855) | 0.5855 | 0.9549 | 0.7259 | 0.6031 | 0.5635 | 0.1721 | 0.2086 |
| Cost-minimising (FN=2xFP) | 0.3320 | 92.21% (9,095) | 0.5777 | 0.9676 | 0.7234 | 0.5927 | 0.5506 | 0.1335 | 0.1877 |

### Threshold stability (test)

Shifting each threshold by ±0.01 and ±0.02 moves the flagged share by at most: High tier: 0.00 pp; Precision >= 0.80: 0.00 pp; Max-F1: 2.46 pp (limit 5 pp).

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

## Folder Structure

```
rto_shield_model1/
├── app.py                        # Streamlit entry point: page config, grouped navigation, sidebar chrome
├── ui/                           # dashboard package (no numbers typed in UI code)
│   ├── theme.py                  # palette, CSS, shared Plotly template
│   ├── components.py             # kpi_card, risk_badge, callout, styled_table, fmt_pct, fmt_money, plain_sentence
│   ├── charts.py                 # every chart (Plotly)
│   ├── data.py                   # cached loaders (st.cache_resource / st.cache_data), background model warm-up
│   └── pages/                    # home, score_order, real_orders, impact, batch,
│                                 # performance, explainability, robustness, limitations
├── .streamlit/config.toml        # light theme, indigo primary #4F46E5, sans-serif
├── data/
│   ├── raw/                      # DataCoSupplyChainDataset.csv (not committed)
│   └── processed/orders.parquet  # order-level feature table
├── src/
│   ├── config.py                 # Paths, feature lists, constants
│   ├── features.py               # Feature matrix builder
│   ├── tiering.py                # Cluster-aware cutoff selection + stability checks
│   ├── reasons.py                # Plain-English SHAP reason codes
│   ├── predict.py                # score_order() / score_batch() + CLI
│   └── 01_prepare.py … 11_render_docs.py
├── templates/                    # README / MODEL_CARD templates (no hand-typed numbers)
├── models/                       # Saved models + thresholds.json
├── outputs/                      # final_metrics.json, CSVs, PNGs, report_tables.md
├── tests/
├── run_all.py
├── sample_order.json
├── README.md                     # generated
└── MODEL_CARD.md                 # generated
```

---

## Licence

Code: MIT. Dataset: CC BY 4.0 (cite Constante et al. 2019).
