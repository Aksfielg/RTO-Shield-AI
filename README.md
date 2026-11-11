# RTO Shield AI — Model 1

> **Early-warning late-delivery risk scoring at order-placement time.**

---

## Goal

RTO Shield AI - Model 1 is an academic prototype evaluating whether pre-dispatch order features correlate with **late-delivery risk** (`Late_delivery_risk = 1`) at order placement time, before shipment dispatch.

> **Important Distinction:** The target is `Late_delivery_risk` (delivery delay), which is an upstream operational proxy. It is **NOT** a direct measure of Return-To-Origin (RTO), doorstep refusal, or COD cash default. In actual e-commerce operations, late delivery can increase RTO risk, but this model predicts delivery delay rather than return events.

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
| Date range | Jan 2015 – Jan 2018 |
| Countries | 164 |
| Positive rate (Late_delivery_risk = 1) | ≈ 54.9 % |

---

## ⚠️ Critical Proxy-Target & Scope Caveats

1. **Synthetic / Rule-Based Labelling:** `Late_delivery_risk` is a pre-computed label in the DataCo dataset that heavily reflects synthetic dispatch rules rather than real-world courier variance. For instance, `order_hour` alone determines ~94 % of Same Day delivery outcomes (orders placed before noon → 0 % late; at/after noon → 94.4 % late).
2. **Standard Class Discrimination:** In the largest shipping tier (Standard Class, comprising ~60 % of orders), the model achieves an ROC-AUC of **0.5296** (near-random guess). The aggregate test ROC-AUC (0.7713) is largely driven by First Class (0.8732) and Same Day (0.9914) rule predictability.
3. **Not Validated on Indian D2C:** This dataset represents global supply chain orders (predominantly North America and LATAM). It does not capture Indian D2C last-mile dynamics (NDR verification, fake door attempts, PIN-code delivery tiers, or COD payment fraud). Do not deploy without retraining on actual courier and return telemetry.

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
**Key packages:** pandas, numpy, scikit-learn, xgboost, lightgbm, shap, joblib, matplotlib, scipy, pyarrow, pytest

---

## How to Run

### Full pipeline (recommended)

```bash
python run_all.py
```

Runs all scripts in order and logs output to `outputs/run_log.txt`.  
⚠️ Full pipeline takes **~30–60 minutes** (dominated by SHAP and multi-seed tuning).

### Individual scripts

```bash
venv\Scripts\python.exe src/01_prepare.py              # Wrangle & feature-engineer
venv\Scripts\python.exe src/02_train.py                # Train 5 models, compare metrics
venv\Scripts\python.exe src/03_calibrate_threshold.py  # Calibrate + threshold search
venv\Scripts\python.exe src/04_explain.py              # SHAP explanations
venv\Scripts\python.exe src/05_tune_multiseed.py       # Hyperparameter tuning + stability
venv\Scripts\python.exe src/06_robustness.py           # Ablations, backtest, PSI
venv\Scripts\python.exe src/07_predict_demo.py         # Predict demo (5 orders)
venv\Scripts\python.exe src/08_sanity_checks.py        # Sanity checks suite & provenance
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

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

### Model Comparison (threshold = 0.5, TEST set)

| Model | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Brier |
|---|---|---|---|---|---|
| Baseline LR (shipping+days) | 0.7276 | 0.7291 | 0.7503 | 0.658 | 0.1965 |
| Logistic Regression | 0.7412 | 0.7497 | 0.8200 | 0.674 | 0.1949 |
| Random Forest | 0.7626 | 0.7653 | 0.8312 | 0.705 | 0.1882 |
| XGBoost | 0.7626 | 0.7658 | 0.8351 | 0.708 | 0.1864 |
| **LightGBM** | **0.7638** | **0.7678** | **0.8354** | **0.696** | **0.1833** |

### Final Model: LightGBM (tuned, threshold = 0.3538)

| Metric | TEST | 95 % Bootstrap CI |
|---|---|---|
| ROC-AUC | 0.7713 | [0.7622, 0.7801] |
| PR-AUC | 0.8134 | [0.8047, 0.8215] |
| F1 | 0.7259 | — |
| Recall | 0.9549 | — |
| Brier | 0.1831 | — |
| ECE (after calibration) | 0.0067 | — |

**Multi-seed stability (10 seeds):** ROC-AUC 0.7706 ± 0.0011, PR-AUC 0.8134 ± 0.0034

**Δ ROC-AUC vs baseline (bootstrap):** [+0.0365, +0.0477] — significantly better.

---

## Folder Structure

```
rto_shield_model1/
├── data/
│   ├── raw/                      # DataCoSupplyChainDataset.csv (not committed)
│   └── processed/
│       └── orders.parquet        # 65,752 orders × 32 columns
├── src/
│   ├── config.py                 # Paths, feature lists, constants
│   ├── features.py               # Feature matrix builder
│   ├── reasons.py                # Plain-English SHAP reason codes
│   ├── predict.py                # score_order() / score_batch() + CLI
│   ├── 01_prepare.py             # Data wrangling & feature engineering
│   ├── 02_train.py               # Train 5 models
│   ├── 03_calibrate_threshold.py # Calibration + threshold selection
│   ├── 04_explain.py             # SHAP explanations
│   ├── 05_tune_multiseed.py      # Hyperparameter tuning + stability
│   ├── 06_robustness.py          # Ablations, backtest, drift check
│   └── 07_predict_demo.py        # Prediction demo
├── models/                       # Saved models + artefacts (.joblib, .json)
├── outputs/                      # Metrics CSVs, PNGs, tables
├── tests/
│   ├── test_pipeline.py          # 4 correctness tests
│   ├── test_no_leakage.py        # Leakage brute-force tests
│   └── test_imports.py           # Import smoke tests
├── run_all.py                    # Full pipeline runner
├── sample_order.json             # Example order for CLI demo
├── README.md
└── MODEL_CARD.md
```

---

## Licence

Code: MIT. Dataset: CC BY 4.0 (cite Constante et al. 2019).

