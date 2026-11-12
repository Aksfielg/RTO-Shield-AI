# Viva cheat-sheet — RTO Shield AI, Model 1

<!-- GENERATED FILE: edit templates/VIVA_CHEATSHEET.md.tmpl, then run `python src/11_render_docs.py`. Every number comes from outputs/final_metrics.json / models/thresholds.json. -->

**One sentence to open with:** *"Model 1 scores the risk that an order will be delivered late, using only information known when the order is placed, on public DataCo data. It is a methodology prototype for delivery-failure risk, not an RTO or Indian-COD model."*

---

## 1. The 10 numbers to memorize

| # | Number | Value | One-line meaning |
|---|---|---|---|
| 1 | Test ROC-AUC (95% CI) | **0.772** [0.7625, 0.7804] | Ranking quality on 9,863 unseen, later orders |
| 2 | PR-AUC vs prevalence | **0.836** vs 55.1% | 1.52x lift over just guessing "late" |
| 3 | Standard Class ROC-AUC | **0.529** (59.7% of volume) | No usable ranking inside the biggest mode |
| 4 | Second Class ROC-AUC | **0.567** (19.5% of volume) | Same story: both modes are near chance |
| 5 | Same Day / First Class ROC-AUC | **0.99** / **0.87** | Where the headline AUC comes from (near rule-based labels) |
| 6 | High-tier precision / coverage / recall | **88.4%** / 32.5% / 52.2% | The flagged tier (only High is flagged) |
| 7 | Tier cutoffs | **0.4657** and **0.7423** | Midpoints of low-density gaps in the score distribution |
| 8 | ECE raw → calibrated | **0.050 → 0.007** | Isotonic calibration on validation |
| 9 | Label-shuffle AUC | **0.496** (val 0.495) | Chance-level when labels are shuffled: no leakage artefact |
| 10 | Cancel / fraud orders | **2,855**, all labelled on-time | Why TRANSFER looks protective: gap -9.1 pp → -0.0 pp once removed |

**Also know:** split sizes 46,026 / 9,863 / 9,863 (train ends 2017-03-16, validation ends 2017-09-05, test to 2018-01-31); backtest AUC 0.770 ± 0.007; history features change AUC by 0.0008 (CI -0.0017 to 0.0033); dropping payment type costs 0.016 AUC.

---

## 2. Five-minute demo script (by page)

| Time | Page | Say / do |
|---|---|---|
| 0:00 | **Home** | Read the subtitle: *public-data proxy*. Point at the three KPIs and the "What works / What does not / Next" cards. Show the timeline: train earlier, test later, so the test set is genuinely the future. |
| 0:40 | **Score an Order** | Pick the real high-risk preset: badge, gauge and sentence all say the same number. Point at the **Information level** box: for Standard / Second Class it says *Limited*. Switch to Custom input and read the note that history features are imputed. |
| 1:50 | **Real Test Orders** | Change the shipping mode filter and show the selection resetting. Show the confusion summary and the Correct / Missed / False alarm chip. Mention the CSV download. |
| 2:40 | **Performance** | Four lead cards. On the Lift tab say *"1.82x is the mathematical ceiling (1 / prevalence), not a perfect model."* On Risk tiers point at the stability table: flagged share does not move. |
| 3:20 | **Robustness** | Per-mode chips (🟢 / 🟠), the ablation bar chart (history adds nothing), then the **Sanity checks** tab: label shuffle ≈ 0.50 and the TRANSFER finding. This is your honesty slide. |
| 4:10 | **Impact Simulator** | Leave the defaults (First Class excluded from rescuable modes), then show the break-even success rate. Say clearly: *"costs and success rates are my assumptions, not measured."* |
| 4:40 | **Methodology** | Pipeline diagram, allowed vs forbidden features, leakage guards. Close with Model 2 (uplift). |

*Backup if the app fails:* `outputs/report_tables.md` and `MODEL_CARD.md` contain every table.

---

## 3. Fifteen likely examiner questions, with honest answers

**1. Is this an RTO model?**
No. The target is `Late_delivery_risk` from the public DataCo dataset: whether delivery exceeded its contracted window. RTO (refusal, NDR, COD failure) is not in this data. I use late delivery as an upstream proxy and I say so on every page. The deliverable is the methodology (leakage-safe features, chronological evaluation, calibration, explanations), to be retrained on real courier and return data.

**2. Why is late delivery a reasonable proxy at all?**
Late or failed delivery attempts are a known driver of refusals and returns, so an early-warning score on delivery risk is the nearest measurable upstream signal. But it is a proxy, and I do not claim the model predicts returns. Whether the proxy transfers to Indian D2C is exactly what Model 2 must test.

**3. Standard Class AUC is 0.53. Is the model useless?**
For ranking *inside* Standard Class, yes: 0.529 (95% CI 0.514 to 0.544) is near chance, and a model trained only on Standard Class does no better (0.528). That is why the app labels the mode **limited** and tells the user the score mostly reflects the mode's base rate. The model's value is across modes, where it separates Same Day / First Class from the rest.

**4. Then where does the 0.772 AUC come from?**
Mostly from shipping mode and scheduled days (74% of attribution) and the near rule-based labels: Same Day orders placed from 12:00 are late 95% of the time versus 0% before noon. I report that openly rather than presenting 0.77 as skill on real courier behaviour.

**5. History features add nothing. Why did you build them?**
They were a hypothesis (repeat customers and risky countries are later). The ablation tests it: including them changes ROC-AUC by 0.0008 (all features minus no-history), with a paired 95% interval of -0.0017 to 0.0033, which includes zero. So the honest finding is "no significant gain on this data". I kept them because they are leakage-safe and cost nothing, and they will matter on real customer data; I do not claim they help here.

**6. Explain the TRANSFER finding.**
At first TRANSFER looked protective (late rate 48.2% vs 57.3%). Checking order status showed all 2,855 CANCELED / SUSPECTED_FRAUD orders are TRANSFER payments and every one is labelled on-time. Removing them makes the within-mode gap go from -9.1 pp to -0.0 pp. So it is a labelling artefact, not payment behaviour. A further honest point: in this dataset every order status occurs with exactly one payment type, so payment type is partly a proxy for post-dispatch status. That is why I never interpret it causally, and dropping it costs 0.016 AUC.

**7. Why not a random train/test split?**
Orders are a time series and the customer-history features look backwards. A random split would let the model train on later orders and be tested on earlier ones, and would put future outcomes into "history". I split strictly by time: train to 2017-03-16, validation to 2017-09-05, test to 2018-01-31. A 5-fold rolling-origin backtest gives AUC 0.770 ± 0.007, so the result is not an artefact of one cut.

**8. How did you choose the thresholds, and are they stable?**
Calibrated scores fall in a few clusters (only 41 distinct values on test), so a cutoff that sits on a cluster flips thousands of orders with a tiny shift. I place cutoffs at midpoints of low-density gaps, chosen on validation only, and verify them on test. Shifting any threshold by ±0.02 moves the flagged share by at most 2.5 points (High tier and precision ≥ 0.80: 0.0), against a hard limit of 5. An earlier version of the precision threshold sat on a cluster edge and was fragile; that is why I redesigned the cutoffs.

**9. Is the model calibrated? How do you know?**
Isotonic regression fitted on the validation set cuts test ECE from 0.050 to 0.007 (Brier 0.186 → 0.183). Two caveats I volunteer: validation ECE after calibration is ~0 by construction (so only the test figure counts), and calibration is good overall but not guaranteed inside every mode; Same Day, for example, is off by -5.2 pp (mean predicted minus actual).

**10. Why LightGBM, and how much did tuning help?**
Five models were compared on the same split; the tree models were within a few thousandths of AUC of each other, and LightGBM was best on validation. Tuning moved validation ROC-AUC from 0.764 to 0.768: a small gain, chosen on validation with time-series cross-validation, never on test. Ten seeds give ROC-AUC 0.771 ± 0.001.

**11. How do you know there is no leakage?**
(a) Forbidden columns (post-dispatch outcomes, PII, the secondary target) are excluded and tested. (b) History features only count orders whose outcome was already known before the current order; a brute-force pytest recomputes them from the raw data. (c) Encoders and scalers are fitted on train only. (d) A label-shuffle test gives AUC 0.496, i.e. chance. The limit of this claim: these checks rule out *my* pipeline leaking; they cannot rule out leakage built into the public dataset's labels, which is exactly what the near rule-based labels and the TRANSFER artefact are.

**12. High-tier precision is 88%. Is that real skill?**
Partly base rate. The top decile is 100.0% First Class, and First Class orders are 95% late anyway, so a lot of that precision is the dispatch rule. The tier is still useful operationally (it flags 32% of orders at 88% precision vs 55% overall), but I would not present it as a discovery about couriers.

**13. Flag-all gets F1 0.71. Why not just flag everything?**
Because F1 rewards it when most orders are late (55%). Balanced accuracy for flag-all is 0.50 and MCC 0.00, versus 0.72 and 0.46 for the High tier. That is why F1 is never a headline here.

**14. Did the test set influence any choice?**
No selection was made on it: hyperparameters use time-series CV on train, the model/calibration/thresholds/tiers use validation, and the test set is used to report. One nuance I volunteer: the tier script *checks* the chosen cutoffs against test scores and stops if they fail; it never picks a value from test, but I am transparent that this is a test-based pass/fail guard. SHAP and the dashboard also read test data, for display only.

**15. Are the Impact Simulator dollars real? And what next?**
No. Cost of a failed delivery, cost per intervention and success rates are user-set assumptions, and the app says so. First Class lateness comes from the carrier SLA, so by default it is excluded from rescuable orders. Measuring the real effect of an intervention is Model 2 (uplift modelling) on real courier / NDR / return outcomes. Until then the simulator shows break-even rates, not forecasts.
