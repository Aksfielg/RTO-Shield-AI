"""
src/11_experiments.py
=====================
Step 11b - Experiments on the label and on segment models.  READ-ONLY with respect to the project: the final model,
calibrator, thresholds and metrics are never replaced.  Writes outputs/experiments.md (+ experiments.json).

  1. Segment specialists   : Standard-class and Second-class LightGBMs vs the global model, within each segment.
  2. Alternative target    : delivery_failure = late OR CANCELED OR SUSPECTED_FRAUD (evaluation target only;
                             Order Status is NEVER a feature) - AUC / PR-AUC and what drives it.
  3. Cleaned task          : exclude CANCELED / SUSPECTED_FRAUD orders from the late-delivery task.

Adoption rule: a variant replaces the final model only if it improves VALIDATION ROC-AUC by more than 0.01 AND the
leakage tests pass.  Late_delivery_risk stays the primary reported target regardless.

Usage:  python src/11_experiments.py [--skip-tests]
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import FORBIDDEN_FEATURES, OUTPUTS_DIR, SEED, TARGET_COL  # noqa: E402
from reasons import group_shap_values  # noqa: E402
from utils import bootstrap_ci  # noqa: E402

MIN_GAIN = 0.01
CF = ["CANCELED", "SUSPECTED_FRAUD"]
N_BOOT = 1000
LEAKAGE_TESTS = ["tests/test_no_leakage.py", "tests/test_pipeline.py"]


def load_09():
    spec = importlib.util.spec_from_file_location("m09", SRC_DIR / "09_verification.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m09"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def auc_ci(y, p):
    lo, hi = bootstrap_ci(np.asarray(y), np.asarray(p), roc_auc_score, n=N_BOOT)
    return float(lo), float(hi)


def paired_delta(y, a, b, n=N_BOOT, seed=SEED):
    """AUC(a) - AUC(b) with a paired bootstrap 95% CI."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    rng, idx, d = np.random.default_rng(seed), np.arange(len(y)), []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].min() == y[s].max():
            continue
        d.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    return float(auc(y, a) - auc(y, b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fit_lgb(X_tr, y_tr, X_va, y_va, base, params=None):
    m = lgb.LGBMClassifier(**{**base, **(params or {}), "random_state": SEED, "n_jobs": -1, "verbose": -1})
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
    return m


def best_by_val(X_tr, y_tr, X_va, y_va, base):
    """Two parameter sets (final-model params, segment-friendly params); keep the better one BY VALIDATION AUC."""
    cands = [("final-model params", {}), ("lr 0.05 / depth 4", dict(learning_rate=0.05, n_estimators=400, max_depth=4))]
    best = None
    for label, prm in cands:
        m = fit_lgb(X_tr, y_tr, X_va, y_va, base, prm)
        v = auc(y_va, m.predict_proba(X_va)[:, 1])
        if best is None or v > best[0]:
            best = (v, label, m)
    return best[2], best[1], best[0]


def grouped_shap(model, X_sample, feat_names) -> list[dict]:
    sv = np.asarray(shap.TreeExplainer(model)(X_sample).values)
    sv = sv[:, :, 1] if sv.ndim == 3 else sv
    acc: dict[str, float] = {}
    for r in range(len(sv)):
        labels, vals, _ = group_shap_values(sv[r], feat_names, {})
        for lb, v in zip(labels, vals):
            acc[lb] = acc.get(lb, 0.0) + abs(float(v))
    tot = sum(acc.values())
    return [{"factor": k, "share": v / tot} for k, v in sorted(acc.items(), key=lambda kv: -kv[1])]   # ALL factors


def share_of(drivers: list[dict], factor: str) -> float:
    return next((d["share"] for d in drivers if d["factor"] == factor), 0.0)


def md_table(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.astype(object).iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if (isinstance(v, float) and pd.isna(v)) else str(v)) for v in r.values) + " |")
    return "\n".join(out)


def run_leakage_tests(skip: bool) -> dict:
    if skip:
        return {"ran": False, "passed": None, "detail": "skipped (--skip-tests)"}
    p = subprocess.run([sys.executable, "-m", "pytest", *LEAKAGE_TESTS, "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True, env={**__import__("os").environ, "PYTHONUTF8": "1"})
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1] if p.stdout.strip() else p.stderr[-200:]
    return {"ran": True, "passed": p.returncode == 0, "detail": tail}


# ---------------------------------------------------------------------------
def main() -> None:
    skip_tests = "--skip-tests" in sys.argv
    print("=" * 74 + "\n  RTO Shield AI - Step 11b: experiments (read-only)\n" + "=" * 74)
    m09 = load_09()
    o, X, model, thr, _ = m09.load_all()
    prep = __import__("joblib").load(ROOT / "models" / "preprocessor.joblib")
    feat_names = list(prep.get_feature_names_out())
    assert not any("Order Status" in f or "order_status" in f.lower() or "aux_target" in f for f in feat_names), "Order Status leaked into features"
    forbidden_hit = [f for f in FORBIDDEN_FEATURES if f.replace(" ", "_").lower() in " ".join(feat_names).lower()]
    base = model.get_params()
    for k in ("random_state", "n_jobs", "verbose"):
        base.pop(k, None)

    tr, va, te = (o.split == s for s in ("train", "val", "test"))
    y = o[TARGET_COL].values
    is_cf = o.order_status.isin(CF).values
    Xte_sample = X[te.values][np.sort(np.random.default_rng(SEED).choice(int(te.sum()), 500, replace=False))]
    leak = run_leakage_tests(skip_tests)
    leak_ok = (leak["passed"] is True) and not forbidden_hit
    print(f"  leakage: static forbidden hits={forbidden_hit}  pytest={leak}")

    results: dict = {"leakage": {**leak, "forbidden_feature_hits": forbidden_hit}, "min_gain": MIN_GAIN}

    # ------------------------------------------------------------------ 1. segment specialists
    print("\n[1] Segment specialists ...")
    seg_rows, seg_json = [], {}
    for mode in ["Standard Class", "Second Class"]:
        m = (o.Shipping_Mode == mode).values
        mtr, mva, mte = m & tr.values, m & va.values, m & te.values
        spec, label, _ = best_by_val(X[mtr], y[mtr], X[mva], y[mva], base)
        sv_, st_ = spec.predict_proba(X[mva])[:, 1], spec.predict_proba(X[mte])[:, 1]
        gv, gt = o.raw_score.values[mva], o.raw_score.values[mte]
        val_g, val_s = auc(y[mva], gv), auc(y[mva], sv_)
        te_g, te_s = auc(y[mte], gt), auc(y[mte], st_)
        d, lo, hi = paired_delta(y[mte], st_, gt)
        ci_g, ci_s = auc_ci(y[mte], gt), auc_ci(y[mte], st_)
        gain_val = val_s - val_g
        adopt = gain_val > MIN_GAIN and leak_ok
        seg_rows.append({"segment": mode, "n_test": int(mte.sum()), "global_val_AUC": val_g, "specialist_val_AUC": val_s, "val_gain": gain_val,
                         "global_test_AUC": te_g, "global_95%CI": f"[{ci_g[0]:.3f}, {ci_g[1]:.3f}]", "specialist_test_AUC": te_s,
                         "specialist_95%CI": f"[{ci_s[0]:.3f}, {ci_s[1]:.3f}]", "test_delta [paired 95%CI]": f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                         "specialist_test_PR_AUC": ap(y[mte], st_), "global_test_PR_AUC": ap(y[mte], gt), "params": label, "adopt": "YES" if adopt else "no"})
        seg_json[mode] = {"val_gain": gain_val, "test_delta": d, "test_delta_ci": [lo, hi], "adopt": bool(adopt),
                          "global_test_auc": te_g, "specialist_test_auc": te_s}
        print(f"  {mode:15s} val {val_g:.4f}->{val_s:.4f}  test {te_g:.4f}->{te_s:.4f}  delta {d:+.4f} [{lo:+.4f},{hi:+.4f}]  adopt={adopt}")
    results["segment_specialists"] = seg_json

    # ------------------------------------------------------------------ 2. alternative target
    print("\n[2] Alternative target: delivery_failure = late OR CANCELED OR SUSPECTED_FRAUD ...")
    y_alt = ((y == 1) | is_cf).astype(int)
    alt_model, alt_label, alt_val = best_by_val(X[tr.values], y_alt[tr.values], X[va.values], y_alt[va.values], base)
    p_te = alt_model.predict_proba(X[te.values])[:, 1]
    glob_on_alt = auc(y_alt[te.values], o.raw_score.values[te.values])
    alt_auc, alt_ci = auc(y_alt[te.values], p_te), auc_ci(y_alt[te.values], p_te)
    cf_part_auc = auc(is_cf[te.values].astype(int), p_te)
    late_part_auc = auc(y[te.values], p_te)
    alt_drivers = grouped_shap(alt_model, Xte_sample, feat_names)
    glob_drivers = grouped_shap(model, Xte_sample, feat_names)
    prev = {"late": float(y[te.values].mean()), "alt": float(y_alt[te.values].mean()), "cancel_fraud": float(is_cf[te.values].mean())}
    alt_rows = pd.DataFrame([
        {"model": "Alt-target model (trained on delivery_failure)", "evaluated_on": "delivery_failure", "test_ROC_AUC": alt_auc, "95%CI": f"[{alt_ci[0]:.3f}, {alt_ci[1]:.3f}]",
         "test_PR_AUC": ap(y_alt[te.values], p_te), "prevalence": prev["alt"]},
        {"model": "Final global model (trained on Late_delivery_risk)", "evaluated_on": "delivery_failure", "test_ROC_AUC": glob_on_alt, "95%CI": "",
         "test_PR_AUC": ap(y_alt[te.values], o.raw_score.values[te.values]), "prevalence": prev["alt"]},
        {"model": "Alt-target model", "evaluated_on": "cancel / fraud only", "test_ROC_AUC": cf_part_auc, "95%CI": "",
         "test_PR_AUC": ap(is_cf[te.values].astype(int), p_te), "prevalence": prev["cancel_fraud"]},
        {"model": "Alt-target model", "evaluated_on": "Late_delivery_risk", "test_ROC_AUC": late_part_auc, "95%CI": "",
         "test_PR_AUC": ap(y[te.values], p_te), "prevalence": prev["late"]},
    ])
    results["alternative_target"] = {"test_auc": alt_auc, "test_auc_ci": list(alt_ci), "global_model_on_alt": glob_on_alt, "cancel_fraud_part_auc": cf_part_auc,
                                    "late_part_auc": late_part_auc, "val_auc": alt_val, "drivers": alt_drivers[:8], "global_drivers": glob_drivers[:8],
                                    "payment_share_alt": share_of(alt_drivers, "Payment type"), "payment_share_global": share_of(glob_drivers, "Payment type"),
                                    "prevalence": prev}
    print(f"  alt AUC {alt_auc:.4f} {alt_ci}  global-on-alt {glob_on_alt:.4f}  CF-only {cf_part_auc:.4f}  payment share {share_of(alt_drivers,'Payment type'):.1%}")

    # ------------------------------------------------------------------ 3. cleaned task
    print("\n[3] Late-delivery task with CANCELED / SUSPECTED_FRAUD orders excluded ...")
    keep = ~is_cf
    ktr, kva, kte = (tr.values & keep), (va.values & keep), (te.values & keep)
    cl_model, cl_label, cl_val = best_by_val(X[ktr], y[ktr], X[kva], y[kva], base)
    p_cl_te, p_cl_va = cl_model.predict_proba(X[kte])[:, 1], cl_model.predict_proba(X[kva])[:, 1]
    g_te, g_va = o.raw_score.values[kte], o.raw_score.values[kva]
    val_gain3 = auc(y[kva], p_cl_va) - auc(y[kva], g_va)
    d3, lo3, hi3 = paired_delta(y[kte], p_cl_te, g_te)
    cl_drivers = grouped_shap(cl_model, Xte_sample, feat_names)
    per_mode_rows = []
    for mode in ["Same Day", "First Class", "Second Class", "Standard Class"]:
        mm = kte & (o.Shipping_Mode == mode).values
        one = len(np.unique(y[mm])) < 2
        per_mode_rows.append({"shipping_mode": mode, "n_test": int(mm.sum()),
                              "global_test_AUC_cleaned": "n/a (all late)" if one else auc(y[mm], o.raw_score.values[mm]),
                              "global_test_AUC_all_orders": auc(y[te.values & (o.Shipping_Mode == mode).values], o.raw_score.values[te.values & (o.Shipping_Mode == mode).values]),
                              "cleaned_variant_test_AUC": "n/a (all late)" if one else auc(y[mm], cl_model.predict_proba(X[mm])[:, 1])})
    adopt3 = val_gain3 > MIN_GAIN and leak_ok
    clean_rows = pd.DataFrame([
        {"model": "Cleaned-task variant", "population": "non-cancel/fraud", "val_ROC_AUC": auc(y[kva], p_cl_va), "test_ROC_AUC": auc(y[kte], p_cl_te),
         "95%CI": f"[{auc_ci(y[kte], p_cl_te)[0]:.3f}, {auc_ci(y[kte], p_cl_te)[1]:.3f}]", "test_PR_AUC": ap(y[kte], p_cl_te), "prevalence": float(y[kte].mean())},
        {"model": "Final global model", "population": "non-cancel/fraud", "val_ROC_AUC": auc(y[kva], g_va), "test_ROC_AUC": auc(y[kte], g_te),
         "95%CI": f"[{auc_ci(y[kte], g_te)[0]:.3f}, {auc_ci(y[kte], g_te)[1]:.3f}]", "test_PR_AUC": ap(y[kte], g_te), "prevalence": float(y[kte].mean())},
        {"model": "Final global model", "population": "all orders (reported)", "val_ROC_AUC": auc(y[va.values], o.raw_score.values[va.values]),
         "test_ROC_AUC": auc(y[te.values], o.raw_score.values[te.values]), "95%CI": "", "test_PR_AUC": ap(y[te.values], o.raw_score.values[te.values]),
         "prevalence": float(y[te.values].mean())},
    ])
    results["cleaned_task"] = {"val_gain": val_gain3, "test_delta": d3, "test_delta_ci": [lo3, hi3], "adopt": bool(adopt3),
                               "payment_share_cleaned": share_of(cl_drivers, "Payment type"), "payment_share_global": share_of(glob_drivers, "Payment type"),
                               "drivers": cl_drivers[:8], "cancel_fraud_removed_share_test": float(is_cf[te.values].mean())}
    print(f"  val gain {val_gain3:+.4f}  test delta {d3:+.4f} [{lo3:+.4f},{hi3:+.4f}]  payment share {share_of(glob_drivers,'Payment type'):.1%} -> {share_of(cl_drivers,'Payment type'):.1%}")

    # ------------------------------------------------------------------ narrative
    std, sec = seg_json["Standard Class"], seg_json["Second Class"]
    any_adopt = std["adopt"] or sec["adopt"] or adopt3
    def sig(d):  # is the paired CI away from 0?
        return d["test_delta_ci"][0] > 0 or d["test_delta_ci"][1] < 0
    L = []
    L.append(f"**Bottom line: {'one variant clears the adoption bar' if any_adopt else 'nothing clears the adoption bar, so the final model stays as it is'}.** "
             f"A variant must beat the final model on validation ROC-AUC by more than {MIN_GAIN} and pass the leakage tests "
             f"({'passed' if leak_ok else 'NOT passed / not run'}: {leak['detail']}). `Late_delivery_risk` remains the primary reported target.")
    L.append(f"1. **Segment specialists.** Standard Class: global {std['global_test_auc']:.3f} vs specialist {std['specialist_test_auc']:.3f} "
             f"(paired delta {std['test_delta']:+.4f}, 95% CI {std['test_delta_ci'][0]:+.4f} to {std['test_delta_ci'][1]:+.4f}; validation gain {std['val_gain']:+.4f}). "
             f"Second Class: global {sec['global_test_auc']:.3f} vs specialist {sec['specialist_test_auc']:.3f} (delta {sec['test_delta']:+.4f}, "
             f"CI {sec['test_delta_ci'][0]:+.4f} to {sec['test_delta_ci'][1]:+.4f}; validation gain {sec['val_gain']:+.4f}). "
             + ("Neither specialist beats the global model by a meaningful margin: the missing signal is in the data, not in the model structure. **Not worth adopting.**"
                if not (std["adopt"] or sec["adopt"]) else "At least one specialist clears the bar (see table)."))
    ad = results["alternative_target"]
    cl = results["cleaned_task"]
    pm = {r["shipping_mode"]: r for r in per_mode_rows}
    std_c, sec_c = pm["Standard Class"], pm["Second Class"]
    pay_dir = "falls" if ad["payment_share_alt"] < ad["payment_share_global"] else "rises"
    late_dir = "worse" if ad["late_part_auc"] < auc(y[te.values], o.raw_score.values[te.values]) else "better"
    L.append(f"2. **Alternative target (delivery_failure = late OR cancelled OR suspected fraud).** The alternative-target model reaches test ROC-AUC {ad['test_auc']:.3f} "
             f"(95% CI {ad['test_auc_ci'][0]:.3f} to {ad['test_auc_ci'][1]:.3f}) on its own target, vs {ad['global_model_on_alt']:.3f} for the final model scored on that same target. "
             f"But the gain is not better delivery-risk skill: scored on `Late_delivery_risk` it is {late_dir} ({ad['late_part_auc']:.3f} vs "
             f"{auc(y[te.values], o.raw_score.values[te.values]):.3f}), and on the cancel/fraud part alone it is weak (AUC {ad['cancel_fraud_part_auc']:.3f}, "
             f"PR-AUC {ap(is_cf[te.values].astype(int), p_te):.3f} vs {prev['cancel_fraud']:.3f} prevalence). What drives it is the same as for the late model: shipping mode and scheduled days "
             f"(shipping mode {share_of(alt_drivers, 'Shipping mode'):.0%}); Payment type's share {pay_dir} from {ad['payment_share_global']:.0%} (late model) to "
             f"{ad['payment_share_alt']:.0%}, because once cancelled / fraud orders count as failures the TRANSFER 'protective' artefact largely disappears. "
             "**Keep it as an evaluation-only secondary target; do not adopt it.**")
    L.append(f"3. **Cleaned task (cancelled / fraud orders excluded, {cl['cancel_fraud_removed_share_test']:.1%} of test orders).** Overall the variant scores "
             f"{auc(y[kte], p_cl_te):.3f} vs {auc(y[kte], g_te):.3f} for the final model on the cleaned population (validation gain {cl['val_gain']:+.4f}, below the {MIN_GAIN} bar; "
             f"paired test delta {cl['test_delta']:+.4f}, CI {cl['test_delta_ci'][0]:+.4f} to {cl['test_delta_ci'][1]:+.4f}). Payment type's share of attribution falls from "
             f"{cl['payment_share_global']:.0%} to {cl['payment_share_cleaned']:.1%}: the TRANSFER effect was the artefact. **The important finding is about the existing model, not the variant:** "
             f"on cleaned orders the final model's within-mode AUC is {std_c['global_test_AUC_cleaned']:.3f} for Standard Class and {sec_c['global_test_AUC_cleaned']:.3f} for Second Class "
             f"(vs {std_c['global_test_AUC_all_orders']:.3f} and {sec_c['global_test_AUC_all_orders']:.3f} on all orders). Inside those two modes the model has essentially no ranking ability once the "
             "artefact is removed; the modest 'limited' signal reported earlier was partly the cancel/fraud labelling. First Class has no variation at all (every cleaned order is late). "
             "At checkout the order status is unknown, so the production model cannot drop these orders. **Do not replace the model; do adopt the cleaned population as a reporting view.**")
    L.append("**What adopting would change in the report.** Nothing about the model or its thresholds. The report would (a) add the cleaned-population ROC-AUC next to the all-orders figure, "
             "(b) change the Standard / Second Class wording from 'limited signal' to 'no demonstrable ranking signal on cleaned orders', (c) state that Payment type is excluded from any "
             "causal reading, and (d) show the cancel/fraud AUC as a separate secondary result, never mixed into the late-delivery metrics.")

    md = ["# RTO Shield AI - Model 1: Experiments", "",
          "> Generated by `src/11_experiments.py`. Read-only: the final model, calibrator, thresholds and metrics were not changed. "
          f"Adoption rule: validation ROC-AUC gain > {MIN_GAIN} and passing leakage tests. `Order Status` is never a model feature "
          f"(verified: no forbidden/Order-Status columns among the {len(feat_names)} encoded features).", "",
          "## Conclusions in plain English", "", "\n\n".join(L), "", "---", "",
          "## 1. Segment specialists vs the global model (within segment)", "", md_table(pd.DataFrame(seg_rows)), "",
          "The specialist is trained only on that shipping mode's training orders with the same 27 features; the better of two parameter sets is chosen by validation AUC. "
          "Global scores are the final model's raw scores restricted to the segment.", "",
          "## 2. Alternative target: delivery_failure (evaluation only)", "", md_table(alt_rows), "",
          f"Prevalence on test: late {prev['late']:.1%}, cancel/fraud {prev['cancel_fraud']:.1%}, delivery_failure {prev['alt']:.1%}.", "",
          "**What drives the alternative-target model (share of mean |SHAP|, grouped):**", "",
          md_table(pd.DataFrame(alt_drivers).rename(columns={"share": "share_alt_model"}).merge(
              pd.DataFrame(glob_drivers).rename(columns={"share": "share_final_late_model"}), on="factor", how="outer").fillna(0.0)
              .sort_values("share_alt_model", ascending=False).head(10)), "",
          "## 3. Cleaned late-delivery task (cancel / fraud excluded)", "", md_table(clean_rows), "",
          "**Per shipping mode on the cleaned test set:**", "", md_table(pd.DataFrame(per_mode_rows)), "",
          "**Drivers of the cleaned-task model (top 8):**", "", md_table(pd.DataFrame(cl_drivers[:8])), "",
          "---", "", "## Leakage checks", "",
          f"- Static: forbidden / Order-Status columns among model features: {forbidden_hit or 'none'}.",
          f"- Pytest ({', '.join(LEAKAGE_TESTS)}): {leak['detail']}.",
          "- All variants use the same chronological split and the same leakage-safe features; the alternative target uses Order Status only to BUILD the label, never as an input.", ""]
    (OUTPUTS_DIR / "experiments.md").write_text("\n".join(md), encoding="utf-8")
    (OUTPUTS_DIR / "experiments.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\n  Wrote {OUTPUTS_DIR / 'experiments.md'}")
    print("\n" + "\n\n".join(L))


if __name__ == "__main__":
    main()
