"""
src/09b_type_audit.py
=====================
Step 9b - Payment-type (Type) proxy audit: how much of the model's ranking comes from a labelling artifact?

Facts established by 09_verification.py: every CANCELED / SUSPECTED_FRAUD order is a TRANSFER payment and is labelled
on-time, so `Type` partly proxies post-dispatch order status. Order Status is NEVER a model feature and
`aux_target_cancel_fraud` is used here for EVALUATION ONLY (to define the "cleaned" test subset).

  Model A = the current final model (with the payment features).
  Model B = the SAME tuned hyper-parameters, retrained on TRAIN only, WITHOUT the payment features. The drop list is
            imported from ABLATION_SPECS["No payment/is_cash"] in 06_robustness.py (not redefined here).

Both are evaluated on (i) all test orders and (ii) test orders with aux_target_cancel_fraud == 0:
RAW-score ROC-AUC and PR-AUC with 95% bootstrap CIs and the paired-bootstrap delta (A minus B) with its 95% CI.
Model B also gets its OWN tiers (cutoffs chosen on validation with tiering.py, verified on test).

Writes outputs/type_audit.json (copied into outputs/final_metrics.json by 10_finalize.py) and outputs/type_audit.csv.
Does not modify any model, calibrator, threshold or app file.

Usage:  python src/09b_type_audit.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (CATEGORICAL_FEATURES, MODELS_DIR, NUMERIC_FEATURES, OUTPUTS_DIR, PROCESSED_DIR, SEED,  # noqa: E402
                    TARGET_COL, TRAIN_FRAC, VAL_FRAC)
from features import _select_feature_cols, build_feature_matrix  # noqa: E402
import tiering  # noqa: E402

N_BOOT = 1000
MIN_USEFUL_DELTA = 0.005        # decision rule: the cleaned-order delta must be at least this AND its CI must exclude 0
AUX_COL = "aux_target_cancel_fraud"
OUT_JSON, OUT_CSV = OUTPUTS_DIR / "type_audit.json", OUTPUTS_DIR / "type_audit.csv"


def load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fast_auc(y, s) -> float:
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap(y, sa, sb, n_boot=N_BOOT, seed=0) -> dict:
    """Point estimates + percentile CIs for A, B and the paired delta A-B (identical resample indices)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    keys = ("roc_a", "roc_b", "pr_a", "pr_b")
    draws = {k: np.empty(n_boot) for k in keys}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        draws["roc_a"][i], draws["roc_b"][i] = fast_auc(yb, sa[idx]), fast_auc(yb, sb[idx])
        draws["pr_a"][i], draws["pr_b"][i] = average_precision_score(yb, sa[idx]), average_precision_score(yb, sb[idx])

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    pt = {"roc_a": roc_auc_score(y, sa), "roc_b": roc_auc_score(y, sb),
          "pr_a": average_precision_score(y, sa), "pr_b": average_precision_score(y, sb)}
    return {
        "n": int(n), "prevalence": float(y.mean()),
        "model_a": {"roc_auc": float(pt["roc_a"]), "roc_auc_ci": ci(draws["roc_a"]),
                    "pr_auc": float(pt["pr_a"]), "pr_auc_ci": ci(draws["pr_a"])},
        "model_b": {"roc_auc": float(pt["roc_b"]), "roc_auc_ci": ci(draws["roc_b"]),
                    "pr_auc": float(pt["pr_b"]), "pr_auc_ci": ci(draws["pr_b"])},
        "delta_roc_auc": {"point": float(pt["roc_a"] - pt["roc_b"]), "ci": ci(draws["roc_a"] - draws["roc_b"])},
        "delta_pr_auc": {"point": float(pt["pr_a"] - pt["pr_b"]), "ci": ci(draws["pr_a"] - draws["pr_b"])},
    }


def equal_coverage(y, flag_a, raw_b, n_boot=N_BOOT, seed=1) -> dict:
    """
    Like-for-like High-tier comparison: Model B flags its top-k raw scores with k = the number of orders in Model A's
    High tier (within the same subset). Paired bootstrap (identical resample indices) of the precision / recall difference.
    """
    y = np.asarray(y)
    k = int(flag_a.sum())
    flag_b = np.zeros(len(y), dtype=bool)
    flag_b[np.argsort(-raw_b, kind="stable")[:k]] = True

    def pr(f, yy):
        tp = int((f & (yy == 1)).sum())
        return (tp / f.sum() if f.sum() else 0.0), (tp / (yy == 1).sum() if (yy == 1).sum() else 0.0)

    rng = np.random.default_rng(seed)
    dp, dr = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        pa, ra = pr(flag_a[idx], y[idx])
        pb, rb = pr(flag_b[idx], y[idx])
        dp[i], dr[i] = pa - pb, ra - rb
    pa, ra = pr(flag_a, y)
    pb, rb = pr(flag_b, y)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"k": k, "n": int(len(y)), "flagged_share": float(k / len(y)),
            "model_a": {"precision": pa, "recall": ra}, "model_b": {"precision": pb, "recall": rb},
            "delta_precision": {"point": pa - pb, "ci": ci(dp)}, "delta_recall": {"point": ra - rb, "ci": ci(dr)}}


def tier_block(y_val, cal_val, y_te, cal_te, low, high, mask_clean) -> dict:
    """High-tier precision / recall / flagged share on all and on cleaned test orders for one set of cutoffs."""
    def pol(y, s):
        r = tiering.policy_row(y, s, high)
        return {"precision": r["precision"], "recall": r["recall"], "flagged_share": r["flagged_share_pct"] / 100.0,
                "flagged_orders": r["flagged_orders"]}
    chk = tiering.cutoff_check(cal_te, high)
    return {"cutoffs": {"low_max": float(low), "high_min": float(high)},
            "cutoff_verified_on_test": {"in_gap": bool(chk["in_gap"]), "big_clusters_split": int(chk["big_clusters_split"]),
                                        "share_within_0.02": float(chk["share_within_0.02"])},
            "all_orders": pol(y_te, cal_te), "cleaned_orders": pol(y_te[mask_clean], cal_te[mask_clean])}


def main() -> None:
    print("\n" + "=" * 70 + "\n  RTO Shield AI - Step 9b: payment-type proxy audit\n" + "=" * 70)
    m06 = load_mod("m06_for_audit", "06_robustness.py")
    spec = next(s for s in m06.ABLATION_SPECS if s["label"] == "No payment/is_cash")
    cat_b, num_b = list(spec["cat"]), list(spec["num"])
    dropped = [c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c not in cat_b + num_b]
    print(f"  drop list (from 06_robustness ABLATION_SPECS): {dropped}")

    orders = pd.read_parquet(PROCESSED_DIR / "orders.parquet").sort_values("order_date").reset_index(drop=True)
    n = len(orders)
    a, b = int(np.floor(n * TRAIN_FRAC)), int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    tr, va, te = orders.iloc[:a].copy(), orders.iloc[a:b].copy(), orders.iloc[b:].copy()
    y_tr, y_va, y_te = (d[TARGET_COL].values for d in (tr, va, te))
    clean = (te[AUX_COL].values == 0)

    # ---- Model A: the saved final model, prep and calibrator -------------------------------------------------
    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model_a = joblib.load(MODELS_DIR / "final_model.joblib")
    cal_a = joblib.load(MODELS_DIR / "final_calibrator.joblib")
    thr = json.loads((MODELS_DIR / "thresholds.json").read_text(encoding="utf-8"))
    xs = lambda d: prep.transform(_select_feature_cols(d, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    raw_a_va, raw_a_te = model_a.predict_proba(xs(va))[:, 1], model_a.predict_proba(xs(te))[:, 1]

    # ---- Model B: same tuned hyper-parameters, TRAIN only, without the payment features ----------------------
    params = model_a.get_params()
    Xb_tr, (Xb_va, Xb_te), _, _ = build_feature_matrix(tr, [va, te], cat_cols=cat_b, num_cols=num_b, save_preprocessor=False)
    model_b = lgb.LGBMClassifier(**params)
    model_b.fit(Xb_tr, y_tr)
    raw_b_va, raw_b_te = model_b.predict_proba(Xb_va)[:, 1], model_b.predict_proba(Xb_te)[:, 1]

    # Symmetry check: Model A's recipe re-fit on TRAIN only (the saved model used validation early stopping)
    Xa_tr, (Xa_te,), _, _ = build_feature_matrix(tr, [te], cat_cols=CATEGORICAL_FEATURES, num_cols=NUMERIC_FEATURES, save_preprocessor=False)
    refit_a = lgb.LGBMClassifier(**params).fit(Xa_tr, y_tr)
    a_refit_auc = float(roc_auc_score(y_te, refit_a.predict_proba(Xa_te)[:, 1]))

    # ---- Evaluation -------------------------------------------------------------------------------------------
    res_all = bootstrap(y_te, raw_a_te, raw_b_te)
    res_clean = bootstrap(y_te[clean], raw_a_te[clean], raw_b_te[clean])
    res_clean["removed_orders"] = int((~clean).sum())

    # ---- Tiers: A (thresholds.json) vs B (its own, chosen on validation) ---------------------------------------
    cal_a_va, cal_a_te = cal_a.predict(raw_a_va), cal_a.predict(raw_a_te)
    tiers = {"model_a": tier_block(y_va, cal_a_va, y_te, cal_a_te, thr["tier_cutoffs"]["low_max"], thr["tier_cutoffs"]["high_min"], clean)}
    try:
        ir_b = IsotonicRegression(out_of_bounds="clip").fit(raw_b_va, y_va)
        cal_b_va, cal_b_te = ir_b.predict(raw_b_va), ir_b.predict(raw_b_te)
        low_b, high_b, _ = tiering.select_tier_cutoffs(cal_b_va)
        tiers["model_b"] = tier_block(y_va, cal_b_va, y_te, cal_b_te, low_b, high_b, clean)
        tiers["model_b"]["status"] = "ok"
    except tiering.CutoffError as exc:
        tiers["model_b"] = {"status": f"cutoff selection failed: {exc}"}

    flag_a = cal_a_te >= thr["tier_cutoffs"]["high_min"]
    eq = {"all_orders": equal_coverage(y_te, flag_a, raw_b_te),
          "cleaned_orders": equal_coverage(y_te[clean], flag_a[clean], raw_b_te[clean])}

    d = res_clean["delta_roc_auc"]
    includes_zero = bool(d["ci"][0] <= 0 <= d["ci"][1])
    below_min = bool(d["point"] < MIN_USEFUL_DELTA)
    out = {
        "description": "Model A = final model (with payment features); Model B = same tuned hyper-parameters, TRAIN only, without payment features.",
        "drop_list": {"source": "06_robustness.ABLATION_SPECS['No payment/is_cash']", "dropped_features": dropped,
                      "kept_categorical": cat_b, "kept_numeric": num_b},
        "hyperparameters": {k: params[k] for k in ("learning_rate", "max_depth", "n_estimators", "colsample_bytree", "subsample",
                                                   "min_child_weight", "reg_lambda")},
        "n_bootstrap": N_BOOT, "min_useful_delta": MIN_USEFUL_DELTA,
        "all_orders": res_all, "cleaned_orders": res_clean,
        "tiers": tiers,
        "equal_coverage": eq,
        "model_a_refit_train_only_roc_auc_all": a_refit_auc,
        "cleaned_delta_ci_includes_zero": includes_zero,
        "cleaned_delta_below_min": below_min,
        "stop_and_review": bool(includes_zero or below_min),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    rows = []
    for subset, r in (("all test orders", res_all), ("cleaned (aux_target_cancel_fraud == 0)", res_clean)):
        for m in ("model_a", "model_b"):
            rows.append({"subset": subset, "model": "A (with Type)" if m == "model_a" else "B (without payment)", "n": r["n"],
                         "roc_auc": r[m]["roc_auc"], "roc_ci_lo": r[m]["roc_auc_ci"][0], "roc_ci_hi": r[m]["roc_auc_ci"][1],
                         "pr_auc": r[m]["pr_auc"], "pr_ci_lo": r[m]["pr_auc_ci"][0], "pr_ci_hi": r[m]["pr_auc_ci"][1]})
        for k, lab in (("delta_roc_auc", "delta ROC-AUC (A-B)"), ("delta_pr_auc", "delta PR-AUC (A-B)")):
            rows.append({"subset": subset, "model": lab, "n": r["n"], "roc_auc" if "roc" in k else "pr_auc": r[k]["point"],
                         "roc_ci_lo" if "roc" in k else "pr_ci_lo": r[k]["ci"][0], "roc_ci_hi" if "roc" in k else "pr_ci_hi": r[k]["ci"][1]})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # ---- console table ------------------------------------------------------------------------------------------
    print(f"\n  {'subset':<42}{'model':<24}{'ROC-AUC [95% CI]':>30}{'PR-AUC [95% CI]':>30}")
    for r in rows:
        if r["model"].startswith("delta"):
            k = "roc" if "ROC" in r["model"] else "pr"
            v, lo, hi = r.get(f"{k}_auc"), r.get(f"{k}_ci_lo"), r.get(f"{k}_ci_hi")
            print(f"  {r['subset']:<42}{r['model']:<24}{v:>+10.4f} [{lo:+.4f}, {hi:+.4f}]")
        else:
            print(f"  {r['subset']:<42}{r['model']:<24}{r['roc_auc']:>10.4f} [{r['roc_ci_lo']:.4f}, {r['roc_ci_hi']:.4f}]"
                  f"{r['pr_auc']:>10.4f} [{r['pr_ci_lo']:.4f}, {r['pr_ci_hi']:.4f}]")
    print(f"\n  Model A refit on TRAIN only (symmetry check): test ROC-AUC {a_refit_auc:.4f}")
    for m, t in tiers.items():
        if t.get("status", "ok") != "ok":
            print(f"  {m} tiers: {t['status']}")
            continue
        print(f"  {m} High tier (cutoff {t['cutoffs']['high_min']:.4f}): all orders precision {t['all_orders']['precision']:.3f}, "
              f"recall {t['all_orders']['recall']:.3f}, flags {t['all_orders']['flagged_share']:.1%}; cleaned orders precision "
              f"{t['cleaned_orders']['precision']:.3f}, recall {t['cleaned_orders']['recall']:.3f}, flags {t['cleaned_orders']['flagged_share']:.1%}")
    for sub, e in eq.items():
        print(f"  equal coverage ({sub}, k={e['k']:,} = {e['flagged_share']:.1%}): A precision {e['model_a']['precision']:.3f} vs B top-k "
              f"{e['model_b']['precision']:.3f}; delta {e['delta_precision']['point']:+.3f} [{e['delta_precision']['ci'][0]:+.3f}, "
              f"{e['delta_precision']['ci'][1]:+.3f}]; recall A {e['model_a']['recall']:.3f} vs B {e['model_b']['recall']:.3f}")
    print(f"\n  DECISION RULE: cleaned delta {d['point']:+.4f} CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] -> "
          f"{'STOP AND REVIEW (CI includes 0 or delta < ' + str(MIN_USEFUL_DELTA) + ')' if out['stop_and_review'] else 'proceed'}")
    print(f"  Wrote {OUT_JSON} and {OUT_CSV}")


if __name__ == "__main__":
    main()
