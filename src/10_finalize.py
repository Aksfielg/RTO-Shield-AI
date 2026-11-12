"""
src/10_finalize.py
==================
Step 10 - Compute EVERY number reported anywhere (dashboard, README,
MODEL_CARD, report_tables.md) for the FINAL tuned LightGBM model and write

    outputs/final_metrics.json

Documentation and the app read this file (plus models/thresholds.json) and
contain no hand-typed metrics.  Also regenerates the calibration plot for the
FINAL model.

Requires: 02, 05, 03 already run (final model, calibrator, thresholds.json).
Reuses functions from 03/05/06/08/09 so every number has one implementation.

Usage:  python src/10_finalize.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES, CATEGORICAL_FEATURES,
    MODELS_DIR, NUMERIC_FEATURES, OUTPUTS_DIR, SEED, TARGET_COL,
)
from features import _select_feature_cols  # noqa: E402
from reasons import group_shap_values  # noqa: E402
from utils import compute_metrics, expected_calibration_error  # noqa: E402
import tiering  # noqa: E402

OUT_JSON = OUTPUTS_DIR / "final_metrics.json"
MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]
CF = ["CANCELED", "SUSPECTED_FRAUD"]


def load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if (isinstance(x, float) and np.isnan(x)) or (isinstance(x, np.floating) and np.isnan(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    if isinstance(x, pd.DataFrame):
        return jsonable(x.to_dict(orient="records"))
    return x


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def safe_ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None


def policy_metrics(y, flag) -> dict:
    flag = np.asarray(flag).astype(int)
    tn = int(((flag == 0) & (y == 0)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    return {
        "flagged_orders": int(flag.sum()),
        "flagged_share": float(flag.mean()),
        "precision": float(precision_score(y, flag, zero_division=0)),
        "recall": float(recall_score(y, flag, zero_division=0)),
        "f1": float(f1_score(y, flag, zero_division=0)),
        "accuracy": float(accuracy_score(y, flag)),
        "balanced_accuracy": float(balanced_accuracy_score(y, flag)),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "mcc": float(matthews_corrcoef(y, flag)) if len(np.unique(flag)) > 1 else 0.0,
    }


def calib_bins(y, p, n_bins=10) -> dict:
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    return {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac.tolist()}


def main() -> None:
    print("=" * 74 + "\n  RTO Shield AI - Step 10: Finalize all reported numbers\n" + "=" * 74)

    m03 = load_mod("m03", "03_calibrate_threshold.py")
    m05 = load_mod("m05", "05_tune_multiseed.py")
    m06 = load_mod("m06", "06_robustness.py")
    m08 = load_mod("m08", "08_sanity_checks.py")
    m09 = load_mod("m09", "09_verification.py")

    thr = json.loads((MODELS_DIR / "thresholds.json").read_text(encoding="utf-8"))
    meta = json.loads((OUTPUTS_DIR / "final_results_meta.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ data
    (train_df, val_df, test_df, X_tr, X_val, X_te, y_tr, y_val, y_te, prep, feat_names) = m05._load_splits()
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    cal = joblib.load(MODELS_DIR / "final_calibrator.joblib")
    raw_val = model.predict_proba(X_val)[:, 1]
    raw_te = model.predict_proba(X_te)[:, 1]
    cal_val, cal_te = cal.predict(raw_val), cal.predict(raw_te)

    out: dict = {"meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_scope": "FINAL tuned LightGBM + isotonic calibration only",
        "sources": ["models/thresholds.json", "models/final_model.joblib",
                    "models/final_calibrator.joblib", "outputs/*.csv (02 model comparison)"],
    }}

    # --------------------------------------------------------- dataset/splits
    print("\n[1] Dataset / provenance ...")
    prov = m08.run_provenance_check()
    orders = pd.concat([train_df, val_df, test_df])
    raw_cols = int(pd.read_csv(m08.RAW_CSV, encoding="latin-1", nrows=0).shape[1]) if prov["file_exists"] else None
    out["dataset"] = {
        "raw_rows": prov["row_count"], "raw_columns": raw_cols,
        "n_orders": int(len(orders)), "n_countries": int(orders["Order_Country"].nunique()),
        "date_min": str(orders["order_date"].min().date()), "date_max": str(orders["order_date"].max().date()),
        "positive_rate": float(orders[TARGET_COL].mean()),
        "sha256": prov["sha256"], "file_size_bytes": prov["file_size_bytes"],
        "file_size_mb": prov.get("file_size_mb"),
        "n_parquet_columns": int(orders.shape[1] - 0),
    }
    out["features"] = {
        "n_categorical": len(CATEGORICAL_FEATURES), "n_numeric": len(NUMERIC_FEATURES),
        "n_total": len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES), "n_encoded": len(feat_names),
    }
    out["splits"] = {nm: {"orders": int(len(d)), "date_min": str(d["order_date"].min().date()),
                          "date_max": str(d["order_date"].max().date()),
                          "positive_rate": float(d[TARGET_COL].mean())}
                     for nm, d in [("train", train_df), ("val", val_df), ("test", test_df)]}

    # ------------------------------------------------------------ model block
    print("[2] Model, test metrics, bootstrap CIs, multi-seed ...")
    default_model = joblib.load(MODELS_DIR / "lightgbm.joblib")
    out["model"] = {
        "type": "LightGBM gradient boosted trees (tuned)",
        "hyperparameters": meta["best_params"],
        "cv_roc_auc": meta["best_cv_roc"],
        "val_roc_auc_tuned": float(roc_auc_score(y_val, raw_val)),
        "val_roc_auc_default": float(roc_auc_score(y_val, default_model.predict_proba(X_val)[:, 1])),
        "calibration": "Isotonic regression fitted on validation predictions",
        "n_bootstrap": m05.N_BOOTSTRAP, "n_seeds": m05.N_SEEDS,
    }

    thr_f1 = float(thr["thresholds"]["max_f1"])
    base_model = joblib.load(MODELS_DIR / "baseline_lr.joblib")
    base_prep = joblib.load(MODELS_DIR / "baseline_preprocessor.joblib")
    Xvb = base_prep.transform(_select_feature_cols(val_df, BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES))
    Xtb = base_prep.transform(_select_feature_cols(test_df, BASELINE_CAT_FEATURES, BASELINE_NUM_FEATURES))
    base_raw_te = base_model.predict_proba(Xtb)[:, 1]

    # RANKING metrics (ROC-AUC, PR-AUC, CIs) use RAW scores; calibrated scores are
    # used only for Brier / ECE / thresholds / decisions.
    cis = m05.compute_bootstrap_cis(y_te, raw_te, base_raw_te, n_boot=m05.N_BOOTSTRAP)
    mt = compute_metrics(y_te, cal_te, threshold=thr_f1, y_rank=raw_te)
    out["test"] = {
        "ranking_basis": "raw (uncalibrated) scores; calibrated scores only for Brier, ECE and threshold decisions",
        "roc_auc": float(roc_auc_score(y_te, raw_te)), "roc_auc_ci": list(cis["final_roc_auc"]),
        "pr_auc": float(average_precision_score(y_te, raw_te)), "pr_auc_ci": list(cis["final_pr_auc"]),
        "roc_auc_calibrated_scores": float(roc_auc_score(y_te, cal_te)),
        "pr_auc_calibrated_scores": float(average_precision_score(y_te, cal_te)),
        "delta_roc_auc_ci": list(cis["delta_roc_auc"]), "delta_pr_auc_ci": list(cis["delta_pr_auc"]),
        "baseline_roc_auc": float(roc_auc_score(y_te, base_raw_te)),
        "baseline_pr_auc": float(average_precision_score(y_te, base_raw_te)),
        "brier_raw": float(np.mean((raw_te - y_te) ** 2)), "brier_calibrated": float(np.mean((cal_te - y_te) ** 2)),
        "ece_raw": float(expected_calibration_error(y_te, raw_te)),
        "ece_calibrated": float(expected_calibration_error(y_te, cal_te)),
        "threshold_max_f1": thr_f1, "precision_at_max_f1": mt["precision"], "recall_at_max_f1": mt["recall"],
        "f1_at_max_f1": mt["f1"], "accuracy_at_max_f1": mt["accuracy"],
        "balanced_accuracy_at_max_f1": mt["balanced_accuracy"], "specificity_at_max_f1": mt["specificity"],
        "mcc_at_max_f1": mt["mcc"], "precision_at_top10pct": mt["precision_at_top10pct"],
        "base_rate": float(y_te.mean()), "n": int(len(y_te)),
        "pr_auc_lift": float(average_precision_score(y_te, raw_te) / y_te.mean()),
    }
    out["validation"] = {"roc_auc": float(roc_auc_score(y_val, raw_val)),
                         "ece_raw": float(expected_calibration_error(y_val, raw_val)),
                         "ece_calibrated_in_sample": float(expected_calibration_error(y_val, cal_val))}  # isotonic is fitted on val, so ~0 by construction

    seeds = m05.multi_seed_train(meta["best_params"], X_tr, y_tr, X_val, y_val, X_te, y_te, meta["winner"])
    out["multi_seed"] = {"n_seeds": int(len(seeds)), **{
        f"{k}_{s_}": float(getattr(seeds[k], s_)()) for k in ["roc_auc", "pr_auc", "f1", "recall", "brier"]
        for s_ in ("mean", "std")}}
    # keep outputs/final_results.csv + final_results_meta.json consistent with the raw-score convention
    m05.save_final_results(seeds, mt, cis, thr_f1, meta["winner"], meta["best_params"], meta["best_cv_roc"])

    # ---------------------------------------------------- model comparison (02)
    csv_rows = pd.read_csv(OUTPUTS_DIR / "metrics_model_comparison.csv")
    mc_rows = []
    for name in ["baseline_lr", "logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        mp = MODELS_DIR / f"{name}.joblib"
        if not mp.exists():       # large artefact not available -> keep the 02 CSV rows
            mc_rows += csv_rows[csv_rows["model"] == name].to_dict("records")
            continue
        mdl = joblib.load(mp)
        for split, Xs, ys in [("val", Xvb if name == "baseline_lr" else X_val, y_val),
                              ("test", Xtb if name == "baseline_lr" else X_te, y_te)]:
            mc_rows.append({"model": name, "split": split, **compute_metrics(ys, mdl.predict_proba(Xs)[:, 1], 0.5)})
    out["model_comparison"] = jsonable(pd.DataFrame(mc_rows))

    # ------------------------------------------------------------ calibration
    print("[3] Calibration data + plot (final model) ...")
    out["calibration"] = {"raw": calib_bins(y_te, raw_te), "calibrated": calib_bins(y_te, cal_te)}
    m03.save_calibration_plot(y_te, raw_te, cal_te, OUTPUTS_DIR / "calibration.png", "final tuned LightGBM")

    # ------------------------------------------------------------------ tiers
    print("[4] Tiers, policies, stability ...")
    low, high = thr["tier_cutoffs"]["low_max"], thr["tier_cutoffs"]["high_min"]
    out["tiers"] = {
        "cutoffs": {"low_max": low, "high_min": high},
        "test": m03.tier_stats(y_te, cal_te, low, high),
        "validation": m03.tier_stats(y_val, cal_val, low, high),
        "descriptions": thr["tier_descriptions"],
        "selection_rule": thr["cutoff_selection"]["rule"],
        "candidates": thr["cutoff_selection"]["candidates"],
    }

    p80 = thr["thresholds"]["precision_80"]
    pol_defs = [
        ("Flag all", None, np.ones(len(y_te), int)),
        ("High tier (Operational Flag)", high, (cal_te >= high).astype(int)),
        ("Precision >= 0.80", p80, (cal_te >= p80).astype(int)),
        ("Max-F1", thr_f1, (cal_te >= thr_f1).astype(int)),
    ]
    if "cost_2_1" in thr["thresholds"]:
        c21 = thr["thresholds"]["cost_2_1"]
        pol_defs.append(("Cost-minimising (FN=2xFP)", c21, (cal_te >= c21).astype(int)))
    out["policies"] = [{"policy": n, "threshold": t, **policy_metrics(y_te, f)} for n, t, f in pol_defs]
    out["thresholds"] = thr["thresholds"]

    stab_rows, swings = [], {}
    for name, t in [("High tier", high), ("Precision >= 0.80", p80), ("Max-F1", thr_f1)]:
        rows, sw = tiering.stability_table(cal_te, y_te, t, name)
        stab_rows += rows
        swings[name] = sw
    out["stability"] = {"rows": stab_rows, "swing_pp": swings, "limit_pp": tiering.MAX_FLAG_SWING_PP,
                        "passed": all(v <= tiering.MAX_FLAG_SWING_PP for v in swings.values())}

    # ---------------------------------------------- score clusters / deciles
    cl_te = tiering.clusters(cal_te)
    std_mask = (test_df["Shipping_Mode"] == "Standard Class").values
    std_vals = pd.Series(cal_te[std_mask]).round(10).value_counts()
    big = float(std_vals.index[0])
    comp_top = m08.run_top_decile_composition(m08.load_pipeline_data())
    out["score_clusters"] = {
        "unique_scores_val": int(len(np.unique(cal_val))), "unique_scores_test": int(len(np.unique(cal_te))),
        "largest_standard_cluster_score": big,
        "largest_standard_cluster_share_of_standard": float(std_vals.iloc[0] / std_mask.sum()),
        "share_of_test_within_0.01_of_it": float((np.abs(cal_te - big) <= 0.01).mean()),
        "n_clusters_test": len(cl_te),
        "top_decile": comp_top,
    }
    dt = m03.compute_decile_table(y_te, raw_te)   # ranking -> raw scores
    out["deciles"] = jsonable(dt)
    out["decile_highlights"] = {
        "top_decile_lift": float(dt.loc[0, "lift"]), "top_decile_late_rate": float(dt.loc[0, "late_rate"]),
        "top3_gain_pct": float(dt.loc[2, "cum_gain_pct"]), "top3_volume_pct": float(dt.loc[:2, "n"].sum() / dt["n"].sum() * 100),
    }

    # ---------------------------------------------------------------- per mode
    print("[5] Per-shipping-mode, Same Day rule ...")
    out["per_mode"] = jsonable(m06.per_mode_table(test_df, raw_te, cal_te, y_te))

    sd = test_df[test_df["Shipping_Mode"] == "Same Day"]
    allp = pd.concat([train_df, val_df, test_df])
    sd_all = allp[allp["Shipping_Mode"] == "Same Day"]
    out["same_day_noon_rule"] = {
        "scope": "all orders", "orders": int(len(sd_all)),
        "late_rate_before_noon": float(sd_all[sd_all.order_hour < 12][TARGET_COL].mean()),
        "late_rate_from_noon": float(sd_all[sd_all.order_hour >= 12][TARGET_COL].mean()),
        "test_orders": int(len(sd)),
    }

    # ------------------------------------------------------------------- SHAP
    print("[6] SHAP importance (500-row test sample) ...")
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(X_te), size=min(500, len(X_te)), replace=False))
    sv = shap.TreeExplainer(model)(X_te[idx]).values
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    out["shap"] = {
        "sample_n": int(len(idx)),
        "top_features": [{"feature": feat_names[i].split("__", 1)[-1], "mean_abs_shap": float(mean_abs[i]),
                          "share": float(mean_abs[i] / mean_abs.sum())} for i in order[:10]],
    }
    g_abs: dict[str, float] = {}
    for r in range(len(idx)):
        labels, vals, _ = group_shap_values(sv[r], feat_names, {})
        for lb, v in zip(labels, vals):
            g_abs[lb] = g_abs.get(lb, 0.0) + abs(v)
    tot = sum(g_abs.values())
    grouped = sorted(((k, v / len(idx), v / tot) for k, v in g_abs.items()), key=lambda t: -t[1])
    out["shap"]["grouped"] = [{"factor": k, "mean_abs_shap": a, "share": s} for k, a, s in grouped[:10]]
    out["shap"]["shipping_share"] = float(sum(s for k, _, s in grouped if k in ("Shipping mode", "Scheduled shipping days")))

    # ---------------------------------------------------- robustness (06 / 08)
    print("[7] Ablation, backtest, PSI, label-shuffle ...")
    (orders6, tr6, va6, te6, tr_end, val_end, _) = m06._load_splits()
    abl, abl_verdict = m06.run_ablations(tr6, va6, te6)
    out["ablation"] = jsonable(abl)
    out["ablation_summary"] = {"history_verdict": abl_verdict,
                               "bootstrap_resamples": m06.N_BOOT, "threshold_rule": "each model's own max-F1 threshold on validation"}
    bt = m06.rolling_backtest(orders6, n_folds=5)
    out["backtest"] = {"rows": jsonable(bt), "roc_auc_mean": float(bt.roc_auc.mean()), "roc_auc_std": float(bt.roc_auc.std()),
                       "pr_auc_mean": float(bt.pr_auc.mean()), "pr_auc_std": float(bt.pr_auc.std()),
                       "pr_auc_lift_mean": float(bt.pr_auc_lift.mean()), "pr_auc_lift_std": float(bt.pr_auc_lift.std()),
                       "prevalence_mean": float(bt.prevalence.mean()),
                       "first_test_start": str(bt.test_start.min()), "last_test_end": str(bt.test_end.max())}
    psi = m06.compute_psi(orders6, tr_end, val_end, prep)
    out["psi"] = {"rows": jsonable(psi), "max_psi": float(psi.psi.max())}
    data08 = m08.load_pipeline_data()
    out["label_shuffle"] = m08.run_label_shuffle_test(data08)
    out["trivial_policies"] = m08.run_trivial_policies(data08)

    # ---------------------------------------------------------- verification
    print("[8] Verification block (label artifact, TRANSFER, segment models) ...")
    o, X, model9, thr9, multi = m09.load_all()
    o = o.copy()
    o["is_cf"] = o.order_status.isin(CF)
    te9 = o[o.split == "test"]
    te9c = te9[~te9.order_status.isin(CF)]
    clean = o[~o.is_cf]
    allt, cleant = m09.type_by_mode(o), m09.type_by_mode(clean)
    base_params = model9.get_params()
    for k in ("random_state", "n_jobs", "verbose"):
        base_params.pop(k, None)
    seg = {}
    for mode in ("Standard Class", "Second Class"):
        v1 = m09.seg_model(o, X, base_params, mode, "final-model params", {})
        v2 = m09.seg_model(o, X, base_params, mode, "segment params",
                           dict(learning_rate=0.05, n_estimators=400, max_depth=4))
        best = v1 if v1["val_auc"] >= v2["val_auc"] else v2
        tm = te9[te9.Shipping_Mode == mode]
        seg[mode] = {"final_model_roc_auc": safe_auc(tm[TARGET_COL], tm.raw_score), "final_model_pr_auc": safe_ap(tm[TARGET_COL], tm.raw_score),
                     "segment_model_roc_auc": best["test_auc"], "segment_model_pr_auc": best["test_pr_auc"],
                     "segment_params": best["params"]}
    out["verification"] = {
        "cancel_fraud_orders": int(o.is_cf.sum()),
        "cancel_fraud_share_transfer": float(o[o.Type == "TRANSFER"].is_cf.mean()),
        "cancel_fraud_share_other_types": float(o[o.Type != "TRANSFER"].is_cf.mean()),
        "cancel_fraud_share_by_market": o.groupby("Market").is_cf.mean().to_dict(),
        "late_rate_canceled": float(o[o.order_status == "CANCELED"][TARGET_COL].mean()),
        "late_rate_suspected_fraud": float(o[o.order_status == "SUSPECTED_FRAUD"][TARGET_COL].mean()),
        "late_rate_other_statuses": float(o[~o.is_cf][TARGET_COL].mean()),
        "transfer_gap_raw_all": float(o[o.Type == "TRANSFER"][TARGET_COL].mean() - o[o.Type != "TRANSFER"][TARGET_COL].mean()),
        "transfer_gap_within_mode_all": m09.weighted_gap(allt),
        "transfer_gap_within_mode_clean": m09.weighted_gap(cleant),
        "transfer_late_rate_all": float(o[o.Type == "TRANSFER"][TARGET_COL].mean()),
        "other_types_late_rate_all": float(o[o.Type != "TRANSFER"][TARGET_COL].mean()),
        "test_roc_auc_all": safe_auc(te9[TARGET_COL], te9.raw_score),
        "test_roc_auc_without_cancel_fraud": safe_auc(te9c[TARGET_COL], te9c.raw_score),
        "segment_models": seg,
    }

    out = jsonable(out)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Wrote {OUT_JSON}")

    t = out["tiers"]["test"]
    print("\nTIER TABLE (test)")
    for k in ["Low", "Medium", "High"]:
        print(f"  {k:<7}{t[k]['orders']:>7,}{t[k]['share']:>8.1%}{t[k]['late_rate']:>8.1%}{t[k]['late_captured']:>8.1%}")
    print("\nPOLICY TABLE (test)")
    for p in out["policies"]:
        th = "-" if p["threshold"] is None else f"{p['threshold']:.4f}"
        print(f"  {p['policy']:<28}{th:>8} prec={p['precision']:.4f} rec={p['recall']:.4f} f1={p['f1']:.4f} "
              f"acc={p['accuracy']:.4f} bal={p['balanced_accuracy']:.4f} spec={p['specificity']:.4f} mcc={p['mcc']:.4f}")
    print("\nSTABILITY (flagged-share swing pp):", {k: round(v, 2) for k, v in out["stability"]["swing_pp"].items()},
          "passed =", out["stability"]["passed"])


if __name__ == "__main__":
    main()
