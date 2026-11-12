"""
src/09_verification.py
======================
Step 9 - Verification of label artifacts, payment-type confound, score
clusters, Standard/Second Class headroom and policy fragility.

READ-ONLY with respect to the model, thresholds and app: it only loads the
existing parquet, final model and calibrator, and writes
outputs/verification_findings.md.

Order Status is not in orders.parquet (dropped in 01_prepare), so it is joined
back from the raw CSV by Order Id (read-only).

Usage:
    python src/09_verification.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES, MODELS_DIR, NUMERIC_FEATURES, OUTPUTS_DIR,
    PROCESSED_DIR, RAW_CSV, SEED, TARGET_COL, TRAIN_FRAC, VAL_FRAC,
)
from features import _select_feature_cols  # noqa: E402

OUT_MD = OUTPUTS_DIR / "verification_findings.md"
MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]
CF = ["CANCELED", "SUSPECTED_FRAUD"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.astype(object).iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class Report:
    def __init__(self):
        self.sections: list[str] = []

    def add(self, title: str, body: str):
        text = f"## {title}\n\n{body}\n"
        self.sections.append(text)
        print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
        print(body)


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_all():
    orders = (pd.read_parquet(PROCESSED_DIR / "orders.parquet")
              .sort_values("order_date").reset_index(drop=True))
    n = len(orders)
    tr_end = int(np.floor(n * TRAIN_FRAC))
    val_end = int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    orders["split"] = "train"
    orders.loc[tr_end:val_end - 1, "split"] = "val"
    orders.loc[val_end:, "split"] = "test"

    raw = pd.read_csv(RAW_CSV, encoding="latin-1", low_memory=False,
                      usecols=["Order Id", "Order Status"])
    raw.columns = raw.columns.str.strip()
    nuniq = raw.groupby("Order Id")["Order Status"].nunique()
    status = raw.groupby("Order Id")["Order Status"].first().rename("order_status")
    orders = orders.merge(status, left_on="Order Id", right_index=True, how="left")
    multi = int((nuniq > 1).sum())

    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    cal_path = MODELS_DIR / "final_calibrator.joblib"
    cal = joblib.load(cal_path)
    thr = json.loads((MODELS_DIR / "thresholds.json").read_text(encoding="utf-8"))

    X = prep.transform(_select_feature_cols(orders, CATEGORICAL_FEATURES, NUMERIC_FEATURES))
    raw = model.predict_proba(X)[:, 1]
    orders["raw_score"] = raw                  # ranking metrics (AUC) use RAW scores
    orders["score"] = cal.predict(raw)         # calibrated: clusters, thresholds, decisions
    return orders, X, model, thr, multi


# ---------------------------------------------------------------------------
# 1. label artifact
# ---------------------------------------------------------------------------
def part1(o: pd.DataFrame, multi: int):
    out = []
    out.append(f"Orders with more than one distinct Order Status across line items: {multi} "
               f"(first status used). Orders missing a status after join: {int(o['order_status'].isna().sum())}.\n")

    ct = pd.crosstab(o["order_status"], o["Type"])
    ct["Total"] = ct.sum(axis=1)
    out.append("**Counts: Order Status x Type**\n\n" + md_table(ct.reset_index(), "{:.0f}"))
    pct = ct.drop(columns="Total").div(ct["Total"], axis=0) * 100
    out.append("\n**Row %: share of each status by Type**\n\n" + md_table(pct.round(2).reset_index(), "{:.2f}"))

    st = (o.groupby("order_status")[TARGET_COL].agg(["count", "mean"])
          .rename(columns={"count": "orders", "mean": "late_rate"}).reset_index())
    out.append("\n**Late_delivery_risk rate by Order Status**\n\n" + md_table(st))

    o = o.copy()
    o["is_cf"] = o["order_status"].isin(CF)
    by_type = o.groupby("Type")["is_cf"].agg(["sum", "count", "mean"]).reset_index()
    by_type.columns = ["Type", "cancel_fraud_orders", "orders", "cancel_fraud_share"]
    by_mkt = o.groupby("Market")["is_cf"].agg(["sum", "count", "mean"]).reset_index()
    by_mkt.columns = ["Market", "cancel_fraud_orders", "orders", "cancel_fraud_share"]
    out.append("\n**Cancel/fraud share by Type**\n\n" + md_table(by_type))
    out.append("\n**Cancel/fraud share by Market**\n\n" + md_table(by_mkt))
    late_cf = {s: float(o.loc[o.order_status == s, TARGET_COL].mean()) for s in CF}
    return "\n".join(out), by_type, late_cf


# ---------------------------------------------------------------------------
# 2. payment-type confound
# ---------------------------------------------------------------------------
def type_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODES:
        sub = df[df.Shipping_Mode == m]
        row = {"Shipping_Mode": m}
        for t in ["TRANSFER", "DEBIT", "PAYMENT", "CASH"]:
            s = sub[sub.Type == t]
            row[f"{t}_late"] = s[TARGET_COL].mean() if len(s) else np.nan
            row[f"{t}_n"] = len(s)
        oth = sub[sub.Type != "TRANSFER"]
        row["TRANSFER_minus_others"] = row["TRANSFER_late"] - oth[TARGET_COL].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_gap(tbl: pd.DataFrame) -> float:
    w = tbl["TRANSFER_n"].values.astype(float)
    return float(np.nansum(tbl["TRANSFER_minus_others"].values * w) / w.sum())


def part2(o: pd.DataFrame):
    out = []
    allt = type_by_mode(o)
    clean = o[~o.order_status.isin(CF)]
    cleant = type_by_mode(clean)
    cols = ["Shipping_Mode", "TRANSFER_late", "TRANSFER_n", "DEBIT_late", "DEBIT_n",
            "PAYMENT_late", "PAYMENT_n", "CASH_late", "CASH_n", "TRANSFER_minus_others"]
    out.append("**A. Late rate by Type within shipping mode - ALL orders**\n\n" + md_table(allt[cols]))
    out.append(f"\n**B. Same, CANCELED/SUSPECTED_FRAUD removed** ({len(o) - len(clean):,} orders removed, "
               f"{len(clean):,} remain)\n\n" + md_table(cleant[cols]))

    raw_gap_all = o[o.Type == "TRANSFER"][TARGET_COL].mean() - o[o.Type != "TRANSFER"][TARGET_COL].mean()
    raw_gap_clean = clean[clean.Type == "TRANSFER"][TARGET_COL].mean() - clean[clean.Type != "TRANSFER"][TARGET_COL].mean()
    wg_all, wg_clean = weighted_gap(allt), weighted_gap(cleant)
    gaps = pd.DataFrame([
        {"set": "all orders", "raw_gap(no mode control)": raw_gap_all, "within_mode_gap(TRANSFER-weighted)": wg_all},
        {"set": "cancel/fraud removed", "raw_gap(no mode control)": raw_gap_clean, "within_mode_gap(TRANSFER-weighted)": wg_clean},
    ])
    out.append("\n**C. TRANSFER minus other types, late-rate gap** (negative = TRANSFER later less often)\n\n" + md_table(gaps))

    te = o[o.split == "test"]
    te_clean = te[~te.order_status.isin(CF)]
    auc_all = safe_auc(te[TARGET_COL], te.raw_score)
    auc_clean = safe_auc(te_clean[TARGET_COL], te_clean.raw_score)
    out.append(f"\n**D. Final model test ROC-AUC** (scores NOT retrained): all test orders = {auc_all:.4f} "
               f"(n={len(te):,}); cancel/fraud removed = {auc_clean:.4f} (n={len(te_clean):,}).")
    return "\n".join(out), dict(raw_all=raw_gap_all, raw_clean=raw_gap_clean, wg_all=wg_all,
                                wg_clean=wg_clean, auc_all=auc_all, auc_clean=auc_clean)


# ---------------------------------------------------------------------------
# 3. score clusters
# ---------------------------------------------------------------------------
def part3(o: pd.DataFrame):
    out, info = [], {}
    for sp in ["val", "test"]:
        d = o[o.split == sp]
        vc = d.score.value_counts()
        out.append(f"### {sp.upper()} (n={len(d):,}): {d.score.nunique():,} unique calibrated scores\n")
        top = vc.head(10)
        rows = []
        for sc, cnt in top.items():
            sub = d[d.score == sc]
            mix = sub.Shipping_Mode.value_counts()
            row = {"score": sc, "count": int(cnt), "share_%": cnt / len(d) * 100,
                   "late_rate": sub[TARGET_COL].mean()}
            for m in MODES:
                row[m] = int(mix.get(m, 0))
            rows.append(row)
        out.append(md_table(pd.DataFrame(rows)))
        std = d[d.Shipping_Mode == "Standard Class"]
        big = std.score.value_counts().index[0]
        big_n = int((std.score == big).sum())
        near_all = int(((d.score - big).abs() <= 0.01).sum())
        near_std = int(((std.score - big).abs() <= 0.01).sum())
        out.append(f"\nLargest Standard Class cluster: score {big:.4f} ({big_n:,} Standard orders = "
                   f"{big_n / len(std) * 100:.1f}% of Standard Class). Orders within +/-0.01 of it: "
                   f"{near_all:,} = {near_all / len(d) * 100:.1f}% of all {sp} orders "
                   f"({near_std / len(std) * 100:.1f}% of Standard Class orders).\n")
        info[sp] = dict(uniq=d.score.nunique(), big=big, near_all=near_all / len(d) * 100,
                        near_std=near_std / len(std) * 100)
    return "\n".join(out), info


# ---------------------------------------------------------------------------
# 4. segment models
# ---------------------------------------------------------------------------
def seg_model(o, X, base_params, mode, params_label, params):
    tr = (o.split == "train") & (o.Shipping_Mode == mode)
    va = (o.split == "val") & (o.Shipping_Mode == mode)
    te = (o.split == "test") & (o.Shipping_Mode == mode)
    m = lgb.LGBMClassifier(**{**base_params, **params, "random_state": SEED, "n_jobs": -1, "verbose": -1})
    m.fit(X[tr.values], o.loc[tr, TARGET_COL].values,
          eval_set=[(X[va.values], o.loc[va, TARGET_COL].values)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
    pv = m.predict_proba(X[va.values])[:, 1]
    pt = m.predict_proba(X[te.values])[:, 1]
    yv, yt = o.loc[va, TARGET_COL].values, o.loc[te, TARGET_COL].values
    return dict(params=params_label, n_train=int(tr.sum()), n_test=int(te.sum()),
                best_iter=int(m.best_iteration_ or m.n_estimators),
                val_auc=safe_auc(yv, pv), test_auc=safe_auc(yt, pt), test_pr_auc=safe_ap(yt, pt))


def part4(o, X, model):
    base = model.get_params()
    for k in ("random_state", "n_jobs", "verbose"):
        base.pop(k, None)
    variants = [
        ("final-model params (lr 0.009, 200 trees, depth 4)", {}),
        ("segment params (lr 0.05, <=400 trees, depth 4, early stop)",
         dict(learning_rate=0.05, n_estimators=400, max_depth=4)),
    ]
    out = [
        "Feature set = all 27 existing features. Note: Order_Region, main_item_category, order_dow and "
        "order_month are ALREADY in the final model's feature set (config.py), so no new columns are "
        "added; the only change is that the model is trained on one shipping mode alone.\n",
    ]
    res_all, summary = [], {}
    for mode in ["Standard Class", "Second Class"]:
        te = o[(o.split == "test") & (o.Shipping_Mode == mode)]
        fa = safe_auc(te[TARGET_COL], te.raw_score)
        fp = safe_ap(te[TARGET_COL], te.raw_score)
        rows = []
        for lbl, prm in variants:
            r = seg_model(o, X, base, mode, lbl, prm)
            rows.append(r)
        df = pd.DataFrame(rows)
        df.insert(0, "segment", mode)
        out.append(f"### {mode} (test n={len(te):,}, late rate {te[TARGET_COL].mean():.3f})\n")
        out.append(f"Final model on this segment: ROC-AUC {fa:.4f}, PR-AUC {fp:.4f}\n")
        out.append(md_table(df))
        best = df.loc[df.val_auc.idxmax()]  # chosen on validation, not test
        out.append(f"\nVariant chosen by validation AUC: {best.params} -> test ROC-AUC {best.test_auc:.4f}, "
                   f"PR-AUC {best.test_pr_auc:.4f} (delta vs final {best.test_auc - fa:+.4f}).\n")
        summary[mode] = dict(final_auc=fa, seg_auc=float(best.test_auc), seg_pr=float(best.test_pr_auc),
                             final_pr=fp)
    return "\n".join(out), summary


# ---------------------------------------------------------------------------
# 5. policies
# ---------------------------------------------------------------------------
def policy_metrics(y, flag):
    tn = int(((flag == 0) & (y == 0)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    return dict(
        flagged_share_pct=flag.mean() * 100,
        precision=precision_score(y, flag, zero_division=0),
        recall=recall_score(y, flag, zero_division=0),
        f1=f1_score(y, flag, zero_division=0),
        accuracy=accuracy_score(y, flag),
        balanced_acc=balanced_accuracy_score(y, flag),
        specificity=tn / (tn + fp) if (tn + fp) else 0.0,
        mcc=matthews_corrcoef(y, flag) if len(np.unique(flag)) > 1 else 0.0,
    )


def part5(o, thr):
    te = o[o.split == "test"]
    y, s = te[TARGET_COL].values, te.score.values
    t = thr["thresholds"]
    pols = {
        "Flag all": (None, np.ones(len(y), int)),
        f"High tier (>= {thr['tier_cutoffs']['high_min']:.4f})": (thr["tier_cutoffs"]["high_min"], None),
        f"Precision>=0.80 (>= {t['precision_80']:.4f})": (t["precision_80"], None),
        f"Max-F1 (>= {t['final_tuned_max_f1']:.4f})": (t["final_tuned_max_f1"], None),
    }
    rows, sens, fragility = [], [], {}
    for name, (th, fixed) in pols.items():
        flag = fixed if fixed is not None else (s >= th).astype(int)
        rows.append({"policy": name, **policy_metrics(y, flag)})
        if th is not None:
            for d in (-0.02, -0.01, 0.0, 0.01, 0.02):
                f2 = (s >= th + d).astype(int)
                pm = policy_metrics(y, f2)
                sens.append({"policy": name.split(" (")[0], "threshold": th + d, "shift": d,
                             "flagged_share_pct": pm["flagged_share_pct"],
                             "precision": pm["precision"], "recall": pm["recall"], "f1": pm["f1"]})
            near = (np.abs(s - th) <= 0.01).mean() * 100
            lo = (s >= th - 0.02).mean() * 100
            hi = (s >= th + 0.02).mean() * 100
            fragility[name.split(" (")[0]] = dict(thr=th, near_pct=near, swing=lo - hi)
    body = ("**Policy metrics on TEST** (n=%d)\n\n" % len(y) + md_table(pd.DataFrame(rows)) +
            "\n\n**Threshold sensitivity on TEST** (shift of threshold by +/-0.01, +/-0.02)\n\n" +
            md_table(pd.DataFrame(sens)))
    body += "\n\n**Share of test orders sitting within +/-0.01 of each threshold**\n\n" + md_table(
        pd.DataFrame([{"policy": k, "threshold": v["thr"], "within_0.01_%": v["near_pct"],
                       "flagged_share_swing_-0.02_to_+0.02_pp": v["swing"]} for k, v in fragility.items()]))
    return body, fragility


# ---------------------------------------------------------------------------
# 6. conclusions
# ---------------------------------------------------------------------------
def conclusions(p1late, by_type, p2, p3, p4, frag):
    L = []
    cf_type = by_type.set_index("Type")["cancel_fraud_share"]
    # TRANSFER
    survives = abs(p2["wg_clean"]) >= 0.01 and abs(p2["wg_clean"]) >= 0.5 * abs(p2["wg_all"])
    verdict = ("**survives**, so it is not a cancellation artifact" if survives
               else "**largely disappears**, so it is mostly a cancellation / mix artifact")
    L.append(
        f"1. **TRANSFER.** Raw TRANSFER-vs-others late-rate gap is {p2['raw_all'] * 100:+.1f} pp; controlling for "
        f"shipping mode it is {p2['wg_all'] * 100:+.1f} pp, and with CANCELED/SUSPECTED_FRAUD orders removed it is "
        f"{p2['wg_clean'] * 100:+.1f} pp. The effect {verdict}. Cancel/fraud share by Type: "
        + ", ".join(f"{k} {v * 100:.1f}%" for k, v in cf_type.items())
        + f". Late rate among CANCELED = {p1late['CANCELED'] * 100:.1f}%, SUSPECTED_FRAUD = "
        f"{p1late['SUSPECTED_FRAUD'] * 100:.1f}%. Final-model test ROC-AUC is {p2['auc_all']:.4f} on all orders and "
        f"{p2['auc_clean']:.4f} with cancel/fraud orders removed.")
    # Standard class
    s, sc = p4["Standard Class"], p4["Second Class"]
    helps = s["seg_auc"] - s["final_auc"] >= 0.03
    L.append(
        f"2. **Standard Class.** A segment-only model reaches test ROC-AUC {s['seg_auc']:.4f} / PR-AUC "
        f"{s['seg_pr']:.4f} vs the final model's {s['final_auc']:.4f} / {s['final_pr']:.4f}. "
        + ("That is a meaningful gain." if helps else
           "That is not a meaningful gain: the segment stays close to chance, so the missing signal is not in "
           "these pre-dispatch features (and the 'extra' features were already in the model).")
        + f" Second Class: {sc['seg_auc']:.4f} vs {sc['final_auc']:.4f}.")
    # score clusters
    L.append(
        f"3. **Score clusters.** Calibrated scores are heavily discretised (isotonic steps): "
        f"{p3['val']['uniq']} unique values on validation, {p3['test']['uniq']} on test. "
        f"{p3['test']['near_all']:.1f}% of test orders sit within +/-0.01 of the largest Standard Class cluster "
        f"({p3['test']['big']:.4f}), i.e. {p3['test']['near_std']:.1f}% of Standard Class orders get "
        f"essentially the same score, so ranking inside Standard Class is almost arbitrary.")
    # fragility
    parts = []
    for k, v in frag.items():
        parts.append(f"{k} at {v['thr']:.4f}: {v['near_pct']:.1f}% of orders within +/-0.01, flagged share moves "
                     f"{v['swing']:.1f} pp across a -0.02..+0.02 shift")
    L.append("4. **Threshold fragility.** " + "; ".join(parts) +
             ". A large swing for a small threshold shift means the threshold sits on a score cluster and is fragile.")
    return "\n\n".join(L)


# ---------------------------------------------------------------------------
def main():
    print("Loading data, model and calibrator (read-only) ...")
    o, X, model, thr, multi = load_all()
    print(f"orders={len(o):,}  splits: " + ", ".join(f"{k}={v:,}" for k, v in o.split.value_counts().items()))

    r = Report()
    b1, by_type, p1late = part1(o, multi)
    r.add("1. Label artifact check (Order Status x Type)", b1)
    b2, p2 = part2(o)
    r.add("2. Payment-type confound check", b2)
    b3, p3 = part3(o)
    r.add("3. Score clusters", b3)
    b4, p4 = part4(o, X, model)
    r.add("4. Standard Class / Second Class potential", b4)
    b5, frag = part5(o, thr)
    r.add("5. Policy metrics and threshold fragility", b5)

    concl = conclusions(p1late, by_type, p2, p3, p4, frag)
    print("\n" + "=" * 78 + "\nCONCLUSIONS\n" + "=" * 78 + "\n" + concl)

    header = ("# RTO Shield AI - Model 1: Verification Findings\n\n"
              "> Generated by `src/09_verification.py`. Read-only: no model, threshold or app file was modified.\n\n"
              "## Conclusions\n\n" + concl + "\n\n---\n\n")
    OUT_MD.write_text(header + "\n".join(r.sections), encoding="utf-8")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
