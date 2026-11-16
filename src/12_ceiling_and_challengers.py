"""
src/12_ceiling_and_challengers.py
=================================
How much of the shipped model's test AUC can simple group-mean lookup tables
reproduce, and can per-mode / time-aware LightGBM challengers beat them?

Part A  Group-mean lookup tables (fit on train+val) - TEST ROC-AUC + 95% bootstrap CI
Part B  Per-Shipping_Mode LightGBM (early stopping on val) + random-hash control feature
Part C  Lookup (Shipping_Mode + same_day_pm + Type) vs the shipped model, paired bootstrap
Part D  4-fold expanding-window time check, lookup vs LightGBM

Read-only with respect to models/: nothing in models/ is written or modified.
The only artefact written is outputs/ceiling_and_challengers.md.

Usage
-----
  python src/12_ceiling_and_challengers.py
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    SEED,
    TARGET_COL,
)
from features import build_feature_matrix  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ORDER_PARQUET = PROCESSED_DIR / "orders.parquet"
REPORT_PATH = OUTPUTS_DIR / "ceiling_and_challengers.md"
N_BOOT = 1000
N_BOOT_FOLD = 500
HASH_COL = "cust_hash7"
MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]

GROUPINGS: dict[str, list[str]] = {
    "(a) Shipping_Mode": ["Shipping_Mode"],
    "(b) Shipping_Mode + same_day_pm": ["Shipping_Mode", "same_day_pm"],
    "(c) Shipping_Mode + same_day_pm + Type": ["Shipping_Mode", "same_day_pm", "Type"],
    "(d) (c) + order_hour": ["Shipping_Mode", "same_day_pm", "Type", "order_hour"],
}
LOOKUP_C = GROUPINGS["(c) Shipping_Mode + same_day_pm + Type"]


# ===========================================================================
# Data + split (reuses src/02_train.py so the split is identical by construction)
# ===========================================================================

def _load_train_module():
    spec = importlib.util.spec_from_file_location("train02", SRC_DIR / "02_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["same_day_pm"] = (df["Shipping_Mode"] == "Same Day") & (df["order_hour"] >= 12)
    df[HASH_COL] = (df["Customer_Id"] % 7).astype(int)
    return df


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = add_derived_columns(pd.read_parquet(ORDER_PARQUET))
    return _load_train_module().chrono_split_orders(orders)


# ===========================================================================
# Group-mean lookup
# ===========================================================================

class GroupMeanLookup:
    """Mean of the target per group; unseen groups fall back to the global mean."""

    def __init__(self, cols: list[str]):
        self.cols = list(cols)

    def fit(self, df: pd.DataFrame, y_col: str = TARGET_COL) -> "GroupMeanLookup":
        self.global_mean_ = float(df[y_col].mean())
        self.table_ = df.groupby(self.cols, observed=True)[y_col].mean().rename("_p")
        return self

    def _lookup(self, df: pd.DataFrame) -> pd.Series:
        return df[self.cols].merge(
            self.table_.reset_index(), on=self.cols, how="left", sort=False
        )["_p"]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._lookup(df).fillna(self.global_mean_).to_numpy()

    def unseen_share(self, df: pd.DataFrame) -> float:
        return float(self._lookup(df).isna().mean())


# ===========================================================================
# Bootstrap helpers
# ===========================================================================

def _boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def auc_with_ci(y, s, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    y, s = np.asarray(y), np.asarray(s)
    point = _safe_auc(y, s)
    vals = [_safe_auc(y[i], s[i]) for i in _boot_indices(len(y), n_boot, seed)]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_auc_diff(y, s_a, s_b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """AUC(a) - AUC(b) on the same rows, resampling the SAME row indices for both."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    point = _safe_auc(y, s_a) - _safe_auc(y, s_b)
    diffs = []
    for i in _boot_indices(len(y), n_boot, seed):
        a, b = _safe_auc(y[i], s_a[i]), _safe_auc(y[i], s_b[i])
        diffs.append(a - b)
    diffs = np.asarray(diffs)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi),
            "p_a_better": float(np.nanmean(diffs > 0))}


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


# ===========================================================================
# Part A
# ===========================================================================

def part_a(train_df, val_df, test_df) -> list[dict]:
    trval = pd.concat([train_df, val_df], ignore_index=True)
    y = test_df[TARGET_COL].to_numpy()
    rows = []
    for name, cols in GROUPINGS.items():
        lk = GroupMeanLookup(cols).fit(trval)
        s = lk.predict(test_df)
        auc, lo, hi = auc_with_ci(y, s)
        unseen = lk.unseen_share(test_df)
        rows.append({"name": name, "cols": cols, "n_groups": int(len(lk.table_)),
                     "auc": auc, "lo": lo, "hi": hi, "unseen_share": unseen})
        print(f"  A {name:<42} AUC={fmt_ci(auc, lo, hi)}  groups={len(lk.table_)}")
    return rows


# ===========================================================================
# Part B
# ===========================================================================

def _lgb_params(**over) -> dict:
    p = dict(n_estimators=2000, learning_rate=0.03, max_depth=4, num_leaves=15,
             min_child_samples=20, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
             n_jobs=-1, verbose=-1)
    p.update(over)
    return p


def fit_lgb_early_stop(X_tr, y_tr, X_val, y_val, **over) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(**_lgb_params(**over))
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m


def part_b(train_df, val_df, test_df) -> list[dict]:
    cat, num = CATEGORICAL_FEATURES, NUMERIC_FEATURES + [HASH_COL]
    rows = []
    for mode in MODES:
        tr = train_df[train_df["Shipping_Mode"] == mode]
        va = val_df[val_df["Shipping_Mode"] == mode]
        te = test_df[test_df["Shipping_Mode"] == mode]
        X_tr, (X_va, X_te), names, _ = build_feature_matrix(
            tr, [va, te], cat_cols=cat, num_cols=num, save_preprocessor=False)
        y_tr, y_va, y_te = (d[TARGET_COL].to_numpy() for d in (tr, va, te))
        m = fit_lgb_early_stop(X_tr, y_tr, X_va, y_va)
        s = m.predict_proba(X_te)[:, 1]
        auc, lo, hi = auc_with_ci(y_te, s)

        gain = m.booster_.feature_importance(importance_type="gain")
        share = gain / gain.sum() if gain.sum() > 0 else np.zeros_like(gain)
        order = np.argsort(-share)
        h_idx = names.index(f"num__{HASH_COL}")
        hash_rank = int(np.where(order == h_idx)[0][0]) + 1
        rows.append({
            "mode": mode, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "prevalence": float(y_te.mean()), "best_iter": int(m.best_iteration_ or m.n_estimators_),
            "auc": auc, "lo": lo, "hi": hi,
            "hash_gain_share": float(share[h_idx]), "hash_rank": hash_rank, "n_feat": len(names),
        })
        print(f"  B {mode:<15} AUC={fmt_ci(auc, lo, hi)}  iters={rows[-1]['best_iter']}  "
              f"hash gain={share[h_idx]:.3%} (rank {hash_rank}/{len(names)})")
    return rows


# ===========================================================================
# Part C
# ===========================================================================

def part_c(train_df, val_df, test_df) -> dict:
    y = test_df[TARGET_COL].to_numpy()
    trval = pd.concat([train_df, val_df], ignore_index=True)

    prep = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    X_te = prep.transform(test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    s_model = model.predict_proba(X_te)[:, 1]  # raw scores, as in MODEL_CARD metric conventions

    s_lk = GroupMeanLookup(LOOKUP_C).fit(trval).predict(test_df)
    s_lk_tr = GroupMeanLookup(LOOKUP_C).fit(train_df).predict(test_df)

    out = {
        "model": auc_with_ci(y, s_model),
        "lookup_trval": auc_with_ci(y, s_lk),
        "lookup_train": auc_with_ci(y, s_lk_tr),
        "diff_trval": paired_auc_diff(y, s_lk, s_model),
        "diff_train": paired_auc_diff(y, s_lk_tr, s_model),
        "n_test": len(y),
    }
    print(f"  C shipped model      AUC={fmt_ci(*out['model'])}")
    print(f"  C lookup (train+val) AUC={fmt_ci(*out['lookup_trval'])}  "
          f"diff vs model={out['diff_trval']['diff']:+.4f} "
          f"[{out['diff_trval']['lo']:+.4f}, {out['diff_trval']['hi']:+.4f}]")
    print(f"  C lookup (train)     AUC={fmt_ci(*out['lookup_train'])}  "
          f"diff vs model={out['diff_train']['diff']:+.4f} "
          f"[{out['diff_train']['lo']:+.4f}, {out['diff_train']['hi']:+.4f}]")
    return out


# ===========================================================================
# Part D
# ===========================================================================

def expanding_window_folds(n: int, n_folds: int = 4) -> list[tuple[int, int, int]]:
    """Split n chronologically-ordered rows into n_folds+1 blocks.
    Fold k trains on blocks[:k+1] and tests on block k+1. Returns (train_end, test_start, test_end)."""
    edges = np.linspace(0, n, n_folds + 2).astype(int)
    return [(edges[k + 1], edges[k + 1], edges[k + 2]) for k in range(n_folds)]


def part_d(all_df: pd.DataFrame, n_folds: int = 4) -> list[dict]:
    df = all_df.sort_values("order_date").reset_index(drop=True)
    rows = []
    for k, (tr_end, te_start, te_end) in enumerate(expanding_window_folds(len(df), n_folds), 1):
        train_w, test_w = df.iloc[:tr_end], df.iloc[te_start:te_end]
        y = test_w[TARGET_COL].to_numpy()

        s_lk = GroupMeanLookup(LOOKUP_C).fit(train_w).predict(test_w)

        # LightGBM: preprocessor + trees fit on the first 85% of the window,
        # early stopping on the chronologically last 15% of the window.
        cut = int(np.floor(len(train_w) * 0.85))
        fit_w, val_w = train_w.iloc[:cut], train_w.iloc[cut:]
        X_fit, (X_val, X_te), _, _ = build_feature_matrix(
            fit_w, [val_w, test_w], cat_cols=CATEGORICAL_FEATURES,
            num_cols=NUMERIC_FEATURES, save_preprocessor=False)
        m = fit_lgb_early_stop(X_fit, fit_w[TARGET_COL].to_numpy(), X_val,
                               val_w[TARGET_COL].to_numpy())
        s_gb = m.predict_proba(X_te)[:, 1]

        a_lk = auc_with_ci(y, s_lk, N_BOOT_FOLD, SEED + k)
        a_gb = auc_with_ci(y, s_gb, N_BOOT_FOLD, SEED + k)
        d = paired_auc_diff(y, s_gb, s_lk, N_BOOT_FOLD, SEED + k)   # LightGBM - lookup
        rows.append({
            "fold": k, "n_train": len(train_w), "n_test": len(test_w),
            "train_end": str(train_w["order_date"].max().date()),
            "test_start": str(test_w["order_date"].min().date()),
            "test_end": str(test_w["order_date"].max().date()),
            "lookup": a_lk, "lgbm": a_gb, "diff": d, "best_iter": int(m.best_iteration_ or m.n_estimators_),
        })
        print(f"  D fold {k}: lookup={a_lk[0]:.4f}  lightgbm={a_gb[0]:.4f}  "
              f"diff={d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    return rows


# ===========================================================================
# Report
# ===========================================================================

def _sig(lo: float, hi: float) -> str:
    return "excludes 0" if (lo > 0 or hi < 0) else "includes 0"


def write_report(split_info, A, B, C, D) -> str:
    L: list[str] = []
    L += ["# Ceiling and challengers", "",
          "<!-- GENERATED by src/12_ceiling_and_challengers.py - do not edit by hand. -->", "",
          "Question: how much of the shipped model's test AUC do simple group-mean lookup tables "
          "reproduce, and do per-mode or time-aware LightGBM challengers beat them? "
          "Target is the proxy `Late_delivery_risk` (not RTO). All AUCs are on the TEST split with "
          f"95% percentile bootstrap CIs ({N_BOOT} resamples of test rows).", "",
          "## Setup", "",
          "| Split | Orders | Date range | Positive rate |", "|---|---|---|---|"]
    for name, n, lo, hi, pos in split_info:
        L.append(f"| {name} | {n:,} | {lo} -> {hi} | {pos:.2%} |")
    L += ["", "Chronological 70/15/15 split produced by `chrono_split_orders` from `src/02_train.py`. "
          "`same_day_pm = (Shipping_Mode == 'Same Day') & (order_hour >= 12)`. "
          "Lookup tables are plain group means (no smoothing); a group unseen in the fit data "
          "falls back to the global mean.", ""]

    L += ["## Part A - Group-mean lookup tables (fit on train+val)", "",
          "| Grouping | Groups | ROC-AUC [95% CI] | Unseen test rows |", "|---|---|---|---|"]
    for r in A:
        L.append(f"| {r['name']} | {r['n_groups']} | {fmt_ci(r['auc'], r['lo'], r['hi'])} | {r['unseen_share']:.2%} |")
    L.append("")

    L += ["## Part B - Per-Shipping_Mode LightGBM with random-hash control", "",
          f"Repo features (`CATEGORICAL_FEATURES` + `NUMERIC_FEATURES`) plus control `{HASH_COL} = Customer_Id % 7`, "
          "preprocessor and trees fit on that mode's train rows only, early stopping (50 rounds, AUC) on that mode's validation rows. "
          "A useful model should rank the control near the bottom; a high gain share means the trees are fitting noise.", "",
          "| Mode | n train / val / test | Test prevalence | Trees | ROC-AUC [95% CI] | Control gain share | Control rank |",
          "|---|---|---|---|---|---|---|"]
    for r in B:
        L.append(f"| {r['mode']} | {r['n_train']:,} / {r['n_val']:,} / {r['n_test']:,} | {r['prevalence']:.1%} | "
                 f"{r['best_iter']} | {fmt_ci(r['auc'], r['lo'], r['hi'])} | {r['hash_gain_share']:.2%} | "
                 f"{r['hash_rank']} of {r['n_feat']} |")
    L.append("")

    L += ["## Part C - 3-column lookup vs shipped model (same test rows)", "",
          "Lookup = Shipping_Mode + same_day_pm + Type. Shipped model = `models/final_model.joblib` raw scores "
          "(preprocessor from `models/preprocessor.joblib`). Paired bootstrap resamples the same test rows for both scorers; "
          "diff = AUC(lookup) - AUC(shipped).", "",
          "| Scorer | Fit data | Test ROC-AUC [95% CI] |", "|---|---|---|",
          f"| Shipped LightGBM | train | {fmt_ci(*C['model'])} |",
          f"| 3-column lookup | train+val | {fmt_ci(*C['lookup_trval'])} |",
          f"| 3-column lookup | train only | {fmt_ci(*C['lookup_train'])} |", "",
          "| Comparison | AUC difference [95% CI] | CI vs 0 | Share of resamples lookup better |", "|---|---|---|---|"]
    for label, k in (("Lookup (train+val) - shipped", "diff_trval"),
                     ("Lookup (train only) - shipped", "diff_train")):
        d = C[k]
        L.append(f"| {label} | {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | {_sig(d['lo'], d['hi'])} | {d['p_a_better']:.1%} |")
    L += ["", "The train+val lookup sees 15% more (and more recent) data than the shipped model, so the train-only row is the "
          "like-for-like comparison.", ""]

    L += ["## Part D - 4-fold expanding-window time check", "",
          "Train+val+test rows pooled and sorted by `order_date`, cut into 5 equal blocks. Fold k trains on blocks 1..k and tests on block k+1. "
          "Lookup = Shipping_Mode + same_day_pm + Type fit on the whole training window. LightGBM = all repo features, "
          "preprocessor and trees fit on the first 85% of the window, early stopping on its last 15%. "
          f"CIs use {N_BOOT_FOLD} bootstrap resamples; diff = AUC(LightGBM) - AUC(lookup), paired.", "",
          "| Fold | Train n (to) | Test n (window) | Lookup AUC [95% CI] | LightGBM AUC [95% CI] | Diff [95% CI] | CI vs 0 |",
          "|---|---|---|---|---|---|---|"]
    for r in D:
        d = r["diff"]
        L.append(f"| {r['fold']} | {r['n_train']:,} ({r['train_end']}) | {r['n_test']:,} ({r['test_start']} -> {r['test_end']}) | "
                 f"{fmt_ci(*r['lookup'])} | {fmt_ci(*r['lgbm'])} | {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | {_sig(d['lo'], d['hi'])} |")
    mean_lk = np.mean([r["lookup"][0] for r in D])
    mean_gb = np.mean([r["lgbm"][0] for r in D])
    wins = sum(r["diff"]["diff"] > 0 for r in D)
    L += ["", f"Mean over folds: lookup {mean_lk:.4f}, LightGBM {mean_gb:.4f} (LightGBM ahead in {wins} of {len(D)} folds).", ""]

    text = "\n".join(L) + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"\n  Report written -> {REPORT_PATH}")
    return text


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  RTO Shield AI - Ceiling and challengers")
    print("=" * 70)

    train_df, val_df, test_df = load_splits()
    split_info = [(n, len(d), d["order_date"].min().date(), d["order_date"].max().date(),
                   d[TARGET_COL].mean())
                  for n, d in (("Train", train_df), ("Validation", val_df), ("Test", test_df))]
    for s in split_info:
        print(f"  {s[0]:<11} n={s[1]:,}  {s[2]} -> {s[3]}  pos={s[4]:.4f}")

    print("\n[Part A] Group-mean lookups")
    A = part_a(train_df, val_df, test_df)
    print("\n[Part B] Per-mode LightGBM + hash control")
    B = part_b(train_df, val_df, test_df)
    print("\n[Part C] Lookup vs shipped model")
    C = part_c(train_df, val_df, test_df)
    print("\n[Part D] Expanding-window check")
    D = part_d(pd.concat([train_df, val_df, test_df], ignore_index=True))

    write_report(split_info, A, B, C, D)


if __name__ == "__main__":
    main()
