"""
src/12_auc_experiments.py
=========================
Step 12 - Does ANY leakage-free change raise ROC-AUC? (read-only experiment)

Writes outputs/auc_experiments.md and outputs/auc_experiments.json.
It NEVER touches models/final_model.joblib, final_calibrator.joblib, preprocessor.joblib or thresholds.json.

Protocol
--------
* Target = Late_delivery_risk. Chronological split 70/15/15 (same row cut as every other script).
* Tuning/selection on TRAIN only (TimeSeriesSplit); variants are COMPARED on VALIDATION only.
* The TEST set is touched at most ONCE, for the single best variant that already passed every
  validation-side rule (a-e). Test scores of losing variants are never computed or printed.
* Baseline = current tuned LightGBM. Its validation ROC-AUC is read from outputs/final_metrics.json.
  For the paired comparison the same hyper-parameters are refit with the same 5 seeds.
* Every ranking metric uses RAW scores. Every variant: 5 seeds (mean, std) + a paired bootstrap
  (1000 resamples, the SAME resample indices for both models) of the validation delta.

Acceptance (ALL must hold to promote)
  a. validation delta >= +0.005 and the paired-bootstrap 95% CI excludes 0
  b. the 5-seed std of the variant's validation ROC-AUC < delta
  c. label-shuffle AUC within [0.47, 0.53]  (+ the leakage tests pass, run at the end of this script)
  d. 5-fold rolling-backtest mean ROC-AUC (06_robustness.py fold logic) >= the baseline's
  e. Standard Class and Second Class within-mode AUC < 0.60
  f. after re-fitting isotonic calibration on validation, TEST ECE <= 0.015

Usage:  python src/12_auc_experiments.py            (about 20-40 minutes)
        python src/12_auc_experiments.py --quick    (smoke test: few trials, 2 seeds)
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from config import (CATEGORICAL_FEATURES, FORBIDDEN_FEATURES, NUMERIC_FEATURES, OUTPUTS_DIR, PROCESSED_DIR,
                    PROJECT_ROOT, SEED, TARGET_COL, TRAIN_FRAC, VAL_FRAC)
from features import build_feature_matrix
from reasons import _CAT_PARENT_LABELS, _NUMERIC_LABELS
from utils import expected_calibration_error

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:                                   # pragma: no cover
    optuna = None
try:
    import catboost
except Exception as _exc:                           # pragma: no cover
    catboost, _CATBOOST_ERR = None, repr(_exc)

DELTA_MIN, N_BOOT, N_SEEDS = 0.005, 1000, 5
N_TRIALS_E1, N_TRIALS_E2 = 60, 100
SHUFFLE_RANGE, MODE_CAP, ECE_CAP = (0.47, 0.53), 0.60, 0.015
PRUNE_SHARE = 0.01
LGB_THREADS = 12
JSON_PATH, MD_PATH = OUTPUTS_DIR / "auc_experiments.json", OUTPUTS_DIR / "auc_experiments.md"
CAT8 = ["Type", "Shipping_Mode", "Market", "Order_Region", "Order_Country", "Customer_Segment", "Department_Name",
        "main_item_category"]
TE_COLS = ["Order_Country", "Order_Region", "Department_Name", "main_item_category"]
TE_K = 20


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ===========================================================================
# Data, baseline, helpers
# ===========================================================================

def load_data():
    orders = pd.read_parquet(PROCESSED_DIR / "orders.parquet").sort_values("order_date").reset_index(drop=True)
    n = len(orders)
    a, b = int(np.floor(n * TRAIN_FRAC)), int(np.floor(n * (TRAIN_FRAC + VAL_FRAC)))
    tr, va, te = orders.iloc[:a].copy(), orders.iloc[a:b].copy(), orders.iloc[b:].copy()
    assert (len(tr), len(va), len(te)) == (46026, 9863, 9863), (len(tr), len(va), len(te))
    return orders, tr, va, te


def assert_clean(columns) -> None:
    bad = sorted(set(columns) & set(FORBIDDEN_FEATURES))
    assert not bad, f"forbidden columns used as features: {bad}"
    assert TARGET_COL not in columns and "aux_target_cancel_fraud" not in columns


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_bootstrap(y, base_scores, var_scores, n_boot=N_BOOT, seed=0):
    """delta = mean over seeds of [AUC(variant) - AUC(baseline)]; the SAME resample indices for both models."""
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(np.mean([fast_auc(y, v) - fast_auc(y, b) for v, b in zip(var_scores, base_scores)]))
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        deltas[i] = np.mean([fast_auc(yb, v[idx]) - fast_auc(yb, b[idx]) for v, b in zip(var_scores, base_scores)])
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _take(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


class ESLGBM(ClassifierMixin, BaseEstimator):
    """LightGBM with early stopping on an INNER chronological fold (the last `inner_frac` of whatever it is given)."""

    def __init__(self, num_leaves=31, max_depth=-1, learning_rate=0.05, n_estimators=1500, min_child_samples=20,
                 colsample_bytree=0.8, subsample=0.8, reg_alpha=0.0, reg_lambda=0.0, min_split_gain=0.0,
                 cat_smooth=10.0, min_data_per_group=100, max_cat_to_onehot=4, inner_frac=0.15, random_state=0,
                 n_jobs=3):
        for k, v in dict(locals()).items():
            if k != "self":
                setattr(self, k, v)

    def _params(self):
        return dict(num_leaves=self.num_leaves, max_depth=self.max_depth, learning_rate=self.learning_rate,
                    min_child_samples=self.min_child_samples, colsample_bytree=self.colsample_bytree,
                    subsample=self.subsample, subsample_freq=1, reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
                    min_split_gain=self.min_split_gain, cat_smooth=self.cat_smooth,
                    min_data_per_group=self.min_data_per_group, max_cat_to_onehot=self.max_cat_to_onehot,
                    random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)

    def fit(self, X, y):
        y = np.asarray(y)
        cut = int(len(y) * (1 - self.inner_frac))
        self.model_ = lgb.LGBMClassifier(n_estimators=self.n_estimators, **self._params())
        self.model_.fit(_take(X, np.arange(cut)), y[:cut], eval_set=[(_take(X, np.arange(cut, len(y))), y[cut:])],
                        eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        self.best_iter_ = int(self.model_.best_iteration_ or self.n_estimators)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def cv_auc(make_model, X, y, n_splits=4):
    """TimeSeriesSplit CV on TRAIN: (mean AUC, mean best_iteration)."""
    aucs, iters = [], []
    for tr_i, te_i in TimeSeriesSplit(n_splits=n_splits).split(np.arange(len(y))):
        m = make_model().fit(_take(X, tr_i), y[tr_i])
        aucs.append(roc_auc_score(y[te_i], m.predict_proba(_take(X, te_i))[:, 1]))
        iters.append(m.best_iter_)
    return float(np.mean(aucs)), float(np.mean(iters))


def optuna_search(X, y, space: Callable, n_trials: int, n_jobs: int = 4, threads: int = 3):
    """TPE search on TRAIN only (TimeSeriesSplit(4) CV, inner chronological early stopping)."""
    def objective(trial):
        params = space(trial)
        auc, it = cv_auc(lambda: ESLGBM(random_state=SEED, n_jobs=threads, **params), X, y)
        trial.set_user_attr("mean_best_iter", it)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    best = study.best_trial
    params = space(optuna.trial.FixedTrial(best.params))
    n_est = max(50, int(best.user_attrs["mean_best_iter"] * 1.15))     # more data in the final refit
    return params, n_est, float(best.value)


# ===========================================================================
# Variant = (featurise, fit_predict)
# ===========================================================================

@dataclass
class Variant:
    key: str
    title: str
    featurize: Callable          # (train_df, [eval_df, ...]) -> (X_train, [X_eval, ...])
    fit_predict: Callable        # (X_train, y_train, [X_eval, ...], seed) -> [raw score arrays]
    columns: list = field(default_factory=list)
    notes: str = ""
    runtime_s: float = 0.0


def onehot_featurize(cat, num, add_cols: Callable | None = None):
    def f(train_df, others):
        dfs = [d if add_cols is None else add_cols(d) for d in [train_df, *others]]
        X_tr, X_others, _, _ = build_feature_matrix(dfs[0], dfs[1:], cat_cols=cat, num_cols=num, save_preprocessor=False)
        return X_tr, X_others
    return f


def lgb_fixed(params: dict, n_estimators: int):
    def fp(X_tr, y_tr, X_evs, seed):
        m = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=LGB_THREADS, verbose=-1, **params)
        m.fit(X_tr, y_tr)
        return [m.predict_proba(X)[:, 1] for X in X_evs]
    return fp


def native_featurize(cat, num):
    def f(train_df, others):
        cats = {c: pd.CategoricalDtype(sorted(train_df[c].astype(str).unique())) for c in cat}

        def conv(d):
            out = d[cat + num].copy()
            for c in cat:
                out[c] = out[c].astype(str).astype(cats[c])
            return out
        return conv(train_df), [conv(d) for d in others]
    return f


# ---- derived (order-time) columns ----------------------------------------

def add_interactions(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    hb = pd.cut(d["order_hour"], [-1, 5, 11, 17, 23], labels=["night", "morning", "afternoon", "evening"]).astype(str)
    d["same_day_after_noon"] = ((d["Shipping_Mode"] == "Same Day") & (d["order_hour"] >= 12)).astype(int)
    d["mode_x_hour_bucket"] = d["Shipping_Mode"].astype(str) + "|" + hb
    d["mode_x_dow"] = d["Shipping_Mode"].astype(str) + "|" + d["order_dow"].astype(int).astype(str)
    od = pd.to_datetime(d["order_date"])
    d["is_month_end"] = (od.dt.day >= od.dt.days_in_month - 2).astype(int)
    d["days_to_weekend"] = (5 - d["order_dow"].astype(int)).clip(lower=0)          # Monday = 0
    return d


E3_CAT = CATEGORICAL_FEATURES + ["mode_x_hour_bucket", "mode_x_dow"]
E3_NUM = NUMERIC_FEATURES + ["same_day_after_noon", "is_month_end", "days_to_weekend"]


def target_encode_columns(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Leak-safe, outcome-gated target encoding (same gating as the history features in 01_prepare.py): for an
    order placed at T only earlier orders whose outcome is ALREADY KNOWN (outcome_known_date < T) are counted,
    smoothed with k=20 towards the late rate of the training period.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare01", SRC_DIR / "01_prepare.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    raw = prep.parse_dates(prep.load_raw())
    rule = prep.check_date_rule(raw)
    ords = prep.add_outcome_known_date(prep.engineer_features(prep.build_order_level(raw)), rule)
    known = ords.set_index("Order Id")["outcome_known_date"]
    out = orders.copy()
    out["_known"] = out["Order Id"].map(known)
    assert out["_known"].notna().all()
    t_ord = out["order_date"].values.astype("datetime64[ns]").astype(np.int64)
    t_known = out["_known"].values.astype("datetime64[ns]").astype(np.int64)
    y = out[TARGET_COL].values.astype(float)
    prior = float(y[: int(np.floor(len(out) * TRAIN_FRAC))].mean())
    for col in TE_COLS:
        te = np.full(len(out), prior)
        for _, idx in out.groupby(col).indices.items():
            order_k = idx[np.argsort(t_known[idx])]
            ks, cum_y = t_known[order_k], np.cumsum(y[order_k])
            pos = np.searchsorted(ks, t_ord[idx], side="left")        # outcomes known strictly before T
            n_past = pos.astype(float)
            s_past = np.where(pos > 0, cum_y[np.maximum(pos - 1, 0)], 0.0)
            te[idx] = (s_past + TE_K * prior) / (n_past + TE_K)
        out[f"te_{col}"] = te
    return out.drop(columns=["_known"])


# ===========================================================================
# Experiments (one function each). Each returns a Variant plus {seed -> val scores} is computed later.
# ===========================================================================

def baseline_variant(fm) -> Variant:
    hp = {k: v for k, v in fm["model"]["hyperparameters"].items() if k != "n_estimators"}
    n_est = int(fm["model"]["hyperparameters"]["n_estimators"])
    hp = dict(hp)
    hp["subsample_freq"] = 0                                        # exactly what the saved model used (sklearn default)
    return Variant("BASE", "Baseline: tuned LightGBM (one-hot)", onehot_featurize(CATEGORICAL_FEATURES, NUMERIC_FEATURES),
                   lgb_fixed(hp, n_est), CATEGORICAL_FEATURES + NUMERIC_FEATURES,
                   notes=f"hyper-parameters read from final_metrics.json: {fm['model']['hyperparameters']}")


def e1_native_categorical(tr, quick) -> Variant:
    cat, num = CAT8, NUMERIC_FEATURES
    X_tr, _ = native_featurize(cat, num)(tr, [])
    y = tr[TARGET_COL].values

    def space(t):
        return dict(num_leaves=t.suggest_int("num_leaves", 7, 63), max_depth=t.suggest_int("max_depth", 3, 8),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 20, 200),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                    subsample=t.suggest_float("subsample", 0.5, 1.0), reg_lambda=t.suggest_float("reg_lambda", 0, 10),
                    cat_smooth=t.suggest_float("cat_smooth", 1, 100, log=True),
                    min_data_per_group=t.suggest_int("min_data_per_group", 20, 300),
                    max_cat_to_onehot=t.suggest_int("max_cat_to_onehot", 2, 16))
    params, n_est, cv = optuna_search(X_tr, y, space, 6 if quick else N_TRIALS_E1)
    p = {k: v for k, v in params.items() if k != "inner_frac"}
    p["subsample_freq"] = 1

    def fp(Xt, yt, Xevs, seed):
        m = lgb.LGBMClassifier(n_estimators=n_est, random_state=seed, n_jobs=LGB_THREADS, verbose=-1, **p)
        m.fit(Xt, yt)
        return [m.predict_proba(X)[:, 1] for X in Xevs]
    return Variant("E1", "E1 native categorical LightGBM (no one-hot)", native_featurize(cat, num), fp, cat + num,
                   notes=f"Optuna {N_TRIALS_E1} trials, train-CV AUC {cv:.4f}; n_estimators {n_est}; params {params}")


def e2_wide_search(tr, quick) -> Variant:
    X_tr, _ = onehot_featurize(CATEGORICAL_FEATURES, NUMERIC_FEATURES)(tr, [])
    y = tr[TARGET_COL].values

    def space(t):
        return dict(num_leaves=t.suggest_int("num_leaves", 7, 63), max_depth=t.suggest_int("max_depth", -1, 8),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 20, 200),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
                    subsample=t.suggest_float("subsample", 0.5, 1.0), reg_alpha=t.suggest_float("reg_alpha", 0, 10),
                    reg_lambda=t.suggest_float("reg_lambda", 0, 10), min_split_gain=t.suggest_float("min_split_gain", 0, 0.5))
    params, n_est, cv = optuna_search(X_tr, y, space, 6 if quick else N_TRIALS_E2)
    p = {k: v for k, v in params.items() if k not in ("inner_frac", "cat_smooth", "min_data_per_group", "max_cat_to_onehot")}
    p["subsample_freq"] = 1
    return Variant("E2", "E2 wider Optuna search (one-hot)", onehot_featurize(CATEGORICAL_FEATURES, NUMERIC_FEATURES),
                   lgb_fixed(p, n_est), CATEGORICAL_FEATURES + NUMERIC_FEATURES,
                   notes=f"Optuna {N_TRIALS_E2} trials (TPE), train-CV AUC {cv:.4f}; n_estimators {n_est}; params {params}")


def e3_interactions(base_fp) -> Variant:
    return Variant("E3", "E3 order-time interaction features", onehot_featurize(E3_CAT, E3_NUM, add_interactions), base_fp,
                   E3_CAT + E3_NUM, notes="baseline hyper-parameters; +same_day_after_noon, mode x hour bucket, mode x day-of-week, "
                                          "is_month_end, days_to_weekend")


def e4_target_encoding(base_fp, with_te: bool, replace: bool) -> Variant:
    te = [f"te_{c}" for c in TE_COLS]
    if replace:
        cat, num, key, title = [c for c in CATEGORICAL_FEATURES if c not in TE_COLS], NUMERIC_FEATURES + te, "E4b", \
            "E4b target encoding REPLACING one-hot of those 4 columns"
    else:
        cat, num, key, title = CATEGORICAL_FEATURES, NUMERIC_FEATURES + te, "E4a", "E4a target encoding ADDED to one-hot"
    return Variant(key, title, onehot_featurize(cat, num), base_fp, cat + num,
                   notes="outcome-gated (outcome_known_date < order time), smoothing k=20, prior = training-period late rate")


def e5_catboost(quick) -> Variant | None:
    if catboost is None:
        log(f"E5 skipped: catboost not importable ({_CATBOOST_ERR})")
        return None
    cat, num = CAT8, NUMERIC_FEATURES

    def feat(train_df, others):
        return train_df[cat + num].astype({c: str for c in cat}), [d[cat + num].astype({c: str for c in cat}) for d in others]

    def fit(Xt, yt, Xe, seed, depth, lr, iters, es):
        m = catboost.CatBoostClassifier(iterations=iters, depth=depth, learning_rate=lr, random_seed=seed, verbose=0,
                                        thread_count=LGB_THREADS, cat_features=cat, eval_metric="AUC",
                                        allow_writing_files=False)
        if es:
            m.fit(Xt, yt, eval_set=Xe, early_stopping_rounds=50)
        else:
            m.fit(Xt, yt)
        return m

    def fp_factory(depth, lr, iters):
        return lambda Xt, yt, Xevs, seed: [fit(Xt, yt, None, seed, depth, lr, iters, False).predict_proba(X)[:, 1] for X in Xevs]

    def select(tr):
        """Pick depth/lr/iterations on an INNER chronological split of TRAIN (last 15%)."""
        X, _ = feat(tr, [])
        y = tr[TARGET_COL].values
        cut = int(len(y) * 0.85)
        best = (-1, None)
        for depth, lr in itertools.product([4, 6, 8], [0.03, 0.06]):
            m = fit(X.iloc[:cut], y[:cut], (X.iloc[cut:], y[cut:]), 0, depth, lr, 300 if quick else 1500, True)
            auc = roc_auc_score(y[cut:], m.predict_proba(X.iloc[cut:])[:, 1])
            if auc > best[0]:
                best = (auc, (depth, lr, max(50, int(m.get_best_iteration() * 1.15))))
        return best

    return Variant("E5", "E5 CatBoost (native categorical)", feat, None, cat + num, notes="_select_", runtime_s=0.0), (select, fp_factory)


def e7_pruned(fm, base_fp) -> Variant:
    thresholds = {_CAT_PARENT_LABELS.get(c, c): c for c in CATEGORICAL_FEATURES}
    thresholds.update({lbl: c for c, lbl in _NUMERIC_LABELS.items()})
    keep_labels = [g["factor"] for g in fm["shap"]["grouped"] if g["share"] >= PRUNE_SHARE]
    kept = {thresholds[l] for l in keep_labels}
    cat = [c for c in CATEGORICAL_FEATURES if c in kept]
    num = [c for c in NUMERIC_FEATURES if c in kept]
    dropped = [c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c not in kept]
    return Variant("E7", f"E7 pruned to {len(cat) + len(num)} features (grouped SHAP share >= {PRUNE_SHARE:.0%})",
                   onehot_featurize(cat, num), base_fp, cat + num,
                   notes=f"kept {cat + num}; dropped {len(dropped)}: {dropped}")


# ===========================================================================
# E6 blend
# ===========================================================================

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_blend_weights(y, comps, step=0.05):
    """Sum-to-one grid search of the logit-average weights maximising ROC-AUC on y."""
    grid = np.arange(0, 1 + 1e-9, step)
    best = (-1, None)
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < -1e-9:
                continue
            w = np.array([w1, w2, max(w3, 0)])
            auc = fast_auc(y, sum(wi * c for wi, c in zip(w, comps)))
            if auc > best[0]:
                best = (auc, w)
    return best[1]


def e6_components(base_fp, fm):
    """Per-seed component raw scores on VALIDATION: LightGBM (baseline), XGBoost, logistic regression."""
    def fp_xgb(Xt, yt, Xevs, seed):
        m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.6,
                              min_child_weight=10, reg_lambda=1.0, random_state=seed, n_jobs=LGB_THREADS, verbosity=0)
        m.fit(Xt, yt)
        return [m.predict_proba(X)[:, 1] for X in Xevs]

    def fp_lr(Xt, yt, Xevs, seed):
        m = LogisticRegression(max_iter=2000, C=0.5)
        m.fit(Xt, yt)
        return [m.predict_proba(X)[:, 1] for X in Xevs]
    return [base_fp, fp_xgb, fp_lr]


# ===========================================================================
# Evaluation machinery
# ===========================================================================

def score_variant(v: Variant, tr, va, seeds):
    """Fit on TRAIN, return one RAW validation score vector per seed (+ the featurised matrices for reuse)."""
    assert_clean(v.columns)
    Xt, (Xv,) = v.featurize(tr, [va])
    y = tr[TARGET_COL].values
    return [v.fit_predict(Xt, y, [Xv], s)[0] for s in seeds], (Xt, Xv)


def per_mode_auc(y, s, modes):
    out = {}
    for m in ("Standard Class", "Second Class"):
        k = modes == m
        out[m] = fast_auc(y[k], s[k])
    return out


def backtest_mean(v: Variant, orders, n_folds=5):
    """06_robustness.py fold logic: train=[0,f), val(skipped here)=next 5%, test = following 10%; RAW ROC-AUC mean."""
    n, aucs = len(orders), []
    for tf in np.linspace(0.40, 0.80, n_folds):
        a, b, c = int(np.floor(n * tf)), int(np.floor(n * (tf + 0.05))), min(int(np.floor(n * (tf + 0.15))), n)
        ftr, fte = orders.iloc[:a], orders.iloc[b:c]
        Xt, (Xe,) = v.featurize(ftr, [fte])
        aucs.append(roc_auc_score(fte[TARGET_COL].values, v.fit_predict(Xt, ftr[TARGET_COL].values, [Xe], SEED)[0]))
    return float(np.mean(aucs)), [float(a) for a in aucs]


def label_shuffle_val(v: Variant, tr, va):
    Xt, (Xv,) = v.featurize(tr, [va])
    ys = np.random.default_rng(42).permutation(tr[TARGET_COL].values)
    return fast_auc(va[TARGET_COL].values, v.fit_predict(Xt, ys, [Xv], SEED)[0])


def run_pytest(files):
    p = subprocess.run([sys.executable, "-m", "pytest", *files, "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONUTF8": "1"})
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1:] or [p.stderr[-200:]]
    return p.returncode == 0, tail[0]


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    quick = args.quick
    seeds = [0, 1] if quick else list(range(N_SEEDS))
    t_all = time.time()

    fm = json.loads((OUTPUTS_DIR / "final_metrics.json").read_text(encoding="utf-8"))
    base_val_auc_json = float(fm["validation"]["roc_auc"])             # never hard-coded
    base_bt_json = float(fm["backtest"]["roc_auc_mean"])
    orders, tr, va, te = load_data()
    y_va = va[TARGET_COL].values
    modes_va = va["Shipping_Mode"].values
    log(f"train/val/test = {len(tr):,}/{len(va):,}/{len(te):,};  baseline val ROC-AUC (final_metrics.json) = {base_val_auc_json:.4f}")

    # --- baseline (5 seeds, same hyper-parameters) ---
    base = baseline_variant(fm)
    base_scores, _ = score_variant(base, tr, va, seeds)
    base_aucs = [fast_auc(y_va, s) for s in base_scores]
    log(f"baseline refit: 5-seed val AUC {np.mean(base_aucs):.4f} +/- {np.std(base_aucs, ddof=1):.4f} (saved model {base_val_auc_json:.4f})")

    # --- target-encoded table (needs the raw CSV for outcome_known_date) ---
    te_ready = True
    try:
        orders_te = target_encode_columns(orders)
        n_tr = len(tr)
        tr_te, va_te, te_te = orders_te.iloc[:n_tr].copy(), orders_te.iloc[n_tr:n_tr + len(va)].copy(), orders_te.iloc[n_tr + len(va):].copy()
    except Exception as exc:                                              # pragma: no cover
        te_ready = False
        log(f"E4 skipped: cannot rebuild outcome_known_date ({exc!r})")

    results, variants = [], {}

    def evaluate(v: Variant, tr_df, va_df, scores, runtime, extra=None):
        aucs = [fast_auc(y_va, s) for s in scores]
        delta, lo, hi = paired_bootstrap(y_va, base_scores, scores)
        std = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        a = bool(delta >= DELTA_MIN and lo > 0)
        b = bool(std < delta)
        r = dict(key=v.key, title=v.title, n_features=len(v.columns), val_auc_mean=float(np.mean(aucs)), val_auc_std=std,
                 delta=delta, delta_ci=[lo, hi], runtime_s=round(runtime, 1), rule_a=a, rule_b=b, notes=v.notes,
                 per_mode_val=per_mode_auc(y_va, np.mean(scores, axis=0), modes_va))
        if extra:
            r.update(extra)
        results.append(r)
        variants[v.key] = (v, tr_df, va_df)
        log(f"{v.key}: val AUC {r['val_auc_mean']:.4f}+/-{std:.4f}  delta {delta:+.4f} CI [{lo:+.4f},{hi:+.4f}]  a={a} b={b}  ({runtime:.0f}s)")

    base_fp = base.fit_predict

    def timed(fn):
        t0 = time.time()
        out = fn()
        return out, time.time() - t0

    # E1
    try:
        v, rt = timed(lambda: e1_native_categorical(tr, quick))
        sc, rt2 = timed(lambda: score_variant(v, tr, va, seeds)[0])
        evaluate(v, tr, va, sc, rt + rt2)
    except Exception as exc:
        log(f"E1 failed: {exc!r}")
        results.append(dict(key="E1", title="E1 native categorical LightGBM", status=f"error: {exc!r}"))

    # E2
    try:
        v, rt = timed(lambda: e2_wide_search(tr, quick))
        sc, rt2 = timed(lambda: score_variant(v, tr, va, seeds)[0])
        evaluate(v, tr, va, sc, rt + rt2)
    except Exception as exc:
        log(f"E2 failed: {exc!r}")
        results.append(dict(key="E2", title="E2 wider search", status=f"error: {exc!r}"))

    # E3
    try:
        v = e3_interactions(base_fp)
        sc, rt = timed(lambda: score_variant(v, tr, va, seeds)[0])
        evaluate(v, tr, va, sc, rt)
    except Exception as exc:
        log(f"E3 failed: {exc!r}")
        results.append(dict(key="E3", title="E3 interactions", status=f"error: {exc!r}"))

    # E4a / E4b
    if te_ready:
        for replace in (False, True):
            try:
                v = e4_target_encoding(base_fp, True, replace)
                sc, rt = timed(lambda: score_variant(v, tr_te, va_te, seeds)[0])
                evaluate(v, tr_te, va_te, sc, rt)
            except Exception as exc:
                log(f"E4 failed: {exc!r}")
                results.append(dict(key="E4b" if replace else "E4a", title="E4 target encoding", status=f"error: {exc!r}"))
    else:
        results.append(dict(key="E4", title="E4 target encoding", status="skipped: raw CSV needed for outcome_known_date"))

    # E5
    try:
        out = e5_catboost(quick)
        if out is None:
            results.append(dict(key="E5", title="E5 CatBoost", status=f"skipped: catboost not importable ({_CATBOOST_ERR})"))
        else:
            v, (select, fp_factory) = out
            t0 = time.time()
            inner_auc, (depth, lr, iters) = select(tr)
            v.fit_predict = fp_factory(depth, lr, iters)
            v.notes = f"depth {depth}, lr {lr}, {iters} iterations (chosen on an inner chronological 15% of TRAIN, inner AUC {inner_auc:.4f})"
            sc = score_variant(v, tr, va, seeds)[0]
            evaluate(v, tr, va, sc, time.time() - t0)
    except Exception as exc:
        log(f"E5 failed: {exc!r}")
        results.append(dict(key="E5", title="E5 CatBoost", status=f"error: {exc!r}"))

    # E6 blend (weights fitted on VALIDATION; judged with out-of-fold weights to avoid selection bias)
    try:
        t0 = time.time()
        comps = e6_components(base_fp, fm)
        Xt, (Xv,) = onehot_featurize(CATEGORICAL_FEATURES, NUMERIC_FEATURES)(tr, [va])
        y_tr = tr[TARGET_COL].values
        half = len(y_va) // 2
        blend_scores, insample_scores, weights = [], [], []
        for s in seeds:
            cs = [logit(f(Xt, y_tr, [Xv], s)[0]) for f in comps]
            w_full = fit_blend_weights(y_va, cs)
            w1, w2 = fit_blend_weights(y_va[:half], [c[:half] for c in cs]), fit_blend_weights(y_va[half:], [c[half:] for c in cs])
            ordered = np.empty(len(y_va))
            ordered[half:] = sum(w * c[half:] for w, c in zip(w1, cs))          # weights fitted on the FIRST half scored on the second
            ordered[:half] = sum(w * c[:half] for w, c in zip(w2, cs))          # and vice versa
            blend_scores.append(ordered)
            insample_scores.append(sum(w * c for w, c in zip(w_full, cs)))
            weights.append([float(x) for x in w_full])
        v = Variant("E6", "E6 logit-blend LightGBM + XGBoost + logistic regression", None, None, CATEGORICAL_FEATURES + NUMERIC_FEATURES,
                    notes=f"weights (LGBM, XGB, LR) fitted on validation: {np.round(np.mean(weights, axis=0), 2).tolist()}; "
                          f"judged on cross-fitted halves (in-sample blend AUC {np.mean([fast_auc(y_va, s) for s in insample_scores]):.4f} is optimistic)")
        evaluate(v, tr, va, blend_scores, time.time() - t0, extra=dict(blend_weights=weights))
    except Exception as exc:
        log(f"E6 failed: {exc!r}")
        results.append(dict(key="E6", title="E6 blend", status=f"error: {exc!r}"))

    # E7
    try:
        v = e7_pruned(fm, base_fp)
        sc, rt = timed(lambda: score_variant(v, tr, va, seeds)[0])
        evaluate(v, tr, va, sc, rt)
    except Exception as exc:
        log(f"E7 failed: {exc!r}")
        results.append(dict(key="E7", title="E7 pruning", status=f"error: {exc!r}"))

    # --- further rules only for candidates that already pass a and b ---
    candidates = [r for r in results if r.get("rule_a") and r.get("rule_b")]
    base_bt, base_bt_folds = backtest_mean(base, orders) if candidates else (None, None)
    for r in results:
        if r.get("rule_a") is None:
            continue
        r["rule_c"] = r["rule_d"] = r["rule_e"] = r["rule_f"] = None
        r["promotable"] = False
        if not (r["rule_a"] and r["rule_b"]):
            continue
        v, tr_df, va_df = variants[r["key"]]
        r["rule_e"] = bool(all(x < MODE_CAP for x in r["per_mode_val"].values()))
        if v.featurize is None:                                              # E6: shuffle/backtest of the blend are not defined
            r["rule_c"] = r["rule_d"] = None
            r["promotable"] = False
            r["note_blend"] = "blend: rules c/d need a per-component refit; judged not promotable without them"
            continue
        sh = label_shuffle_val(v, tr_df, va_df)
        r["label_shuffle_val_auc"] = sh
        r["rule_c"] = bool(SHUFFLE_RANGE[0] <= sh <= SHUFFLE_RANGE[1])
        bt, folds = backtest_mean(v, orders if v.key not in ("E4a", "E4b") else orders_te)
        r["backtest_mean"], r["backtest_folds"] = bt, folds
        r["rule_d"] = bool(bt >= base_bt and bt >= base_bt_json)
        r["promotable"] = bool(r["rule_c"] and r["rule_d"] and r["rule_e"])

    # --- rule f: touch the TEST set once, only for the single best variant that passed a-e ---
    chosen = None
    ok = sorted([r for r in results if r.get("promotable")], key=lambda r: -r["delta"])
    if ok:
        r = ok[0]
        v, tr_df, va_df = variants[r["key"]]
        te_df = te if r["key"] not in ("E4a", "E4b") else te_te
        Xt, (Xv, Xe) = v.featurize(tr_df, [va_df, te_df])
        sv, se = v.fit_predict(Xt, tr_df[TARGET_COL].values, [Xv, Xe], SEED)
        ir = IsotonicRegression(out_of_bounds="clip").fit(sv, y_va)
        ece = expected_calibration_error(te_df[TARGET_COL].values, ir.predict(se))
        r["test_ece"] = float(ece)
        r["test_roc_auc"] = float(roc_auc_score(te_df[TARGET_COL].values, se))
        r["rule_f"] = bool(ece <= ECE_CAP)
        r["promoted"] = r["rule_f"]
        chosen = r["key"] if r["rule_f"] else None

    leak_ok, leak_line = run_pytest(["tests/test_no_leakage.py", "tests/test_pipeline.py"])
    log(f"leakage tests: {'PASS' if leak_ok else 'FAIL'}  ({leak_line})")
    if chosen and not leak_ok:
        for r in results:
            if r.get("key") == chosen:
                r["promoted"] = False
        chosen = None

    payload = dict(generated=datetime.now().isoformat(timespec="seconds"), quick=quick, seeds=seeds,
                   baseline=dict(val_auc_final_metrics=base_val_auc_json, val_auc_refit_mean=float(np.mean(base_aucs)),
                                 val_auc_refit_std=float(np.std(base_aucs, ddof=1)), backtest_final_metrics=base_bt_json,
                                 backtest_refit=base_bt),
                   acceptance=dict(delta_min=DELTA_MIN, shuffle_range=list(SHUFFLE_RANGE), mode_cap=MODE_CAP, ece_cap=ECE_CAP),
                   results=results, leakage_tests=dict(passed=leak_ok, summary=leak_line), promoted=chosen,
                   runtime_s=round(time.time() - t_all, 1))
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    MD_PATH.write_text(render_md(payload), encoding="utf-8")
    log(f"wrote {JSON_PATH.name} and {MD_PATH.name}; promoted = {chosen}; total {time.time() - t_all:.0f}s")


# ===========================================================================
# Markdown
# ===========================================================================

def yn(x):
    return "n/a" if x is None else ("pass" if x else "FAIL")


def render_md(p: dict) -> str:
    b = p["baseline"]
    L = ["# AUC experiments: can any leakage-free change raise ROC-AUC?", "",
         f"_Generated by `src/12_auc_experiments.py` on {p['generated']}"
         f"{' (QUICK smoke run: not a valid result)' if p['quick'] else ''}. Read-only: no model, calibrator, preprocessor or "
         "threshold file was modified._", "",
         "## Protocol", "",
         "- Target `Late_delivery_risk`; chronological split 46,026 / 9,863 / 9,863. Features only from `CATEGORICAL_FEATURES` / "
         "`NUMERIC_FEATURES`, features derived from them, or earlier-known history (all checked against `FORBIDDEN_FEATURES`).",
         "- Tuning and selection on TRAIN (`TimeSeriesSplit`); variants compared on VALIDATION with RAW scores. The test set is touched "
         "once, only for the single best variant that already passed rules a-e; losing variants have no test score.",
         f"- Every variant: {len(p['seeds'])} seeds (mean, std) and a paired bootstrap ({N_BOOT} resamples, identical indices for variant and baseline).",
         f"- Baseline validation ROC-AUC read from `final_metrics.json`: **{b['val_auc_final_metrics']:.4f}**. "
         f"Refit with the same hyper-parameters over the same seeds: {b['val_auc_refit_mean']:.4f} +/- {b['val_auc_refit_std']:.4f} "
         "(this refit is the paired reference for the deltas below).", "",
         "## Acceptance rules (all must hold)", "",
         f"a. validation delta >= +{DELTA_MIN} and the paired 95% CI excludes 0; b. 5-seed std < delta; c. label-shuffle AUC in "
         f"[{SHUFFLE_RANGE[0]}, {SHUFFLE_RANGE[1]}] and leakage tests pass; d. 5-fold rolling-backtest mean ROC-AUC >= baseline; "
         f"e. Standard Class and Second Class within-mode AUC < {MODE_CAP}; f. test ECE <= {ECE_CAP} after re-fitting isotonic on validation.", "",
         "## Results (validation)", "",
         "| Variant | Features | Val ROC-AUC (mean +/- std) | Delta vs baseline | 95% CI (paired) | Runtime | a | b | c | d | e | f |",
         "|---|---:|---|---:|---|---:|---|---|---|---|---|---|"]
    for r in p["results"]:
        if "val_auc_mean" not in r:
            L.append(f"| {r['key']} {r['title']} | - | {r['status']} | - | - | - | - | - | - | - | - | - |")
            continue
        L.append(f"| {r['title']} | {r['n_features']} | {r['val_auc_mean']:.4f} +/- {r['val_auc_std']:.4f} | {r['delta']:+.4f} | "
                 f"[{r['delta_ci'][0]:+.4f}, {r['delta_ci'][1]:+.4f}] | {r['runtime_s']:.0f}s | {yn(r['rule_a'])} | {yn(r['rule_b'])} | "
                 f"{yn(r.get('rule_c'))} | {yn(r.get('rule_d'))} | {yn(r.get('rule_e'))} | {yn(r.get('rule_f'))} |")
    L += ["", "Rules c-f are evaluated only for variants that pass a and b (`n/a` = not needed because an earlier rule already failed).", "",
          "## Details", ""]
    for r in p["results"]:
        if "val_auc_mean" in r:
            pm = r["per_mode_val"]
            L.append(f"- **{r['key']}**: {r['notes']}. Within-mode validation AUC: Standard Class {pm['Standard Class']:.3f}, "
                     f"Second Class {pm['Second Class']:.3f}.")
    L += ["", "## Leakage tests", "",
          f"`tests/test_no_leakage.py` + `tests/test_pipeline.py`: **{'PASS' if p['leakage_tests']['passed'] else 'FAIL'}** "
          f"({p['leakage_tests']['summary']}).", "", "## Verdict", ""]
    if p["promoted"]:
        r = next(x for x in p["results"] if x.get("key") == p["promoted"])
        L.append(f"**Promoted: {r['title']}** (validation delta {r['delta']:+.4f}, CI [{r['delta_ci'][0]:+.4f}, {r['delta_ci'][1]:+.4f}]; "
                 f"test ECE {r['test_ece']:.4f}).")
    else:
        touched = [x for x in p["results"] if x.get("test_ece") is not None]
        L.append("No leakage-free variant beat the tuned LightGBM; the model is at the information ceiling of checkout-time features.")
        if touched:
            x = touched[0]
            L.append(f"\n(The single test-set check was spent on {x['title']}: it passed a-e but test ECE {x['test_ece']:.4f} > {ECE_CAP}.)")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
