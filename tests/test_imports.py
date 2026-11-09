"""
tests/test_imports.py
=====================
Quick smoke-test: verify that all project modules import without errors
and that key symbols are present and callable.

Run with:
    python tests/test_imports.py
or:
    pytest tests/test_imports.py -v
"""

import sys
import types
from pathlib import Path

# Make src/ importable when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PASS = "✅"
FAIL = "❌"
results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    results.append((label, condition, detail))
    status = PASS if condition else FAIL
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# 1. config.py
# ---------------------------------------------------------------------------
print("\n── config.py ──────────────────────────────────────────────────")
try:
    import config
    check("import config", True)
    check("config.SEED == 42",             config.SEED == 42, str(config.SEED))
    check("config.TRAIN_FRAC == 0.70",     abs(config.TRAIN_FRAC - 0.70) < 1e-9)
    check("config.VAL_FRAC == 0.15",       abs(config.VAL_FRAC - 0.15) < 1e-9)
    check("config.TEST_FRAC == 0.15",      abs(config.TEST_FRAC - 0.15) < 1e-9)
    check("fractions sum to 1.0",          abs(config.TRAIN_FRAC + config.VAL_FRAC + config.TEST_FRAC - 1.0) < 1e-9)
    check("FORBIDDEN_FEATURES is list",    isinstance(config.FORBIDDEN_FEATURES, list))
    check("FORBIDDEN_FEATURES non-empty",  len(config.FORBIDDEN_FEATURES) > 0,
          f"{len(config.FORBIDDEN_FEATURES)} entries")
    check("CATEGORICAL_FEATURES is list",  isinstance(config.CATEGORICAL_FEATURES, list))
    check("NUMERIC_FEATURES is list",      isinstance(config.NUMERIC_FEATURES, list))
    check("RAW_CSV is Path",               isinstance(config.RAW_CSV, Path))
    check("TARGET_COL defined",            hasattr(config, "TARGET_COL"))
    check("DATE_COL defined",              hasattr(config, "DATE_COL"))
except Exception as exc:
    check("import config", False, str(exc))

# ---------------------------------------------------------------------------
# 2. utils.py
# ---------------------------------------------------------------------------
print("\n── utils.py ───────────────────────────────────────────────────")
try:
    import utils
    check("import utils", True)
    check("load_raw is callable",                    callable(utils.load_raw))
    check("chrono_split is callable",                callable(utils.chrono_split))
    check("compute_metrics is callable",             callable(utils.compute_metrics))
    check("precision_at_k is callable",              callable(utils.precision_at_k))
    check("expected_calibration_error is callable",  callable(utils.expected_calibration_error))
    check("bootstrap_ci is callable",                callable(utils.bootstrap_ci))
    check("save_table is callable",                  callable(utils.save_table))

    # Functional smoke-test for metrics (synthetic-free: use a trivial array)
    import numpy as np
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = utils.compute_metrics(y, p, threshold=0.5)
    check("compute_metrics returns dict",    isinstance(metrics, dict))
    check("compute_metrics has roc_auc",     "roc_auc" in metrics)
    check("compute_metrics has pr_auc",      "pr_auc" in metrics)
    check("compute_metrics has brier",       "brier" in metrics)
    check("compute_metrics has prec@top10%", "precision_at_top10pct" in metrics)
    check("compute_metrics has ece",         "ece" in metrics)
    check("roc_auc == 1.0 (perfect)",        metrics["roc_auc"] == 1.0,
          str(metrics["roc_auc"]))

    p_at_k = utils.precision_at_k(y, p, k=0.5)
    check("precision_at_k returns float",    isinstance(p_at_k, float))

    ece = utils.expected_calibration_error(y, p)
    check("ECE returns float",               isinstance(ece, float))

    # Use roc_auc_score from sklearn directly as the metric callable
    from sklearn.metrics import roc_auc_score as _roc
    # Larger array ensures bootstrap draws have both classes present
    rng_b = np.random.default_rng(42)
    y_b = np.array([0]*30 + [1]*30)
    p_b = np.concatenate([rng_b.uniform(0, 0.4, 30), rng_b.uniform(0.6, 1.0, 30)])
    lo, hi = utils.bootstrap_ci(y_b, p_b, _roc, n=200)
    check("bootstrap_ci returns (lo, hi)",   lo <= hi, f"[{lo:.3f}, {hi:.3f}]")

except Exception as exc:
    check("import utils", False, str(exc))
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# 3. features.py
# ---------------------------------------------------------------------------
print("\n── features.py ────────────────────────────────────────────────")
try:
    import features
    check("import features", True)
    check("build_feature_matrix is callable", callable(features.build_feature_matrix))
    check("load_preprocessor is callable",    callable(features.load_preprocessor))
except Exception as exc:
    check("import features", False, str(exc))
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{'=' * 60}")
print(f"  {PASS} {n_pass} passed   {FAIL} {n_fail} failed   "
      f"(total {n_pass + n_fail} checks)")
print(f"{'=' * 60}\n")

if n_fail:
    sys.exit(1)
