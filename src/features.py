"""
src/features.py
===============
Feature engineering and preprocessing for RTO Shield AI - Model 1.

The central function build_feature_matrix() fits all encoders/scalers/imputers
on the training split ONLY and applies them to validation and test splits.
Nothing is fitted on val/test data — ever.

Functions
---------
build_feature_matrix(train_df, other_dfs, cat_cols, num_cols)
    Fit a ColumnTransformer on train_df; transform all supplied DataFrames.
    Returns X_train, list[X_other], feature_names, fitted_preprocessor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose   import ColumnTransformer
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer

import joblib

from config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    FORBIDDEN_FEATURES,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_forbidden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any FORBIDDEN_FEATURES columns that are present in *df*.
    Also drops target so the returned frame contains only candidate features.
    """
    cols_to_drop = [c for c in FORBIDDEN_FEATURES + [TARGET_COL] if c in df.columns]
    return df.drop(columns=cols_to_drop)


def _select_feature_cols(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
) -> pd.DataFrame:
    """
    Keep only the columns listed in cat_cols and num_cols that actually exist
    in *df*.  Missing columns raise a ValueError so the caller knows
    immediately if config drifts from the data.
    """
    all_expected = cat_cols + num_cols
    missing = [c for c in all_expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"[features] The following feature columns are missing from the "
            f"DataFrame: {missing}"
        )
    return df[all_expected].copy()


def _build_preprocessor(
    cat_cols: list[str],
    num_cols: list[str],
) -> ColumnTransformer:
    """
    Construct the sklearn ColumnTransformer that:

    * Imputes + OneHotEncodes every categorical feature
        – missing categories filled with literal string "missing"
        – handle_unknown="ignore" → unseen categories produce all-zero rows
        – drop="if_binary" → avoids dummy-variable trap for binary cols
    * Imputes + StandardScales every numeric feature
        – customer_prev_cash_share:  NaN → 0 (no cash history means 0)
        – days_since_prev_order:     NaN → median (first-ever order has no lag)
        – all other numeric NaNs:    filled with median as a safe default

    The transformer is returned UN-fitted; call .fit_transform(X_train) to fit.
    """
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="if_binary",
        )),
    ])

    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("num", num_pipeline, num_cols),
        ],
        remainder="drop",        # silently discard any other columns
        verbose_feature_names_out=True,
    )
    return preprocessor


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract human-readable feature names from a *fitted* ColumnTransformer.
    Returns a flat list ordered as the transformer produces columns.
    """
    return list(preprocessor.get_feature_names_out())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    train_df: pd.DataFrame,
    other_dfs: list[pd.DataFrame],
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    save_preprocessor: bool = True,
    preprocessor_path: Path | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[str], ColumnTransformer]:
    """
    Fit a full preprocessing pipeline on *train_df* and transform every
    DataFrame in *other_dfs* using the same fitted pipeline.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split (encoders / scalers are fitted exclusively on this).
    other_dfs : list[pd.DataFrame]
        Validation and/or test splits to transform (not fitted on).
    cat_cols : list[str] | None
        Categorical feature columns to use. Defaults to CATEGORICAL_FEATURES.
    num_cols : list[str] | None
        Numeric feature columns to use. Defaults to NUMERIC_FEATURES.
    save_preprocessor : bool
        If True, persist the fitted ColumnTransformer to models/ as a joblib
        file so it can be reloaded for inference without retraining.
    preprocessor_path : Path, optional
        Custom save path for the preprocessor; defaults to
        models/preprocessor.joblib.

    Returns
    -------
    X_train : np.ndarray  shape (n_train, n_features)
        Encoded + scaled training features.
    X_others : list[np.ndarray]
        One array per DataFrame in *other_dfs*, same column order as X_train.
    feature_names : list[str]
        Ordered feature name strings matching columns of X_train / X_others.
    preprocessor : ColumnTransformer
        The *fitted* ColumnTransformer; store alongside models for inference.

    Example
    -------
    >>> X_train, (X_val, X_test), names, prep = build_feature_matrix(
    ...     train_df, [val_df, test_df]
    ... )
    >>> y_train = train_df[TARGET_COL].values
    """
    cat_cols = cat_cols if cat_cols is not None else CATEGORICAL_FEATURES
    num_cols = num_cols if num_cols is not None else NUMERIC_FEATURES

    # ── 1. Isolate feature columns (no target, no forbidden cols) ────────────
    X_train_raw  = _select_feature_cols(train_df,  cat_cols, num_cols)
    X_others_raw = [_select_feature_cols(df, cat_cols, num_cols) for df in other_dfs]

    # ── 2. Build & fit preprocessor on training data only ────────────────────
    preprocessor = _build_preprocessor(cat_cols, num_cols)
    X_train = preprocessor.fit_transform(X_train_raw)

    print(
        f"[build_feature_matrix] Fitted preprocessor on {len(X_train_raw):,} "
        f"training rows → {X_train.shape[1]} features"
    )

    # ── 3. Transform other splits (val, test, …) with fitted preprocessor ────
    X_others: list[np.ndarray] = []
    for i, X_raw in enumerate(X_others_raw):
        X_t = preprocessor.transform(X_raw)
        X_others.append(X_t)
        print(
            f"[build_feature_matrix] Transformed other_dfs[{i}]: "
            f"{X_t.shape[0]:,} rows × {X_t.shape[1]} features"
        )

    # ── 4. Extract feature names ──────────────────────────────────────────────
    feature_names = _get_feature_names(preprocessor)

    # ── 5. Persist preprocessor artefact ─────────────────────────────────────
    if save_preprocessor:
        save_path = preprocessor_path or (MODELS_DIR / "preprocessor.joblib")
        joblib.dump(preprocessor, save_path)
        print(f"[build_feature_matrix] Preprocessor saved → {save_path}")

    return X_train, X_others, feature_names, preprocessor


def load_preprocessor(path: Path | None = None) -> ColumnTransformer:
    """
    Reload a previously saved preprocessor from disk.

    Parameters
    ----------
    path : Path, optional
        Defaults to models/preprocessor.joblib.

    Returns
    -------
    ColumnTransformer (fitted).
    """
    load_path = path or (MODELS_DIR / "preprocessor.joblib")
    preprocessor = joblib.load(load_path)
    print(f"[load_preprocessor] Loaded preprocessor from {load_path}")
    return preprocessor
