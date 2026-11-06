"""
preprocessing.py
----------------
Feature engineering and Scikit-learn pipeline construction for
crop and fertilizer models.

Principles
----------
* All transformers are fitted only on training data.
* Pipelines are self-contained — preprocessing and model in one object.
* Categorical columns use OrdinalEncoder with handle_unknown='use_encoded_value'
  so unseen categories at inference time do not crash the pipeline.
* Numeric columns are optionally scaled (required for KNN, optional for trees).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# Crop dataset features
# ---------------------------------------------------------------------------

CROP_FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROP_TARGET_COL = "label"

# ---------------------------------------------------------------------------
# Fertilizer dataset — dynamically determined, but these are the expected ones
# ---------------------------------------------------------------------------

FERT_NUMERIC_COLS = ["Temperature", "Humidity", "Moisture", "N", "K", "P"]
FERT_CATEGORICAL_COLS = ["Soil Type", "Crop Type"]
FERT_TARGET_COL = "Fertilizer Name"


# ---------------------------------------------------------------------------
# Helper: detect actual numeric / categorical columns in a dataframe
# ---------------------------------------------------------------------------

def split_feature_types(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str]]:
    """
    Partition *feature_cols* into (numeric_cols, categorical_cols) based
    on the actual dtypes found in *df*.
    """
    numeric, categorical = [], []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = False,
    scaler: str = "standard",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Optionally scales numeric columns (StandardScaler or MinMaxScaler).
    - OrdinalEncodes categorical columns, handling unseen values gracefully.

    Parameters
    ----------
    numeric_cols : list[str]
    categorical_cols : list[str]
    scale_numeric : bool
        When True, apply *scaler* to numeric columns.
    scaler : str
        'standard' or 'minmax'.
    """
    transformers = []

    if numeric_cols:
        if scale_numeric:
            num_transformer = (
                StandardScaler() if scaler == "standard" else MinMaxScaler()
            )
        else:
            num_transformer = "passthrough"
        transformers.append(("num", num_transformer, numeric_cols))

    if categorical_cols:
        cat_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        transformers.append(("cat", cat_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_crop_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Return a preprocessor for the crop dataset."""
    return build_preprocessor(
        numeric_cols=CROP_FEATURE_COLS,
        categorical_cols=[],
        scale_numeric=scale,
    )


def build_fertilizer_preprocessor(
    df: pd.DataFrame,
    feature_cols: list[str],
    scale: bool = True,
) -> ColumnTransformer:
    """
    Return a preprocessor for the fertilizer dataset, adapted to the
    actual columns present in *df*.
    """
    numeric_cols, categorical_cols = split_feature_types(df, feature_cols)
    return build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale_numeric=scale,
    )


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------

def fit_label_encoder(y: pd.Series) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(y)
    return le


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_crop_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a normalised crop DataFrame.

    Steps
    -----
    1. Drop rows where required columns are all NaN.
    2. Drop exact duplicate rows (after reporting count).
    3. Coerce numeric columns to numeric dtype; rows with non-numeric values
       in feature columns are dropped with a warning.
    4. Strip whitespace from the label column.
    """
    df = df.copy()

    # 1. Drop fully-null rows
    df.dropna(how="all", inplace=True)

    # 2. Duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        print(f"[Preprocessing] Crop data: dropping {n_dup} exact duplicate rows.")
        df.drop_duplicates(inplace=True)

    # 3. Coerce numeric features
    for col in CROP_FEATURE_COLS:
        if col in df.columns:
            before = len(df)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            dropped = before - df[col].notna().sum()
            if dropped > 0:
                print(
                    f"[Preprocessing] Crop data: {dropped} non-numeric values "
                    f"in '{col}' coerced to NaN."
                )
    df.dropna(subset=CROP_FEATURE_COLS, inplace=True)

    # 4. Strip label
    if CROP_TARGET_COL in df.columns:
        df[CROP_TARGET_COL] = df[CROP_TARGET_COL].astype(str).str.strip()
        df = df[df[CROP_TARGET_COL].str.len() > 0]

    return df.reset_index(drop=True)


def clean_fertilizer_df(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Clean a normalised fertilizer DataFrame.
    """
    df = df.copy()
    df.dropna(how="all", inplace=True)

    n_dup = df.duplicated().sum()
    if n_dup > 0:
        print(
            f"[Preprocessing] Fertilizer data: dropping {n_dup} exact duplicate rows."
        )
        df.drop_duplicates(inplace=True)

    numeric_cols, _ = split_feature_types(df, feature_cols)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=[c for c in feature_cols if c in df.columns], inplace=True)

    if FERT_TARGET_COL in df.columns:
        df[FERT_TARGET_COL] = df[FERT_TARGET_COL].astype(str).str.strip()
        df = df[df[FERT_TARGET_COL].str.len() > 0]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature / target extraction
# ---------------------------------------------------------------------------

def get_crop_X_y(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target series for the crop model."""
    available_features = [c for c in CROP_FEATURE_COLS if c in df.columns]
    X = df[available_features]
    y = df[CROP_TARGET_COL]
    return X, y


def get_fertilizer_X_y(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target series for the fertilizer model."""
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features]
    y = df[FERT_TARGET_COL]
    return X, y


def determine_fert_features(df: pd.DataFrame) -> list[str]:
    """
    Return the list of feature columns available in *df* for fertilizer
    model training (excludes the target column).
    """
    candidates = FERT_NUMERIC_COLS + FERT_CATEGORICAL_COLS
    return [c for c in candidates if c in df.columns]
