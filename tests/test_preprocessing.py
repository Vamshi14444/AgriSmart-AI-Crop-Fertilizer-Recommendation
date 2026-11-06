"""
tests/test_preprocessing.py
----------------------------
Unit tests for preprocessing.py: cleaning, feature/target splitting,
pipeline building, and fertilizer feature detection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    CROP_FEATURE_COLS,
    CROP_TARGET_COL,
    FERT_TARGET_COL,
    build_crop_preprocessor,
    build_fertilizer_preprocessor,
    clean_crop_df,
    clean_fertilizer_df,
    determine_fert_features,
    get_crop_X_y,
    get_fertilizer_X_y,
    split_feature_types,
)


@pytest.fixture
def crop_df():
    return pd.DataFrame({
        "N": [90.0, 85.0, 60.0],
        "P": [42.0, 58.0, 55.0],
        "K": [43.0, 41.0, 44.0],
        "temperature": [20.88, 21.77, 23.0],
        "humidity": [82.0, 80.3, 82.0],
        "ph": [6.5, 7.0, 7.2],
        "rainfall": [202.9, 226.6, 263.9],
        "label": ["rice", "rice", "maize"],
    })


@pytest.fixture
def fert_df():
    return pd.DataFrame({
        "Temperature": [26.0, 29.0, 34.0],
        "Humidity": [52.0, 52.0, 65.0],
        "Moisture": [38.0, 45.0, 62.0],
        "Soil Type": ["Sandy", "Loamy", "Black"],
        "Crop Type": ["Maize", "Sugarcane", "Cotton"],
        "N": [37.0, 12.0, 20.0],
        "K": [0.0, 0.0, 0.0],
        "P": [0.0, 36.0, 20.0],
        "Fertilizer Name": ["Urea", "DAP", "14-35-14"],
    })


class TestSplitFeatureTypes:
    def test_numeric_categorical(self, fert_df):
        cols = ["Temperature", "Humidity", "Soil Type", "Crop Type"]
        num, cat = split_feature_types(fert_df, cols)
        assert "Temperature" in num
        assert "Soil Type" in cat

    def test_missing_col_skipped(self, crop_df):
        cols = ["N", "NonExistent"]
        num, cat = split_feature_types(crop_df, cols)
        assert "N" in num
        assert "NonExistent" not in num
        assert "NonExistent" not in cat


class TestCleanCropDf:
    def test_removes_duplicates(self, crop_df):
        df_dup = pd.concat([crop_df, crop_df.iloc[:1]], ignore_index=True)
        df_clean = clean_crop_df(df_dup)
        assert len(df_clean) == len(crop_df)

    def test_coerces_non_numeric(self):
        df = pd.DataFrame({
            "N": ["90", "bad", "60"],
            "P": [42.0, 58.0, 55.0],
            "K": [43.0, 41.0, 44.0],
            "temperature": [20.88, 21.77, 23.0],
            "humidity": [82.0, 80.3, 82.0],
            "ph": [6.5, 7.0, 7.2],
            "rainfall": [202.9, 226.6, 263.9],
            "label": ["rice", "rice", "maize"],
        })
        df_clean = clean_crop_df(df)
        assert len(df_clean) == 2  # 'bad' row dropped

    def test_strips_label_whitespace(self):
        df = pd.DataFrame({
            "N": [90.0], "P": [42.0], "K": [43.0],
            "temperature": [20.0], "humidity": [82.0],
            "ph": [6.5], "rainfall": [202.9],
            "label": ["  rice  "],
        })
        df_clean = clean_crop_df(df)
        assert df_clean["label"].iloc[0] == "rice"

    def test_drops_all_nan_rows(self):
        df = pd.DataFrame({
            "N": [None, 85.0], "P": [None, 58.0], "K": [None, 41.0],
            "temperature": [None, 21.77], "humidity": [None, 80.3],
            "ph": [None, 7.0], "rainfall": [None, 226.6],
            "label": [None, "rice"],
        })
        df_clean = clean_crop_df(df)
        assert len(df_clean) == 1


class TestCleanFertilizerDf:
    def test_removes_duplicates(self, fert_df):
        feature_cols = determine_fert_features(fert_df)
        df_dup = pd.concat([fert_df, fert_df.iloc[:1]], ignore_index=True)
        df_clean = clean_fertilizer_df(df_dup, feature_cols)
        assert len(df_clean) == len(fert_df)


class TestGetCropXy:
    def test_returns_correct_shapes(self, crop_df):
        X, y = get_crop_X_y(crop_df)
        assert X.shape == (3, len(CROP_FEATURE_COLS))
        assert len(y) == 3

    def test_target_values(self, crop_df):
        _, y = get_crop_X_y(crop_df)
        assert set(y) == {"rice", "maize"}


class TestGetFertXy:
    def test_returns_correct_shapes(self, fert_df):
        feature_cols = determine_fert_features(fert_df)
        X, y = get_fertilizer_X_y(fert_df, feature_cols)
        assert len(y) == 3


class TestBuildPreprocessors:
    def test_crop_preprocessor_no_scale(self, crop_df):
        X, y = get_crop_X_y(crop_df)
        pre = build_crop_preprocessor(scale=False)
        result = pre.fit_transform(X)
        assert result.shape[0] == 3

    def test_crop_preprocessor_with_scale(self, crop_df):
        X, y = get_crop_X_y(crop_df)
        pre = build_crop_preprocessor(scale=True)
        result = pre.fit_transform(X)
        assert result.shape[0] == 3

    def test_fert_preprocessor(self, fert_df):
        feature_cols = determine_fert_features(fert_df)
        X, y = get_fertilizer_X_y(fert_df, feature_cols)
        pre = build_fertilizer_preprocessor(fert_df, feature_cols, scale=True)
        result = pre.fit_transform(X)
        assert result.shape[0] == 3


class TestDetermineFertFeatures:
    def test_returns_available_features(self, fert_df):
        features = determine_fert_features(fert_df)
        assert "Temperature" in features
        assert "Fertilizer Name" not in features

    def test_excludes_missing_columns(self):
        df = pd.DataFrame({"Temperature": [26], "Fertilizer Name": ["Urea"]})
        features = determine_fert_features(df)
        assert "Temperature" in features
        assert "Humidity" not in features
