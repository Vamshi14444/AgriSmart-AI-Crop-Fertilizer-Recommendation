"""
tests/test_crop_model.py
------------------------
Tests for crop model training, saving, loading, and prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.crop_model import (
    get_feature_importance,
    load_crop_model,
    predict_crop,
    save_crop_model,
    train_crop_models,
)


def _make_crop_df(n: int = 110) -> pd.DataFrame:
    """Generate a synthetic but valid crop DataFrame."""
    rng = np.random.default_rng(42)
    labels = ["rice", "maize", "wheat", "cotton", "jute"]
    per_class = n // len(labels)
    rows = []
    for lbl in labels:
        for _ in range(per_class):
            rows.append({
                "N": float(rng.integers(0, 150)),
                "P": float(rng.integers(0, 100)),
                "K": float(rng.integers(0, 200)),
                "temperature": float(rng.uniform(10, 40)),
                "humidity": float(rng.uniform(20, 90)),
                "ph": float(rng.uniform(4.5, 8.5)),
                "rainfall": float(rng.uniform(20, 2000)),
                "label": lbl,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def crop_df():
    return _make_crop_df(110)


@pytest.fixture
def trained_models(crop_df):
    return train_crop_models(crop_df)


class TestTrainCropModels:
    def test_returns_three_models(self, trained_models):
        results, pipelines, best_name, split_info = trained_models
        assert len(pipelines) == 3

    def test_best_name_is_one_of_trained(self, trained_models):
        results, pipelines, best_name, split_info = trained_models
        assert best_name in pipelines

    def test_metrics_in_range(self, trained_models):
        results, _, _, _ = trained_models
        for name, m in results.items():
            assert 0.0 <= m["accuracy"] <= 1.0
            assert 0.0 <= m["f1"] <= 1.0

    def test_split_info_keys(self, trained_models):
        _, _, _, split_info = trained_models
        assert "train_size" in split_info
        assert "test_size" in split_info
        assert "classes" in split_info

    def test_raises_on_too_few_rows(self):
        tiny_df = _make_crop_df(n=10)
        with pytest.raises(ValueError, match="usable rows"):
            train_crop_models(tiny_df)


class TestSaveLoadCropModel:
    def test_save_and_load(self, tmp_path, trained_models):
        results, pipelines, best_name, split_info = trained_models
        save_crop_model(
            pipeline=pipelines[best_name],
            model_name=best_name,
            results=results,
            split_info=split_info,
            models_dir=tmp_path,
        )
        pipeline, metadata = load_crop_model(models_dir=tmp_path)
        assert metadata["model_name"] == best_name
        assert "feature_cols" in metadata

    def test_load_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_crop_model(models_dir=tmp_path)


class TestPredictCrop:
    def test_returns_crop_string(self, tmp_path, trained_models):
        results, pipelines, best_name, split_info = trained_models
        save_crop_model(
            pipeline=pipelines[best_name],
            model_name=best_name,
            results=results,
            split_info=split_info,
            models_dir=tmp_path,
        )
        pipeline, metadata = load_crop_model(models_dir=tmp_path)
        result = predict_crop(
            pipeline,
            metadata,
            {
                "N": 90.0, "P": 42.0, "K": 43.0,
                "temperature": 20.88, "humidity": 82.0,
                "ph": 6.5, "rainfall": 202.9,
            },
        )
        assert isinstance(result["crop"], str)
        assert result["crop"] in split_info["classes"]

    def test_confidence_between_0_and_1(self, tmp_path, trained_models):
        results, pipelines, best_name, split_info = trained_models
        save_crop_model(
            pipeline=pipelines[best_name],
            model_name=best_name,
            results=results,
            split_info=split_info,
            models_dir=tmp_path,
        )
        pipeline, metadata = load_crop_model(models_dir=tmp_path)
        result = predict_crop(
            pipeline, metadata,
            {"N": 90.0, "P": 42.0, "K": 43.0,
             "temperature": 20.88, "humidity": 82.0,
             "ph": 6.5, "rainfall": 202.9},
        )
        if result["confidence"] is not None:
            assert 0.0 <= result["confidence"] <= 1.0


class TestFeatureImportance:
    def test_rf_returns_dataframe(self, trained_models):
        _, pipelines, _, _ = trained_models
        rf_pipe = pipelines.get("Random Forest")
        if rf_pipe is None:
            pytest.skip("Random Forest not in pipelines")
        from src.preprocessing import CROP_FEATURE_COLS
        fi_df = get_feature_importance(rf_pipe, CROP_FEATURE_COLS)
        assert fi_df is not None
        assert "Feature" in fi_df.columns
        assert "Importance" in fi_df.columns
