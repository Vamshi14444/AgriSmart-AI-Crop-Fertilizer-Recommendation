"""
tests/test_fertilizer_model.py
-------------------------------
Tests for fertilizer model training, saving, loading, and prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fertilizer_model import (
    load_fertilizer_model,
    predict_fertilizer,
    save_fertilizer_model,
    train_fertilizer_models,
)


def _make_fert_df(n: int = 60) -> pd.DataFrame:
    """Generate a synthetic but valid fertilizer DataFrame."""
    rng = np.random.default_rng(42)
    soil_types = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
    crop_types = ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy"]
    ferts = ["Urea", "DAP", "14-35-14", "28-28", "17-17-17", "20-20"]
    per_fert = n // len(ferts)
    rows = []
    for fert in ferts:
        for _ in range(per_fert):
            rows.append({
                "Temperature": float(rng.integers(15, 40)),
                "Humidity": float(rng.integers(30, 80)),
                "Moisture": float(rng.integers(20, 70)),
                "Soil Type": rng.choice(soil_types),
                "Crop Type": rng.choice(crop_types),
                "N": float(rng.integers(0, 50)),
                "K": float(rng.integers(0, 50)),
                "P": float(rng.integers(0, 50)),
                "Fertilizer Name": fert,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def fert_df():
    return _make_fert_df(60)


@pytest.fixture
def trained_fert(fert_df):
    return train_fertilizer_models(fert_df)


class TestTrainFertilizerModels:
    def test_returns_three_models(self, trained_fert):
        results, pipelines, best_name, split_info = trained_fert
        assert len(pipelines) == 3

    def test_best_in_pipelines(self, trained_fert):
        _, pipelines, best_name, _ = trained_fert
        assert best_name in pipelines

    def test_metrics_valid(self, trained_fert):
        results, _, _, _ = trained_fert
        for m in results.values():
            assert 0.0 <= m["accuracy"] <= 1.0

    def test_raises_on_too_few_rows(self):
        tiny_df = _make_fert_df(n=6)
        with pytest.raises(ValueError, match="usable rows"):
            train_fertilizer_models(tiny_df)

    def test_raises_on_no_fert_cols(self):
        df = pd.DataFrame({"A": [1, 2, 3], "Fertilizer Name": ["X", "Y", "Z"]})
        with pytest.raises(ValueError):
            train_fertilizer_models(df)


class TestSaveLoadFertilizerModel:
    def test_save_and_load(self, tmp_path, trained_fert):
        results, pipelines, best_name, split_info = trained_fert
        save_fertilizer_model(
            pipeline=pipelines[best_name],
            model_name=best_name,
            results=results,
            split_info=split_info,
            models_dir=tmp_path,
        )
        pipeline, metadata = load_fertilizer_model(models_dir=tmp_path)
        assert metadata["model_name"] == best_name

    def test_load_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_fertilizer_model(models_dir=tmp_path)


class TestPredictFertilizer:
    def test_returns_string(self, tmp_path, trained_fert):
        results, pipelines, best_name, split_info = trained_fert
        save_fertilizer_model(
            pipeline=pipelines[best_name],
            model_name=best_name,
            results=results,
            split_info=split_info,
            models_dir=tmp_path,
        )
        pipeline, metadata = load_fertilizer_model(models_dir=tmp_path)
        result = predict_fertilizer(
            pipeline,
            metadata,
            {
                "Temperature": 28.0,
                "Humidity": 55.0,
                "Moisture": 40.0,
                "Soil Type": "Sandy",
                "Crop Type": "Maize",
                "N": 20.0,
                "K": 10.0,
                "P": 5.0,
            },
        )
        assert isinstance(result["fertilizer"], str)
        assert result["fertilizer"] in split_info["classes"]

    def test_invalid_category_does_not_crash(self, tmp_path, trained_fert):
        """Unseen categorical values should not raise an exception."""
        results, pipelines, best_name, split_info = trained_fert
        save_fertilizer_model(
            pipeline=pipelines[best_name],
            model_name=best_name,
            results=results,
            split_info=split_info,
            models_dir=tmp_path,
        )
        pipeline, metadata = load_fertilizer_model(models_dir=tmp_path)
        result = predict_fertilizer(
            pipeline,
            metadata,
            {
                "Temperature": 28.0,
                "Humidity": 55.0,
                "Moisture": 40.0,
                "Soil Type": "UNKNOWN_SOIL_TYPE",  # never seen during training
                "Crop Type": "UNKNOWN_CROP",
                "N": 20.0,
                "K": 10.0,
                "P": 5.0,
            },
        )
        assert isinstance(result["fertilizer"], str)
