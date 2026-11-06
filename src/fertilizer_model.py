"""
fertilizer_model.py
-------------------
Fertilizer recommendation model: training, saving, loading, and prediction.

Trains a RandomForestClassifier (with a Decision Tree and KNN for comparison)
on fertilizer datasets. Categorical features are encoded via OrdinalEncoder
inside the pipeline.

If the fertilizer dataset is unavailable, this module reports the limitation
clearly and does not fabricate a model.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    FERT_TARGET_COL,
    build_fertilizer_preprocessor,
    clean_fertilizer_df,
    determine_fert_features,
    get_fertilizer_X_y,
)
from src.evaluation import evaluate_classifier, select_best_model

RANDOM_SEED = 42
TEST_SIZE = 0.2
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_fert_pipelines(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, Pipeline]:
    """Return training pipelines adapted to the actual fertilizer schema."""
    return {
        "Decision Tree": Pipeline(
            [
                (
                    "preprocessor",
                    build_fertilizer_preprocessor(df, feature_cols, scale=False),
                ),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=10, min_samples_leaf=3, random_state=RANDOM_SEED
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "preprocessor",
                    build_fertilizer_preprocessor(df, feature_cols, scale=False),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                (
                    "preprocessor",
                    build_fertilizer_preprocessor(df, feature_cols, scale=True),
                ),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fertilizer_models(
    df: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, Pipeline], str, dict]:
    """
    Clean, split, train all fertilizer models, evaluate on test set.

    Returns
    -------
    results : dict[str, dict]
    pipelines : dict[str, Pipeline]
    best_name : str
    split_info : dict
    """
    feature_cols = determine_fert_features(df)
    if not feature_cols:
        raise ValueError(
            "No usable feature columns found in the fertilizer dataset. "
            f"Expected some of: Temperature, Humidity, Moisture, Soil Type, "
            f"Crop Type, N, K, P."
        )

    df_clean = clean_fertilizer_df(df, feature_cols)
    if len(df_clean) < 30:
        raise ValueError(
            f"Fertilizer dataset has only {len(df_clean)} usable rows after "
            "cleaning. At least 30 rows are needed."
        )

    X, y = get_fertilizer_X_y(df_clean, feature_cols)

    # Use stratify only when all classes have >= 2 samples
    value_counts = y.value_counts()
    can_stratify = (value_counts >= 2).all()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y if can_stratify else None,
    )

    classes = sorted(y.unique())
    pipelines = _build_fert_pipelines(df_clean, feature_cols)
    results: dict[str, dict] = {}

    for name, pipe in pipelines.items():
        print(f"[FertModel] Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate_classifier(y_test.values, y_pred, class_names=classes)
        results[name] = metrics
        print(
            f"[FertModel]   {name}: accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    best_name = select_best_model(results, metric="f1")
    print(f"[FertModel] Best model: {best_name}")

    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_classes": len(classes),
        "classes": classes,
        "feature_cols": feature_cols,
    }

    return results, pipelines, best_name, split_info


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_fertilizer_model(
    pipeline: Pipeline,
    model_name: str,
    results: dict[str, dict],
    split_info: dict,
    models_dir: Path = MODELS_DIR,
) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "fertilizer_model.joblib"
    meta_path = models_dir / "fertilizer_metadata.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": model_name,
        "trained_at": datetime.now().isoformat(),
        "feature_cols": split_info["feature_cols"],
        "classes": split_info["classes"],
        "train_size": split_info["train_size"],
        "test_size": split_info["test_size"],
        "n_classes": split_info["n_classes"],
        "all_results": {
            name: {k: v for k, v in m.items() if k != "confusion_matrix"}
            for name, m in results.items()
        },
        "best_model_metrics": {
            k: v for k, v in results[model_name].items() if k != "confusion_matrix"
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"[FertModel] Saved model to {model_path}")
    return model_path


def load_fertilizer_model(
    models_dir: Path = MODELS_DIR,
) -> tuple[Pipeline, dict]:
    model_path = models_dir / "fertilizer_model.joblib"
    meta_path = models_dir / "fertilizer_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Fertilizer model not found at {model_path}. "
            "Run: python scripts/train_fertilizer_model.py"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Fertilizer model metadata not found at {meta_path}. "
            "Run: python scripts/train_fertilizer_model.py"
        )

    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    return pipeline, metadata


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_fertilizer(
    pipeline: Pipeline,
    metadata: dict,
    input_values: dict,
) -> dict:
    """
    Predict the recommended fertilizer.

    Parameters
    ----------
    input_values : dict
        Keys matching metadata['feature_cols']; values are str or float.

    Returns
    -------
    dict with keys: fertilizer, confidence, input_values.
    """
    feature_cols = metadata["feature_cols"]
    row = {col: [input_values.get(col, np.nan)] for col in feature_cols}
    X = pd.DataFrame(row)

    fertilizer = pipeline.predict(X)[0]
    result = {
        "fertilizer": fertilizer,
        "confidence": None,
        "input_values": input_values,
    }

    if hasattr(pipeline[-1], "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        classes = pipeline.classes_
        best_idx = int(np.argmax(proba))
        result["confidence"] = float(proba[best_idx])

    return result
