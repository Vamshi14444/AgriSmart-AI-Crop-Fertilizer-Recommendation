"""
crop_model.py
-------------
Crop recommendation model: training, saving, loading, and prediction.

Three algorithms are trained and compared:
  1. Decision Tree Classifier
  2. Random Forest Classifier
  3. K-Nearest Neighbours Classifier

The best model (by weighted F1 on the held-out test set) is saved.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import (
    CROP_FEATURE_COLS,
    CROP_TARGET_COL,
    build_crop_preprocessor,
    clean_crop_df,
    get_crop_X_y,
)
from src.evaluation import evaluate_classifier, select_best_model

RANDOM_SEED = 42
TEST_SIZE = 0.2
MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_pipelines() -> dict[str, Pipeline]:
    """Return a dict of {name: sklearn Pipeline} for crop classification."""
    return {
        "Decision Tree": Pipeline(
            [
                ("preprocessor", build_crop_preprocessor(scale=False)),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=10, min_samples_leaf=5, random_state=RANDOM_SEED
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_crop_preprocessor(scale=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("preprocessor", build_crop_preprocessor(scale=True)),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_crop_models(
    df: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, Pipeline], str, dict]:
    """
    Clean, split, train all crop models, evaluate on test set.

    Returns
    -------
    results : dict[str, dict]
        Evaluation metrics per model.
    pipelines : dict[str, Pipeline]
        Fitted pipelines.
    best_name : str
        Name of the best-performing model.
    split_info : dict
        Train/test sizes and class count.
    """
    df_clean = clean_crop_df(df)
    if len(df_clean) < 50:
        raise ValueError(
            f"Crop dataset has only {len(df_clean)} usable rows after cleaning. "
            "At least 50 rows are needed for meaningful training."
        )

    X, y = get_crop_X_y(df_clean)
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipelines = _build_pipelines()
    results: dict[str, dict] = {}

    classes = sorted(y.unique())

    for name, pipe in pipelines.items():
        print(f"[CropModel] Training {name}...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate_classifier(y_test.values, y_pred, class_names=classes)
        results[name] = metrics
        print(
            f"[CropModel]   {name}: accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    best_name = select_best_model(results, metric="f1")
    print(f"[CropModel] Best model: {best_name}")

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

def save_crop_model(
    pipeline: Pipeline,
    model_name: str,
    results: dict[str, dict],
    split_info: dict,
    models_dir: Path = MODELS_DIR,
) -> Path:
    """Save the best pipeline and metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "crop_model.joblib"
    meta_path = models_dir / "crop_metadata.json"

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

    print(f"[CropModel] Saved model to {model_path}")
    return model_path


def load_crop_model(
    models_dir: Path = MODELS_DIR,
) -> tuple[Pipeline, dict]:
    """
    Load the saved crop pipeline and metadata.

    Raises
    ------
    FileNotFoundError
        When model files are absent.
    """
    model_path = models_dir / "crop_model.joblib"
    meta_path = models_dir / "crop_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Crop model not found at {model_path}. "
            "Run: python scripts/train_crop_model.py"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Crop model metadata not found at {meta_path}. "
            "Run: python scripts/train_crop_model.py"
        )

    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    return pipeline, metadata


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_crop(
    pipeline: Pipeline,
    metadata: dict,
    input_values: dict,
) -> dict:
    """
    Predict the most suitable crop and return a structured result.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    metadata : dict (from load_crop_model)
    input_values : dict
        Keys matching CROP_FEATURE_COLS; values are floats.

    Returns
    -------
    dict with keys: crop, confidence, top3, input_values, feature_cols.
    """
    feature_cols = metadata["feature_cols"]
    row = {col: [input_values.get(col, np.nan)] for col in feature_cols}
    X = pd.DataFrame(row)

    crop = pipeline.predict(X)[0]
    result = {"crop": crop, "confidence": None, "top3": [], "input_values": input_values}

    if hasattr(pipeline[-1], "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        classes = pipeline.classes_
        top_idx = np.argsort(proba)[::-1][:3]
        result["confidence"] = float(proba[top_idx[0]])
        result["top3"] = [
            {"crop": str(classes[i]), "probability": float(proba[i])}
            for i in top_idx
        ]

    return result


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(pipeline: Pipeline, feature_cols: list[str]) -> Optional[pd.DataFrame]:
    """
    Extract feature importances from a Random Forest or Decision Tree
    pipeline.  Returns None for models that do not support it.
    """
    clf = pipeline[-1]
    if not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    # Numeric features passed through the preprocessor in same order
    if len(importances) == len(feature_cols):
        return pd.DataFrame(
            {"Feature": feature_cols, "Importance": importances}
        ).sort_values("Importance", ascending=False)
    return None
