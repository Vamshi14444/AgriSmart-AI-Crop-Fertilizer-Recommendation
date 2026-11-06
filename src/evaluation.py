"""
evaluation.py
-------------
Model evaluation utilities: metrics, confusion matrix, and reporting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    average: str = "weighted",
) -> dict:
    """
    Compute classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, report_dict,
                    confusion_matrix, class_names.
    """
    labels = sorted(set(y_true) | set(y_pred))
    if class_names is None:
        class_names = [str(l) for l in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "report_dict": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "labels": labels,
        "n_samples": len(y_true),
    }


def summarise_metrics(metrics: dict) -> pd.DataFrame:
    """Convert a metrics dict to a single-row summary DataFrame."""
    return pd.DataFrame(
        {
            "Accuracy": [round(metrics["accuracy"], 4)],
            "Precision": [round(metrics["precision"], 4)],
            "Recall": [round(metrics["recall"], 4)],
            "F1-Score": [round(metrics["f1"], 4)],
            "Test Samples": [metrics["n_samples"]],
        }
    )


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a dict of {model_name: metrics_dict}.
    """
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1"], 4),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def select_best_model(results: dict[str, dict], metric: str = "f1") -> str:
    """Return the model name with the highest value for *metric*."""
    return max(results, key=lambda name: results[name][metric])
