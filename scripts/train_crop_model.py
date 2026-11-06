"""
scripts/train_crop_model.py
---------------------------
Discovers the crop dataset, trains all configured classifiers,
evaluates on the held-out test set, and saves the best model.

Usage
-----
    python scripts/train_crop_model.py
    python scripts/train_crop_model.py --datasets-dir datasets --models-dir models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crop_model import save_crop_model, train_crop_models
from src.data_loader import (
    discover_csvs,
    find_best_dataset,
    load_and_prepare,
)
from src.evaluation import compare_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop recommendation models.")
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("datasets"),
        help="Directory containing crop dataset CSV files.",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="Directory where the trained model is saved.",
    )
    args = parser.parse_args()

    print("\n[train_crop_model] Discovering datasets...")
    csvs = discover_csvs(args.datasets_dir)
    if not csvs:
        print(
            f"ERROR: No CSV files found in '{args.datasets_dir}'.\n"
            "Please add your crop dataset. See datasets/README.md."
        )
        sys.exit(1)

    crop_path = find_best_dataset(csvs, "crop")
    if crop_path is None:
        print(
            "ERROR: No crop dataset detected among the available CSV files.\n"
            "Expected columns: N, P, K, temperature, humidity, ph, rainfall, label.\n"
            "Files inspected:"
        )
        for p in csvs:
            print(f"  {p}")
        sys.exit(1)

    print(f"[train_crop_model] Using dataset: {crop_path}")

    try:
        df, profile, mapping = load_and_prepare(crop_path, "crop")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(
        f"[train_crop_model] Dataset: {profile['rows']} rows, "
        f"{len(profile['columns'])} columns, "
        f"{profile['missing_total']} missing values, "
        f"{profile['duplicates']} duplicates."
    )

    print("\n[train_crop_model] Starting model training...\n")
    try:
        results, pipelines, best_name, split_info = train_crop_models(df)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("\n[train_crop_model] Model comparison (test set):")
    comparison = compare_models(results)
    print(comparison.to_string())

    best_pipeline = pipelines[best_name]
    save_crop_model(
        pipeline=best_pipeline,
        model_name=best_name,
        results=results,
        split_info=split_info,
        models_dir=args.models_dir,
    )

    print(f"\n[train_crop_model] ✅ Done. Best model: {best_name}")
    print(
        f"  Accuracy : {results[best_name]['accuracy']:.4f}\n"
        f"  F1-Score : {results[best_name]['f1']:.4f}\n"
        f"  Precision: {results[best_name]['precision']:.4f}\n"
        f"  Recall   : {results[best_name]['recall']:.4f}"
    )
    print(f"\n  Model saved to: {args.models_dir / 'crop_model.joblib'}")


if __name__ == "__main__":
    main()
