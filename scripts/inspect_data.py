"""
scripts/inspect_data.py
-----------------------
Utility script to inspect datasets in the datasets/ directory.
Reports discovered files, detected schema type, column names,
row count, missing values, and duplicate rows.

Usage
-----
    python scripts/inspect_data.py
    python scripts/inspect_data.py --path datasets/my_file.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    detect_dataset_type,
    discover_csvs,
    load_csv,
    normalise_columns,
    profile_dataset,
)


def inspect_file(path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"File : {path}")
    print(f"{'='*60}")

    try:
        df_raw = load_csv(path)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    profile = profile_dataset(df_raw)
    df_norm, mapping = normalise_columns(df_raw)
    dtype = detect_dataset_type(df_raw)

    print(f"  Rows           : {profile['rows']}")
    print(f"  Columns        : {profile['columns']}")
    print(f"  Detected type  : {dtype}")
    print(f"  Missing values : {profile['missing_total']}")
    print(f"  Duplicate rows : {profile['duplicates']}")

    if mapping:
        print(f"\n  Column name normalisations applied:")
        for orig, can in mapping.items():
            print(f"    '{orig}' → '{can}'")

    print(f"\n  Data types:")
    for col, dt in profile["dtypes"].items():
        missing = profile["missing"].get(col, 0)
        print(f"    {col:<30} {dt:<15} missing={missing}")

    print(f"\n  First 3 rows (raw):")
    print(df_raw.head(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect datasets.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Inspect a specific CSV file. Omit to scan datasets/ directory.",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Dataset directory to scan (default: datasets/).",
    )
    args = parser.parse_args()

    if args.path:
        if not args.path.exists():
            print(f"ERROR: File not found: {args.path}")
            sys.exit(1)
        inspect_file(args.path)
    else:
        csvs = discover_csvs(args.datasets_dir)
        if not csvs:
            print(
                f"\nNo CSV files found in '{args.datasets_dir}'.\n"
                "Please add your datasets to the datasets/ directory.\n"
                "See datasets/README.md for instructions."
            )
            sys.exit(0)

        print(f"\nFound {len(csvs)} CSV file(s) in '{args.datasets_dir}':\n")
        for path in csvs:
            inspect_file(path)


if __name__ == "__main__":
    main()
