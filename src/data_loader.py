"""
data_loader.py
--------------
Handles dataset discovery, ZIP extraction, column normalisation,
schema detection, and basic data profiling for AgriSmart AI.

Design principles
-----------------
* No assumptions about dataset content from filename alone.
* Column names are normalised before matching (lower-case, stripped,
  common spelling variants resolved).
* ZIP archives are extracted safely — no path traversal allowed.
* Duplicate extraction is skipped when contents already exist.
* Reports errors clearly instead of silently inventing data.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------

# Map of common mis-spellings / capitalisation variants → canonical name
_CANONICAL: dict[str, str] = {
    # Crop dataset
    "n": "N",
    "p": "P",
    "k": "K",
    "nitrogen": "N",
    "phosphorus": "P",
    "phosphorous": "P",
    "potassium": "K",
    "temperature": "temperature",
    "temparature": "temperature",   # common typo in datasets
    "temp": "temperature",
    "humidity": "humidity",
    "ph": "ph",
    "rainfall": "rainfall",
    "label": "label",
    "crop": "label",
    "crop_name": "label",
    # Fertilizer dataset
    "moisture": "Moisture",
    "soil type": "Soil Type",
    "soiltype": "Soil Type",
    "crop type": "Crop Type",
    "croptype": "Crop Type",
    "fertilizer name": "Fertilizer Name",
    "fertilizername": "Fertilizer Name",
    "fertiliser name": "Fertilizer Name",
    "fertilizer": "Fertilizer Name",
}

# Crop dataset required columns (canonical)
CROP_REQUIRED = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"}

# Fertilizer dataset required columns (canonical)
FERT_REQUIRED = {"Fertilizer Name"}
FERT_EXPECTED = {
    "Temperature", "Humidity", "Moisture", "Soil Type",
    "Crop Type", "N", "K", "P", "Fertilizer Name",
}


def _normalise_col(col: str) -> str:
    """Return the canonical name for a raw column string."""
    key = col.strip().lower()
    # Remove extra whitespace
    key = re.sub(r"\s+", " ", key)
    return _CANONICAL.get(key, col.strip())


def normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Rename DataFrame columns to their canonical equivalents.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with renamed columns.
    mapping : dict[str, str]
        {original_name: canonical_name} for columns that were renamed.
    """
    mapping: dict[str, str] = {}
    new_cols: list[str] = []
    for col in df.columns:
        canonical = _normalise_col(col)
        if canonical != col:
            mapping[col] = canonical
        new_cols.append(canonical)
    df = df.copy()
    df.columns = new_cols
    return df, mapping


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------

def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract ZIP contents to *dest* preventing path-traversal attacks."""
    for member in zf.infolist():
        member_path = dest / member.filename
        resolved = member_path.resolve()
        if not str(resolved).startswith(str(dest.resolve())):
            raise ValueError(
                f"Unsafe ZIP entry detected (path traversal): {member.filename}"
            )
        zf.extract(member, dest)


def extract_zip(zip_path: Path, datasets_dir: Path) -> Path:
    """
    Extract *zip_path* into a subdirectory of *datasets_dir*.

    The subdirectory name matches the ZIP stem.  Extraction is skipped
    when the subdirectory already contains files from a previous run.

    Returns
    -------
    Path
        Path to the extraction directory.
    """
    dest = datasets_dir / zip_path.stem
    dest.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted
    existing = list(dest.rglob("*"))
    if existing:
        return dest

    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_extract(zf, dest)

    return dest


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def discover_csvs(datasets_dir: Path) -> list[Path]:
    """
    Recursively find all CSV files in *datasets_dir*.

    ZIP archives are extracted first so their contents are included.
    """
    datasets_dir = Path(datasets_dir)
    if not datasets_dir.exists():
        return []

    # Extract ZIPs first
    for zip_path in datasets_dir.rglob("*.zip"):
        try:
            extract_zip(zip_path, zip_path.parent)
        except Exception as exc:
            print(f"[DataLoader] Warning: could not extract {zip_path.name}: {exc}")

    return sorted(datasets_dir.rglob("*.csv"))


# ---------------------------------------------------------------------------
# Dataset loading and profiling
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and return a raw DataFrame."""
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path.name}: {exc}") from exc


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a summary profile of a raw DataFrame."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing": df.isnull().sum().to_dict(),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Determine whether a DataFrame looks like a crop or fertilizer dataset.

    Returns
    -------
    str
        'crop', 'fertilizer', or 'unknown'.
    """
    df_norm, _ = normalise_columns(df)
    cols = set(df_norm.columns)

    crop_score = len(cols & CROP_REQUIRED)
    fert_score = len(cols & FERT_EXPECTED)

    # A crop dataset needs at least the nutrient columns + label
    if crop_score >= 6 and "label" in cols:
        return "crop"
    # Fertilizer dataset needs the target label
    if "Fertilizer Name" in cols and fert_score >= 4:
        return "fertilizer"
    return "unknown"


def validate_crop_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check whether *df* has the columns required for crop model training.

    Returns (is_valid, list_of_missing_canonical_columns).
    """
    df_norm, _ = normalise_columns(df)
    missing = [c for c in CROP_REQUIRED if c not in df_norm.columns]
    return len(missing) == 0, missing


def validate_fertilizer_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check whether *df* has the minimum columns for fertilizer model training.

    Returns (is_valid, list_of_missing_canonical_columns).
    """
    df_norm, _ = normalise_columns(df)
    missing = [c for c in FERT_REQUIRED if c not in df_norm.columns]
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def find_best_dataset(
    csv_paths: list[Path], dataset_type: str
) -> Optional[Path]:
    """
    From *csv_paths*, return the first path whose schema matches *dataset_type*
    ('crop' or 'fertilizer').  Returns None when no match is found.
    """
    for path in csv_paths:
        try:
            df = load_csv(path)
        except RuntimeError:
            continue
        if detect_dataset_type(df) == dataset_type:
            return path
    return None


def load_and_prepare(
    path: Path, dataset_type: str
) -> tuple[pd.DataFrame, dict, dict[str, str]]:
    """
    Load, profile, and normalise a dataset file.

    Returns
    -------
    df_norm : pd.DataFrame
        Normalised DataFrame.
    profile : dict
        Raw profiling statistics.
    col_mapping : dict[str, str]
        Column renames applied.

    Raises
    ------
    ValueError
        When required columns are missing.
    """
    df_raw = load_csv(path)
    profile = profile_dataset(df_raw)
    df_norm, col_mapping = normalise_columns(df_raw)

    if dataset_type == "crop":
        ok, missing = validate_crop_schema(df_norm)
        if not ok:
            raise ValueError(
                f"Crop dataset at '{path.name}' is missing required columns: "
                f"{missing}.  Please check your dataset."
            )
    elif dataset_type == "fertilizer":
        ok, missing = validate_fertilizer_schema(df_norm)
        if not ok:
            raise ValueError(
                f"Fertilizer dataset at '{path.name}' is missing required "
                f"columns: {missing}.  Please check your dataset."
            )

    return df_norm, profile, col_mapping
