"""
tests/test_data_loader.py
--------------------------
Unit tests for data_loader.py: dataset discovery, ZIP extraction safety,
column normalisation, schema detection, and data profiling.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    _normalise_col,
    detect_dataset_type,
    discover_csvs,
    extract_zip,
    find_best_dataset,
    load_and_prepare,
    normalise_columns,
    profile_dataset,
    validate_crop_schema,
    validate_fertilizer_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def crop_df():
    return pd.DataFrame({
        "N": [90, 85, 60],
        "P": [42, 58, 55],
        "K": [43, 41, 44],
        "temperature": [20.88, 21.77, 23.0],
        "humidity": [82.0, 80.3, 82.0],
        "ph": [6.5, 7.0, 7.2],
        "rainfall": [202.9, 226.6, 263.9],
        "label": ["rice", "rice", "maize"],
    })


@pytest.fixture
def fert_df():
    return pd.DataFrame({
        "Temperature": [26, 29, 34],
        "Humidity": [52, 52, 65],
        "Moisture": [38, 45, 62],
        "Soil Type": ["Sandy", "Loamy", "Black"],
        "Crop Type": ["Maize", "Sugarcane", "Cotton"],
        "N": [37, 12, 20],
        "K": [0, 0, 0],
        "P": [0, 36, 20],
        "Fertilizer Name": ["Urea", "DAP", "14-35-14"],
    })


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

class TestNormaliseCol:
    def test_lowercase_n(self):
        assert _normalise_col("n") == "N"

    def test_temparature_typo(self):
        assert _normalise_col("Temparature") == "temperature"

    def test_temperature_correct(self):
        assert _normalise_col("temperature") == "temperature"

    def test_phosphorous_variant(self):
        assert _normalise_col("Phosphorous") == "P"

    def test_phosphorus_variant(self):
        assert _normalise_col("Phosphorus") == "P"

    def test_potassium(self):
        assert _normalise_col("potassium") == "K"

    def test_unknown_col_preserved(self):
        assert _normalise_col("MyCustomColumn") == "MyCustomColumn"

    def test_strips_whitespace(self):
        assert _normalise_col("  n  ") == "N"

    def test_soil_type(self):
        assert _normalise_col("Soil Type") == "Soil Type"

    def test_fertilizer_name(self):
        assert _normalise_col("Fertilizer Name") == "Fertilizer Name"


class TestNormaliseColumns:
    def test_renames_typo_column(self):
        df = pd.DataFrame({"Temparature": [20.0], "ph": [6.5], "label": ["rice"]})
        df_norm, mapping = normalise_columns(df)
        assert "temperature" in df_norm.columns
        assert "Temparature" not in df_norm.columns
        assert "Temparature" in mapping

    def test_preserves_unknown_columns(self):
        df = pd.DataFrame({"MyCol": [1], "N": [90]})
        df_norm, mapping = normalise_columns(df)
        assert "MyCol" in df_norm.columns
        assert "N" in df_norm.columns


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

class TestDetectDatasetType:
    def test_detects_crop(self, crop_df):
        assert detect_dataset_type(crop_df) == "crop"

    def test_detects_fertilizer(self, fert_df):
        assert detect_dataset_type(fert_df) == "fertilizer"

    def test_unknown_for_empty(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        assert detect_dataset_type(df) == "unknown"


class TestValidateCropSchema:
    def test_valid_crop(self, crop_df):
        ok, missing = validate_crop_schema(crop_df)
        assert ok
        assert missing == []

    def test_missing_label(self):
        df = pd.DataFrame({
            "N": [90], "P": [42], "K": [43],
            "temperature": [20.0], "humidity": [82.0],
            "ph": [6.5], "rainfall": [202.9],
        })
        ok, missing = validate_crop_schema(df)
        assert not ok
        assert "label" in missing

    def test_typo_column_accepted(self):
        df = pd.DataFrame({
            "N": [90], "P": [42], "K": [43],
            "Temparature": [20.0], "humidity": [82.0],
            "ph": [6.5], "rainfall": [202.9], "label": ["rice"],
        })
        ok, missing = validate_crop_schema(df)
        assert ok, f"Expected valid schema; missing: {missing}"


class TestValidateFertSchema:
    def test_valid_fert(self, fert_df):
        ok, missing = validate_fertilizer_schema(fert_df)
        assert ok

    def test_missing_target(self):
        df = pd.DataFrame({"Temperature": [26], "Humidity": [52]})
        ok, missing = validate_fertilizer_schema(df)
        assert not ok
        assert "Fertilizer Name" in missing


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

class TestProfileDataset:
    def test_row_count(self, crop_df):
        profile = profile_dataset(crop_df)
        assert profile["rows"] == 3

    def test_missing_count(self):
        df = pd.DataFrame({"A": [1, None, 3], "B": [None, None, 3]})
        profile = profile_dataset(df)
        assert profile["missing_total"] == 3

    def test_duplicate_count(self):
        df = pd.DataFrame({"A": [1, 1, 2]})
        profile = profile_dataset(df)
        assert profile["duplicates"] == 1


# ---------------------------------------------------------------------------
# ZIP extraction safety
# ---------------------------------------------------------------------------

class TestZipExtraction:
    def test_safe_extraction(self, tmp_path):
        """A normal ZIP should extract successfully."""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "col1,col2\n1,2\n")

        dest = extract_zip(zip_path, tmp_path)
        assert (dest / "data.csv").exists()

    def test_skips_if_already_extracted(self, tmp_path):
        """Extraction should be skipped when destination already has files."""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "col1\n1\n")

        # First extraction
        dest = extract_zip(zip_path, tmp_path)
        mtime_first = (dest / "data.csv").stat().st_mtime

        # Second extraction — file should not be re-extracted
        extract_zip(zip_path, tmp_path)
        mtime_second = (dest / "data.csv").stat().st_mtime

        assert mtime_first == mtime_second

    def test_path_traversal_rejected(self, tmp_path):
        """Path traversal entries must raise ValueError."""
        zip_path = tmp_path / "evil.zip"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("../../evil.csv")
            zf.writestr(info, "malicious content")
        with open(zip_path, "wb") as f:
            f.write(buf.getvalue())

        with pytest.raises(ValueError, match="path traversal"):
            extract_zip(zip_path, tmp_path / "safe_dir")


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

class TestDiscoverCsvs:
    def test_finds_csvs(self, tmp_path):
        (tmp_path / "a.csv").write_text("col\n1\n")
        (tmp_path / "b.csv").write_text("col\n2\n")
        csvs = discover_csvs(tmp_path)
        assert len(csvs) == 2

    def test_empty_directory(self, tmp_path):
        csvs = discover_csvs(tmp_path)
        assert csvs == []

    def test_nonexistent_directory(self, tmp_path):
        csvs = discover_csvs(tmp_path / "nonexistent")
        assert csvs == []


# ---------------------------------------------------------------------------
# load_and_prepare
# ---------------------------------------------------------------------------

class TestLoadAndPrepare:
    def test_raises_on_missing_crop_cols(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("A,B,C\n1,2,3\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_and_prepare(csv, "crop")

    def test_succeeds_with_valid_crop(self, tmp_path, crop_df):
        csv = tmp_path / "crop.csv"
        crop_df.to_csv(csv, index=False)
        df, profile, mapping = load_and_prepare(csv, "crop")
        assert len(df) == 3
