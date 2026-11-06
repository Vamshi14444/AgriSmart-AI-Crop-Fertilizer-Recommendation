"""
tests/test_new_features.py
--------------------------
Unit tests for the new features:
1. PDF Field Advisory generator
2. Head-to-Head Crop Comparison
3. Seasonal Crop Calendar
"""

from __future__ import annotations

import pytest

from src.crop_calendar import (
    CROP_CALENDAR,
    build_crop_timeline_chart,
    get_calendar_info,
)
from src.crop_comparison import (
    CROP_PROFILES,
    compare_two_crops,
    get_all_crop_names,
    get_crop_profile,
)
from src.report_generator import generate_field_advisory_pdf


class TestReportGenerator:
    def test_pdf_generation_returns_valid_bytes(self):
        inputs = {
            "N": 90.0, "P": 42.0, "K": 43.0,
            "temperature": 25.0, "humidity": 80.0,
            "ph": 6.5, "rainfall": 200.0,
        }
        top3 = [
            {"crop": "rice", "probability": 0.98},
            {"crop": "jute", "probability": 0.015},
            {"crop": "maize", "probability": 0.005},
        ]
        pdf_bytes = generate_field_advisory_pdf(
            input_values=inputs,
            crop_predicted="rice",
            confidence=0.98,
            top3=top3,
            fertilizer_predicted="Urea",
            region_note="Punjab Farm 4B",
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        # Check PDF magic header '%PDF-'
        assert pdf_bytes.startswith(b"%PDF-")


class TestCropComparison:
    def test_all_22_crops_profiled(self):
        names = get_all_crop_names()
        assert len(names) == 22
        assert "rice" in names
        assert "maize" in names
        assert "cotton" in names

    def test_profile_keys(self):
        p = get_crop_profile("rice")
        assert "name" in p
        assert "water_req_mm" in p
        assert "duration_days" in p
        assert "nitrogen_fixing" in p

    def test_compare_two_crops_dataframe(self):
        df = compare_two_crops("rice", "chickpea")
        assert df.shape[0] >= 10
        assert df.shape[1] == 2


class TestCropCalendar:
    def test_all_crops_in_calendar(self):
        for crop in ["rice", "maize", "chickpea", "cotton", "coffee"]:
            info = get_calendar_info(crop)
            assert "season_type" in info
            assert "sowing_months" in info
            assert "harvest_months" in info

    def test_build_crop_timeline_chart(self):
        fig = build_crop_timeline_chart(["rice", "maize", "wheat"])
        assert fig is not None
        assert len(fig.data) > 0
