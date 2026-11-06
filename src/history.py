"""
history.py
----------
Stores and retrieves recent predictions in a local CSV file.

No personal information is collected. Records contain only:
- timestamp, input values, crop prediction, fertilizer prediction (if available).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

HISTORY_FILE = Path("artifacts") / "prediction_history.csv"

_COLUMNS = [
    "timestamp",
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "crop_predicted",
    "crop_confidence",
    "fertilizer_predicted",
    "fertilizer_confidence",
    "region_note",
]


def _ensure_file() -> None:
    """Create the history CSV with headers if it does not exist."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        pd.DataFrame(columns=_COLUMNS).to_csv(HISTORY_FILE, index=False)


def add_record(
    input_values: dict,
    crop_predicted: Optional[str] = None,
    crop_confidence: Optional[float] = None,
    fertilizer_predicted: Optional[str] = None,
    fertilizer_confidence: Optional[float] = None,
    region_note: Optional[str] = None,
) -> None:
    """Append a prediction record to the history CSV."""
    _ensure_file()
    row = {col: None for col in _COLUMNS}
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        row[col] = input_values.get(col)
    row["crop_predicted"] = crop_predicted
    row["crop_confidence"] = (
        round(crop_confidence, 4) if crop_confidence is not None else None
    )
    row["fertilizer_predicted"] = fertilizer_predicted
    row["fertilizer_confidence"] = (
        round(fertilizer_confidence, 4) if fertilizer_confidence is not None else None
    )
    row["region_note"] = region_note

    df_new = pd.DataFrame([row])
    df_new.to_csv(HISTORY_FILE, mode="a", header=False, index=False)


def load_history() -> pd.DataFrame:
    """Load and return the full prediction history."""
    _ensure_file()
    try:
        df = pd.read_csv(HISTORY_FILE)
        # Ensure all expected columns exist (handle older files)
        for col in _COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=_COLUMNS)


def clear_history() -> None:
    """Clear the prediction history file."""
    pd.DataFrame(columns=_COLUMNS).to_csv(HISTORY_FILE, index=False)
