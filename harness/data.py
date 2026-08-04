"""Canonical data access.

2026 competition data (parquet, the real task) and the 2025 parity gold (csv,
used only to byte-check the scorer). TARGET and GROUP are the two load-bearing
column names: the scorer keys on Readability_Level_19 (verified in eval.py), and
CV groups by Document.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
D26 = ROOT / "data" / "raw" / "barec-2026-sent" / "data"
D25 = ROOT / "data" / "raw" / "barec-2025-sent"

TARGET = "Readability_Level_19"
GROUP = "Document"
ID = "ID"
TEXT = "Sentence"

# Columns that leak topic identity; for stratification / error analysis only.
LEAKY = ["Document", "Source", "Book", "Author", "Domain", "Text_Class"]


def load_2026(split: str) -> pl.DataFrame:
    """split in {train, validation, test}."""
    return pl.read_parquet(D26 / f"{split}-00000-of-00001.parquet")


def load_2025(split: str) -> pl.DataFrame:
    """split in {train, dev, test}."""
    return pl.read_csv(D25 / f"{split}.csv")
