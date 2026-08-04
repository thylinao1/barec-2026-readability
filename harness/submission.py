"""Submission writer + validator for the Codabench sentence-level board (16544).

Verified live from the competition Submission page (2026-07-14):
- a single file named EXACTLY 'prediction' (NO extension), header
  'Sentence ID,Prediction', rows 'ID,level', integer level on the 1..19 scale;
- that file zipped into 'prediction.zip' (zip prediction.zip prediction) and
  uploaded via My Submissions.
The scorer joins by ID and checks set equality, so order does not matter but
membership must match the split one-to-one. pandas strips a UTF-8 BOM on read,
so a BOM is harmless; the packer writes plain UTF-8 to match the documented
literal example.
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path

HEADER = ["Sentence ID", "Prediction"]


def validate_predictions(ids, preds) -> None:
    ids = [str(i) for i in ids]
    assert len(ids) == len(preds), "id/pred length mismatch"
    assert len(set(ids)) == len(ids), "duplicate IDs in submission"
    for p in preds:
        pi = int(p)
        assert float(p) == pi, f"non-integer prediction: {p!r}"
        assert 1 <= pi <= 19, f"prediction out of [1,19]: {p!r}"


def write_submission(path, ids, preds, bom: bool = False) -> str:
    """Write a single CSV of (ID, prediction). bom=True matches the 2025 example
    fixture's UTF-8 BOM; default plain UTF-8 matches the 2026 Submission page."""
    ids = [str(i) for i in ids]
    preds = [int(p) for p in preds]
    validate_predictions(ids, preds)
    enc = "utf-8-sig" if bom else "utf-8"
    with open(path, "w", newline="", encoding=enc) as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(zip(ids, preds))
    return str(path)


def write_codabench_zip(out_dir, ids, preds) -> str:
    """Produce the exact upload artifact for board 16544: a file named
    'prediction' (no extension) zipped into 'prediction.zip'. Returns the zip
    path. The inner arcname is 'prediction' regardless of the on-disk path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "prediction"
    write_submission(pred_file, ids, preds, bom=False)
    zip_path = out_dir / "prediction.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_file, arcname="prediction")
    return str(zip_path)


def validate_against_split(path, gold_ids) -> bool:
    """Header, integer range, and exact ID-set match against the split's IDs.
    Uses pandas (which strips a UTF-8 BOM on read, exactly like eval.py)."""
    import pandas as pd

    df = pd.read_csv(path)
    assert list(df.columns) == HEADER, f"bad header: {list(df.columns)}"
    pred_ids = {str(i) for i in df["Sentence ID"]}
    gold = {str(i) for i in gold_ids}
    missing, extra = gold - pred_ids, pred_ids - gold
    assert not missing and not extra, (
        f"ID set mismatch: {len(missing)} missing, {len(extra)} extra"
    )
    assert len(df) == len(gold), f"row count {len(df)} != gold {len(gold)}"
    for p in df["Prediction"]:
        assert 1 <= int(p) <= 19, f"prediction out of [1,19]: {p!r}"
    return True
