"""Document-grouped, stratified 5-fold CV.

Built ONCE, serialized as a Document -> fold map to a committed JSON file, and
loaded identically on the M2 and on Kaggle. Never regenerate per machine (that
is how OOF arrays silently fail to line up and break the blend).

Group key = Document (every sentence of a document stays in one fold, so topic
and vocabulary cannot leak across the split). Stratify on the 19-level label so
each fold's level distribution is balanced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
SEED = 20260714


def build_document_fold_map(y, groups, n_splits: int = N_SPLITS, seed: int = SEED) -> dict:
    """Return {str(Document): fold_index}. Deterministic given (y, groups, seed).

    StratifiedGroupKFold assigns every row of a group to one fold, so reading the
    fold of any sentence of a document yields that document's fold.
    """
    y = np.asarray([int(v) for v in y])
    groups = np.asarray([str(g) for g in groups])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    doc_fold: dict[str, int] = {}
    for fold, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        for i in val_idx:
            doc_fold[str(groups[i])] = fold
    n_groups = len(set(groups.tolist()))
    assert len(doc_fold) == n_groups, "some documents were not assigned a fold"
    return doc_fold


def fold_of_rows(doc_fold: dict, groups) -> np.ndarray:
    """Map a per-row Document array to a per-row fold-index array."""
    return np.asarray([doc_fold[str(g)] for g in groups], dtype=int)


def save_fold_map(doc_fold: dict, path) -> None:
    Path(path).write_text(json.dumps(doc_fold, ensure_ascii=False, indent=0))


def load_fold_map(path) -> dict:
    return json.loads(Path(path).read_text())


def fold_level_coverage(fold_idx, y) -> dict:
    """Per-fold sorted set of present levels. Used to flag sparse-tail (18/19)
    gaps; QWK still pins labels=[1..19] regardless of coverage."""
    fold_idx = np.asarray(fold_idx, dtype=int)
    y = np.asarray([int(v) for v in y])
    out: dict[int, list[int]] = {}
    for f in sorted(set(fold_idx.tolist())):
        out[f] = sorted(set(y[fold_idx == f].tolist()))
    return out
