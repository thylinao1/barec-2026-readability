"""Byte-parity BAREC scoring. Mirrors ref/barec-shared-task-2025/scripts/eval.py.

Two QWK entry points:
- official_qwk: exact mirror of eval.py (cohen_kappa_score with NO labels arg).
  Use for board-parity and full-support sets (Dev, OOF over all of train).
- fold_qwk: pins labels=[1..19] so a fold missing the sparse tail (18/19) does
  not silently build a smaller confusion matrix and return a number the official
  scorer would not. Use for per-fold OOF QWK.

On a set containing all 19 levels the two coincide; the parity test proves it.
QWK is the ONLY ranking metric. The rest of the suite exists to cross-check
parity against eval.py, not to be optimized.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

LEVELS = list(range(1, 20))  # 1..19 inclusive

# Collapse maps lifted verbatim from eval.py. Do not edit.
_BAREC_7 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4,
            11: 4, 12: 5, 13: 5, 14: 6, 15: 6, 16: 7, 17: 7, 18: 7, 19: 7}
_BAREC_5 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 2,
            11: 2, 12: 3, 13: 3, 14: 4, 15: 4, 16: 5, 17: 5, 18: 5, 19: 5}
_BAREC_3 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1,
            11: 1, 12: 2, 13: 2, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3}


def _ints(a) -> list[int]:
    return [int(x) for x in a]


def official_qwk(y_true, y_pred) -> float:
    """Exact mirror of eval.py QWK: weights='quadratic', label set inferred from
    the data (no labels arg). Raw kappa in [-1, 1]; multiply by 100 for percent."""
    return float(cohen_kappa_score(_ints(y_true), _ints(y_pred), weights="quadratic"))


def fold_qwk(y_true, y_pred, labels=LEVELS) -> float:
    """QWK with the full 1..19 support pinned, for per-fold OOF where the sparse
    tail may be absent. Coincides with official_qwk on full-support sets."""
    return float(
        cohen_kappa_score(_ints(y_true), _ints(y_pred), weights="quadratic", labels=labels)
    )


def all_metrics(y_true, y_pred) -> dict:
    """Full eval.py metric suite as raw fractions (eval.py prints these * 100)."""
    yt, yp = _ints(y_true), _ints(y_pred)
    return {
        "acc": accuracy_score(yt, yp),
        "acc_pm1": float(np.mean([abs(p - l) <= 1 for p, l in zip(yp, yt)])),
        "dist": mean_absolute_error(yt, yp),
        "qwk": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "acc7": accuracy_score([_BAREC_7[l] for l in yt], [_BAREC_7[p] for p in yp]),
        "acc5": accuracy_score([_BAREC_5[l] for l in yt], [_BAREC_5[p] for p in yp]),
        "acc3": accuracy_score([_BAREC_3[l] for l in yt], [_BAREC_3[p] for p in yp]),
    }


def format_like_evalpy(metrics: dict) -> list[str]:
    """Reproduce eval.py's exact printed lines from an all_metrics() dict, so a
    test can assert string-level byte parity with the official scorer."""
    return [
        f"Accuracy: {metrics['acc'] * 100:.4f}%",
        f"Accuracy +/-1: {metrics['acc_pm1'] * 100:.4f}%",
        f"Average absolute distance: {metrics['dist']:.6f}",
        f"Quadratic Cohen's Kappa: {metrics['qwk'] * 100:.4f}%",
        f"Accuracy (7 levels): {metrics['acc7'] * 100:.4f}%",
        f"Accuracy (5 levels): {metrics['acc5'] * 100:.4f}%",
        f"Accuracy (3 levels): {metrics['acc3'] * 100:.4f}%",
    ]
