"""PHASE 0 EXIT GATE.

The harness is "done" only when this passes. It reproduces the official eval.py
output to byte parity on the provided Dev fixture, checked three ways:

  1. formatted-string parity: every line my harness prints equals the line
     eval.py prints (README-documented and reproduced by running eval.py):
        Accuracy: 56.6211%
        Accuracy +/-1: 69.8632%
        Average absolute distance: 1.143776
        Quadratic Cohen's Kappa: 80.0040%
        Accuracy (7 levels): 65.8687%
        Accuracy (5 levels): 70.2736%
        Accuracy (3 levels): 76.4569%
  2. numeric parity to the documented reference constants.
  3. official_qwk (no labels) == fold_qwk (labels=[1..19]) on this full-support
     set, so both call styles are proven equivalent when all 19 levels appear.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from harness.metrics import all_metrics, fold_qwk, format_like_evalpy, official_qwk  # noqa: E402

EVALPY_REFERENCE_LINES = [
    "Accuracy: 56.6211%",
    "Accuracy +/-1: 69.8632%",
    "Average absolute distance: 1.143776",
    "Quadratic Cohen's Kappa: 80.0040%",
    "Accuracy (7 levels): 65.8687%",
    "Accuracy (5 levels): 70.2736%",
    "Accuracy (3 levels): 76.4569%",
]


def _load_aligned():
    gold = pl.read_csv(ROOT / "data/raw/barec-2025-sent/dev.csv")
    fixture = pl.read_csv(ROOT / "ref/barec-shared-task-2025/examples/Dev_Sentence_Level.csv")
    pred_of = {str(i): int(p) for i, p in zip(fixture["Sentence ID"], fixture["Prediction"])}
    ids = [str(i) for i in gold["ID"]]
    y_true = [int(v) for v in gold["Readability_Level_19"]]
    y_pred = [pred_of[i] for i in ids]
    assert len(y_true) == 7310
    return y_true, y_pred


def test_parity():
    y_true, y_pred = _load_aligned()
    metrics = all_metrics(y_true, y_pred)

    # 1. formatted-string byte parity with eval.py
    lines = format_like_evalpy(metrics)
    for got, ref in zip(lines, EVALPY_REFERENCE_LINES):
        assert got == ref, f"line mismatch:\n  got: {got}\n  ref: {ref}"

    # 2. numeric parity
    ref = dict(qwk=0.800040, acc=0.566211, acc_pm1=0.698632, dist=1.143776,
               acc7=0.658687, acc5=0.702736, acc3=0.764569)
    for k, v in ref.items():
        assert math.isclose(metrics[k], v, abs_tol=1e-5), f"{k}: {metrics[k]:.6f} != {v}"

    # 3. no-labels vs labels=[1..19] equivalence on full support
    o = official_qwk(y_true, y_pred)
    f = fold_qwk(y_true, y_pred)
    assert math.isclose(o, f, abs_tol=1e-12), f"official {o} != fold {f}"
    assert math.isclose(o, 0.800040, abs_tol=1e-5)

    print("BYTE-PARITY GATE PASSED")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    test_parity()
