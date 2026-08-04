"""Document-grouped CV tests: the grouping property (no document split across
folds), full coverage, determinism, and a real-data smoke on 2026 train."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from harness.cv import (  # noqa: E402
    build_document_fold_map,
    fold_level_coverage,
    fold_of_rows,
)
from harness.data import GROUP, TARGET, load_2026  # noqa: E402


def _toy():
    # 40 documents, each with a random number of sentences, level tied to doc
    rng = np.random.default_rng(0)
    groups, y = [], []
    for d in range(40):
        n = int(rng.integers(3, 12))
        lvl = int(rng.integers(1, 20))
        groups += [f"doc{d}"] * n
        y += [min(19, max(1, lvl + int(rng.integers(-1, 2)))) for _ in range(n)]
    return np.array(y), np.array(groups)


def test_grouping_property_and_coverage():
    y, groups = _toy()
    dmap = build_document_fold_map(y, groups, n_splits=5)
    # every document assigned exactly one fold; all documents covered
    assert set(dmap) == set(groups.tolist())
    fold_idx = fold_of_rows(dmap, groups)
    # grouping property: each document's rows all share one fold
    for g in set(groups.tolist()):
        folds_here = set(fold_idx[groups == g].tolist())
        assert len(folds_here) == 1, f"document {g} split across folds {folds_here}"
    assert set(fold_idx.tolist()) == {0, 1, 2, 3, 4}
    _ = fold_level_coverage(fold_idx, y)


def test_determinism():
    y, groups = _toy()
    m1 = build_document_fold_map(y, groups, seed=20260714)
    m2 = build_document_fold_map(y, groups, seed=20260714)
    assert m1 == m2
    m3 = build_document_fold_map(y, groups, seed=99)
    assert m1 != m3  # different seed -> different assignment (overwhelmingly likely)


def test_real_2026_train_smoke():
    df = load_2026("train")
    y = df[TARGET].to_list()
    groups = df[GROUP].to_list()
    dmap = build_document_fold_map(y, groups)
    assert len(dmap) == df[GROUP].n_unique() == 1518
    fold_idx = fold_of_rows(dmap, groups)
    counts = [int((fold_idx == f).sum()) for f in range(5)]
    # folds should be roughly balanced (no fold under half or over double the mean)
    mean = sum(counts) / 5
    assert all(0.5 * mean < c < 1.6 * mean for c in counts), f"imbalanced folds {counts}"
    cov = fold_level_coverage(fold_idx, y)
    print(f"2026 train fold sizes: {counts}")
    for f in range(5):
        missing = sorted(set(range(1, 20)) - set(cov[f]))
        print(f"  fold {f}: {len(cov[f])} levels present, missing {missing}")


if __name__ == "__main__":
    test_grouping_property_and_coverage()
    test_determinism()
    test_real_2026_train_smoke()
    print("CV TESTS PASSED")
