"""Freeze the canonical Document -> fold map for 2026 train, ONCE.

This file is committed. The M2 and Kaggle both load artifacts/folds_2026_train.json
so their OOF arrays line up exactly. Never regenerate per machine.

    python scripts/make_folds.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.cv import (  # noqa: E402
    N_SPLITS,
    SEED,
    build_document_fold_map,
    fold_level_coverage,
    fold_of_rows,
    save_fold_map,
)
from harness.data import GROUP, TARGET, load_2026  # noqa: E402

OUT = ROOT / "artifacts" / "folds_2026_train.json"
META = ROOT / "artifacts" / "folds_2026_train.meta.json"


def main():
    df = load_2026("train")
    y = df[TARGET].to_list()
    groups = df[GROUP].to_list()
    dmap = build_document_fold_map(y, groups, n_splits=N_SPLITS, seed=SEED)
    save_fold_map(dmap, OUT)

    fold_idx = fold_of_rows(dmap, groups)
    counts = {int(f): int((fold_idx == f).sum()) for f in range(N_SPLITS)}
    cov = fold_level_coverage(fold_idx, y)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]
    meta = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_documents": len(dmap),
        "n_sentences": df.height,
        "target": TARGET,
        "group": GROUP,
        "fold_sentence_counts": counts,
        "fold_levels_missing": {
            int(f): sorted(set(range(1, 20)) - set(cov[f])) for f in range(N_SPLITS)
        },
        "sha256_16": digest,
        "source": "CAMeL-Lab/BAREC-Shared-Task-2026-sent train",
    }
    META.write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT.name} ({len(dmap)} docs) sha={digest}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
