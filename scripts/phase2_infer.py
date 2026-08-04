"""Phase 2 - run each public checkpoint ONCE over train/val/test, cache scores.

One model at a time (8GB M2). d3tok members get the gold D3Tok column (from
build_d3tok.py); word members get the raw Sentence. Continuous score per row is
cached to artifacts/scores/<member>_<split>.npy for the blend step.

Run build_d3tok.py first. Close other memory hogs; watch memguard.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/phase2_infer.py [member ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.checkpoints import CHECKPOINTS, run_checkpoint  # noqa: E402
from harness.data import ID, load_2026  # noqa: E402

SCORES = ROOT / "artifacts" / "scores"
PROC = ROOT / "data" / "processed"
SPLITS = ["train", "validation", "test"]


def gold_texts(split: str, df: pl.DataFrame, col: str) -> list[str]:
    """Aligned gold preprocessing column (D3Tok or Word) for a split, joined by
    ID to the build_d3tok output so byte-parity input reaches each checkpoint."""
    path = PROC / f"d3tok_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/build_d3tok.py first")
    gold = pl.read_parquet(path).select([pl.col(ID).cast(pl.Utf8), col])
    joined = df.select(pl.col(ID).cast(pl.Utf8)).join(gold, on=ID, how="left")
    out = joined[col].to_list()
    miss = sum(1 for c in out if c is None)
    if miss:
        print(f"    WARN {split}: {miss} rows missing {col} (need camel-tools fallback)", flush=True)
    return [c if c is not None else "" for c in out]


def main(members):
    SCORES.mkdir(parents=True, exist_ok=True)
    dfs = {s: load_2026(s) for s in SPLITS}
    for name in members:
        spec = CHECKPOINTS[name]
        t0 = time.time()
        print(f"\n=== {name} ({spec['repo']}, {spec['head']}, {spec['tok']}) ===", flush=True)
        for split in SPLITS:
            out = SCORES / f"{name}_{split}.npy"
            if out.exists():
                print(f"    {split}: cached, skip", flush=True)
                continue
            df = dfs[split]
            col = "D3Tok" if spec["tok"] == "d3tok" else "Word"
            texts = gold_texts(split, df, col)
            scores = run_checkpoint(spec["repo"], texts, spec["head"])
            np.save(out, scores)
            print(f"    {split}: n={len(scores)} mean={scores.mean():.3f} "
                  f"std={scores.std():.3f} range=[{scores.min():.2f},{scores.max():.2f}] "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or list(CHECKPOINTS.keys())
    main(args)
