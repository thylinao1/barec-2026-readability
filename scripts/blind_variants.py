"""Generate several principled BLIND submission variants from cached member scores
(run blind_cache_scores.py first). Ranks configs by val nested-OOF and writes a
prediction.zip per config (+ a conservative-tail cap@17 version). Since the blind
gold is hidden, val nested-OOF is the only offline guide; the board's live score
is the tie-breaker for the final pick.
    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_variants.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.cv import build_document_fold_map, fold_of_rows  # noqa: E402
from harness.data import GROUP, ID, TARGET, load_2026  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints, nested_oof_qwk  # noqa: E402
from harness.submission import write_codabench_zip  # noqa: E402

B = ROOT / "artifacts" / "scores" / "blind"
S = ROOT / "artifacts" / "scores"
OUT = ROOT / "submissions" / "blind_variants"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"

VAL = {  # val score with the SAME preprocessing used on blind
    "reg": B / "reg_valmyd3.npy", "dtCE": B / "dtCE_valmyd3.npy",
    "wordCE": S / "arabertv02-word-CE_validation.npy",
    "camelbert": S / "camelbert-word-CE_validation.npy",
    "qarib": S / "pub_AymanTarig__qarib-barec-optimized-v8_validation.npy",
}
BLD = {n: B / f"{n}_blind.npy" for n in VAL}

CONFIGS = {
    "wordonly3": ["wordCE", "camelbert", "qarib"],
    "full_reg": ["reg", "wordCE", "qarib"],
    "dtCE3": ["dtCE", "wordCE", "qarib"],
    "reg_dtCE_word_qarib": ["reg", "dtCE", "wordCE", "qarib"],
    "word4_reg": ["reg", "wordCE", "camelbert", "qarib"],
    "all5": ["reg", "dtCE", "wordCE", "camelbert", "qarib"],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val = load_2026("validation"); yva = np.array(val[TARGET].to_list(), dtype=int)
    vfold = fold_of_rows(build_document_fold_map(yva, val[GROUP].to_list(), n_splits=5, seed=20260714),
                         val[GROUP].to_list())
    blind = pl.read_parquet(BLIND); bids = [str(i) for i in blind[ID].to_list()]

    def z(a):
        return (a - a.mean()) / (a.std() or 1)
    vsc = {n: z(np.load(VAL[n])) for n in VAL}
    bsc = {n: z(np.load(BLD[n])) for n in VAL}

    results = []
    for tag, members in CONFIGS.items():
        sv = sum(vsc[m] for m in members) / len(members)
        sb = sum(bsc[m] for m in members) / len(members)
        vq = nested_oof_qwk(sv, yva, vfold, n_candidates=200)[0]
        rounder = OptimizedRounder(n_candidates=300).fit(sv, yva)
        pred = apply_cutpoints(sb, rounder.cutpoints_)
        write_codabench_zip(OUT / tag, bids, pred)
        # conservative-tail cap@17 (blind sets have near-zero 18/19)
        capped = pred.copy(); capped[capped > 17] = 17
        write_codabench_zip(OUT / f"{tag}_cap17", bids, capped)
        results.append((tag, vq, int((pred > 17).sum())))
        print(f"  {tag:22s} val_nOOF={vq*100:.3f}  (n>17 in blind: {int((pred>17).sum())})", flush=True)

    print("\nranked by val nested-OOF (best offline guide; blind is hidden):")
    for tag, vq, n17 in sorted(results, key=lambda x: -x[1]):
        print(f"  {vq*100:.3f}  {tag}  (+ {tag}_cap17)")
    print(f"\nzips in: {OUT}")


if __name__ == "__main__":
    main()
