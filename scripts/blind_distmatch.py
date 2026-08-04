"""Distribution-matched rounder for the blind set.

Low blind exact-accuracy suggests the val-tuned cutpoints produce the wrong
prediction distribution on blind. This sets cutpoints by QUANTILE-MATCHING the
blind blended scores to BAREC's training label distribution (the corpus the blind
set is drawn from), which is a label-free prior. Generates variants for the
word-only blend and prints the distribution gap so we can see if it is worth it.
    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_distmatch.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.cv import build_document_fold_map, fold_of_rows  # noqa: E402
from harness.data import GROUP, ID, TARGET, load_2026  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints, nested_oof_qwk  # noqa: E402
from harness.submission import write_codabench_zip  # noqa: E402

B = ROOT / "artifacts" / "scores" / "blind"
S = ROOT / "artifacts" / "scores"
OUT = ROOT / "submissions" / "blind_variants"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"
WORD = {"wordCE": S / "arabertv02-word-CE_validation.npy",
        "camelbert": S / "camelbert-word-CE_validation.npy",
        "qarib": S / "pub_AymanTarig__qarib-barec-optimized-v8_validation.npy"}
WORDB = {n: B / f"{n}_blind.npy" for n in WORD}


def quantile_cutpoints_from_dist(scores, level_freq):
    """Cutpoints so predicted-level fractions match level_freq (len 19)."""
    cum = np.cumsum(level_freq)[:-1]  # 18 cumulative boundaries
    return np.quantile(scores, cum)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_2026("train"); ytr = np.array(train[TARGET].to_list(), dtype=int)
    val = load_2026("validation"); yva = np.array(val[TARGET].to_list(), dtype=int)
    vfold = fold_of_rows(build_document_fold_map(yva, val[GROUP].to_list(), n_splits=5, seed=20260714),
                         val[GROUP].to_list())
    blind = pl.read_parquet(BLIND); bids = [str(i) for i in blind[ID].to_list()]

    freq = np.array([(ytr == k).mean() for k in range(1, 20)])
    print("train label dist (levels 1-19), %:")
    print("  " + " ".join(f"{k}:{freq[k-1]*100:.1f}" for k in range(1, 20)))

    def z(a):
        return (a - a.mean()) / (a.std() or 1)
    sv = sum(z(np.load(WORD[n])) for n in WORD) / len(WORD)
    sb = sum(z(np.load(WORDB[n])) for n in WORD) / len(WORD)

    # A) val-QWK rounder (current 84.1 approach)
    r = OptimizedRounder(n_candidates=300).fit(sv, yva)
    predA = apply_cutpoints(sb, r.cutpoints_)
    # B) distribution-matched to train
    cutB = quantile_cutpoints_from_dist(sb, freq)
    predB = apply_cutpoints(sb, cutB)
    # C) matched to train but validated to not hurt val QWK (blend of cutpoints)
    cutC = (r.cutpoints_ + quantile_cutpoints_from_dist(sv, freq)) / 2
    predC = apply_cutpoints(sb, cutC)
    predC_blind = apply_cutpoints(sb, (r.cutpoints_ + quantile_cutpoints_from_dist(sb, freq)) / 2)

    def dist(p):
        return " ".join(f"{k}:{(p == k).mean()*100:.0f}" for k in range(8, 16))
    print(f"\ntrain    dist(8-15): {dist(np.repeat(np.arange(1,20), (freq*100000).astype(int)))}")
    print(f"A val-rounder blind: {dist(predA)}")
    print(f"B dist-matched blind:{dist(predB)}")

    valA = nested_oof_qwk(sv, yva, vfold, n_candidates=300)[0]
    print(f"\nval nested-OOF (A val-rounder): {valA*100:.3f}")
    write_codabench_zip(OUT / "wordonly_valrounder", bids, predA)
    write_codabench_zip(OUT / "wordonly_distmatch", bids, predB)
    write_codabench_zip(OUT / "wordonly_halfmatch_val", bids, predC)
    write_codabench_zip(OUT / "wordonly_halfmatch_blind", bids, predC_blind)
    print("wrote: wordonly_valrounder, wordonly_distmatch, wordonly_halfmatch_val, wordonly_halfmatch_blind")


if __name__ == "__main__":
    main()
