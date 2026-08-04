"""Calibration using a METADATA-REWEIGHTED label prior.

The half-match trick that scored 84.5 quantile-matched blind scores to the RAW train label
distribution. But blind's composition differs sharply from train (STEM 6.9%->17.8%,
Specialized 26.9%->44.7%), so the raw train prior is the wrong target. This reweights the
prior by blind's known Domain x Text_Class mix -- label-free, uses only metadata shipped
with the blind set -- then applies the same ~50% interpolation with the val-tuned rounder.
    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_reweighted_prior.py
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.data import ID, TARGET, load_2026  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints  # noqa: E402
from harness.submission import write_codabench_zip  # noqa: E402

S = ROOT / "artifacts" / "scores"
B = S / "blind"
OUT = ROOT / "submissions" / "blind_reweighted"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"
VAL = {"wordCE": S / "arabertv02-word-CE_validation.npy",
       "camelbert": S / "camelbert-word-CE_validation.npy",
       "qarib": S / "pub_AymanTarig__qarib-barec-optimized-v8_validation.npy"}
BLD = {"wordCE": B / "wordCE_blind.npy", "camelbert": B / "camelbert_blind.npy",
       "qarib": B / "qarib_blind.npy"}


def z(a):
    a = a.ravel()
    return (a - a.mean()) / (a.std() or 1)


def reweighted_prior(train, ytr, blind, cols):
    """P(level) = sum_s P(s | blind) * P(level | s, train), using metadata only."""
    raw = np.array([(ytr == k).mean() for k in range(1, 20)])
    tv = np.array(["|".join(str(train[c].to_list()[i]) for c in cols) for i in range(train.height)])
    bv = np.array(["|".join(str(blind[c].to_list()[i]) for c in cols) for i in range(blind.height)])
    out = np.zeros(19)
    for c in sorted(set(bv)):
        pb = (bv == c).mean()
        m = tv == c
        out += pb * (np.array([(ytr[m] == k).mean() for k in range(1, 20)]) if m.sum() > 0 else raw)
    return out / out.sum(), raw


def cutpoints_from_prior(scores, prior):
    return np.quantile(scores, np.cumsum(prior)[:-1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_2026("train"); ytr = np.array(train[TARGET].to_list(), int)
    val = load_2026("validation"); yva = np.array(val[TARGET].to_list(), int)
    blind = pl.read_parquet(BLIND); bids = [str(i) for i in blind[ID].to_list()]

    sv = sum(z(np.load(p)) for p in VAL.values()) / len(VAL)
    sb = sum(z(np.load(p)) for p in BLD.values()) / len(BLD)
    rounder = OptimizedRounder(n_candidates=300).fit(sv, yva)

    prior_j, raw = reweighted_prior(train, ytr, blind, ["Domain", "Text_Class"])
    print(f"L1(reweighted vs raw) = {np.abs(prior_j - raw).sum():.4f}")

    def stage(tag, prior, frac):
        cut = np.sort((1 - frac) * rounder.cutpoints_ + frac * cutpoints_from_prior(sb, prior))
        pred = apply_cutpoints(sb, cut)
        write_codabench_zip(OUT / tag, bids, pred)
        d = " ".join(f"{k}:{(pred == k).mean()*100:.0f}" for k in range(10, 16))
        print(f"  {tag:28s} dist(10-15) {d}")
        return pred

    # champion baseline for reference (raw prior, 50%) then the reweighted variants
    stage("raw50_reference", raw, 0.50)
    stage("reweighted50", prior_j, 0.50)
    stage("reweighted60", prior_j, 0.60)
    stage("reweighted40", prior_j, 0.40)
    print(f"\nzips in {OUT}")


if __name__ == "__main__":
    main()
