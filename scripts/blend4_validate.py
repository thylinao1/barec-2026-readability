"""Offline reproduction gate for the 4-member blend (runs on cached val/test .npy only).

Purpose: prove the blend code path in this file is IDENTICAL in semantics to the one that
produced HANDOFF section 4a/4b/4c, BEFORE the MARBERTv2 blind scores exist. If the numbers
below reproduce, the same functions can be trusted to build the blind submission.

Reproduction targets (HANDOFF 2026-07-25):
  solo test QWK   wordCE 84.52 | camelbert 83.34 | qarib8 82.66 | marbert 81.96
  wordonly3 (equal)                      85.07
  + marbert (equal)                      85.48
  + marbert (tuned weights)              85.59
  + marbert (tuned) + doc-shrink a=0.1   85.69
  corr(marbert, wordonly3 blend)         0.945

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blend4_validate.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.data import GROUP, TARGET, load_2026  # noqa: E402
from harness.metrics import official_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints  # noqa: E402

S = ROOT / "artifacts" / "scores"
N_CAND = 300  # matches blind_distmatch.py / blind_reweighted_prior.py

# member -> cached score-file stem (val and test share the stem)
STEMS = {
    "wordCE": "arabertv02-word-CE",
    "camelbert": "camelbert-word-CE",
    "qarib8": "pub_AymanTarig__qarib-barec-optimized-v8",
    "marbert": "marbertv2-reg",
}
TUNED = {"wordCE": 0.306, "camelbert": 0.231, "qarib8": 0.231, "marbert": 0.231}


def z(a):
    """Per-split self-standardization -- EXACTLY as in blind_distmatch.py."""
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def load(split):
    return {m: z(np.load(S / f"{stem}_{split}.npy")) for m, stem in STEMS.items()}


def blend(zs, weights):
    """Weight-normalized mean of standardized member scores."""
    tot = sum(weights.values())
    return sum(weights[m] * zs[m] for m in weights) / tot


def doc_shrink(scores, docs, alpha):
    """score := (1-a)*score + a*document_mean(score)."""
    if alpha == 0:
        return scores
    out = np.asarray(scores, float).copy()
    docs = np.asarray([str(d) for d in docs])
    means = np.empty_like(out)
    for d in np.unique(docs):
        m = docs == d
        means[m] = out[m].mean()
    return (1 - alpha) * out + alpha * means


def main():
    val, test = load_2026("validation"), load_2026("test")
    yva = np.array(val[TARGET].to_list(), int)
    yte = np.array(test[TARGET].to_list(), int)
    zv, zt = load("validation"), load("test")
    print(f"val n={len(yva)}  test n={len(yte)}")

    def evaluate(weights, alpha=0.0):
        """Val-tune the rounder, apply to test. The honest protocol: train is
        contaminated (public checkpoints saw it), so tune on val, score once on test."""
        sv = blend(zv, weights)
        st = blend(zt, weights)
        if alpha:
            sv = doc_shrink(sv, val[GROUP].to_list(), alpha)
            st = doc_shrink(st, test[GROUP].to_list(), alpha)
        r = OptimizedRounder(n_candidates=N_CAND).fit(sv, yva)
        return official_qwk(yte, apply_cutpoints(st, r.cutpoints_)) * 100, st

    print("\n-- solo test QWK (expect wordCE 84.52 camelbert 83.34 qarib8 82.66 marbert 81.96)")
    for m in STEMS:
        q, _ = evaluate({m: 1.0})
        print(f"   {m:11s} {q:6.2f}")

    word3 = {"wordCE": 1.0, "camelbert": 1.0, "qarib8": 1.0}
    all4 = {**word3, "marbert": 1.0}

    print("\n-- blends (val-tuned rounder -> clean test)")
    rows = [
        ("wordonly3 (equal)          [85.07]", word3, 0.0),
        ("+ marbert (equal)          [85.48]", all4, 0.0),
        ("+ marbert (tuned weights)  [85.59]", TUNED, 0.0),
        ("+ marbert (tuned) a=0.1    [85.69]", TUNED, 0.1),
    ]
    got = {}
    for label, w, a in rows:
        q, _ = evaluate(w, a)
        got[label] = q
        print(f"   {label:38s} {q:6.2f}")

    print("\n-- doc-shrinkage sweep on the tuned 4-member blend (expect peak at a=0.1)")
    for a in (0.0, 0.1, 0.2, 0.3, 0.5):
        q, _ = evaluate(TUNED, a)
        print(f"   alpha={a:<4} {q:6.2f}")

    _, st3 = evaluate(word3)
    c = np.corrcoef(zt["marbert"], blend(zt, word3))[0, 1]
    print(f"\n-- corr(marbert, wordonly3 blend) on test = {c:.4f}   (expect ~0.945)")

    print("\n-- correlation matrix of the 4 standardized members (test)")
    ms = list(STEMS)
    print("            " + "".join(f"{m:>11s}" for m in ms))
    for a in ms:
        print(f"   {a:9s}" + "".join(f"{np.corrcoef(zt[a], zt[b])[0,1]:11.4f}" for b in ms))

    base = got["wordonly3 (equal)          [85.07]"]
    best = max(got.values())
    print(f"\nGATE: wordonly3 reproduces 85.07? got {base:.2f}  "
          f"(delta {base - 85.07:+.2f})")
    print(f"GATE: best 4-member blend {best:.2f}  gain over wordonly3 {best - base:+.2f}")


if __name__ == "__main__":
    main()
