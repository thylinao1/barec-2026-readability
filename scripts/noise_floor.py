"""How large is the MEASUREMENT NOISE of our offline evaluation?

Motivation: blend4_forktest.py showed the SAME blend on the SAME data moving 85.44 -> 85.28 ->
85.42 as the rounder grid went 200 -> 300 -> 400 candidates, and the marbert weight sweep was
non-monotonic (w=0.10 -> 85.285 but w=0.15 -> 85.706). If the procedure's own noise is ~0.15
QWK then every gain claimed in run log section 4b (+0.41 marbert, +0.11 tuned weights, +0.22
doc-shrinkage) is at or below the noise floor and cannot be trusted as a real effect.

Three independent estimates of the noise, and one of the honest effect size:
  A) rounder grid resolution:  vary n_candidates, everything else fixed
  B) test-set sampling noise:  bootstrap the test rows with cutpoints FROZEN, so this
                               isolates evaluation noise from tuning noise
  C) fold-seed noise:          vary the val fold seed in the honest nested-OOF protocol
  D) paired bootstrap on the DELTA (all4_tuned - wordonly3), which is the number that
     actually decides whether to spend a submission

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/noise_floor.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.cv import build_document_fold_map, fold_of_rows  # noqa: E402
from harness.data import GROUP, TARGET, load_2026  # noqa: E402
from harness.metrics import official_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints, nested_oof_qwk  # noqa: E402

S = ROOT / "artifacts" / "scores"
STEMS = {"wordCE": "arabertv02-word-CE", "camelbert": "camelbert-word-CE",
         "qarib8": "pub_AymanTarig__qarib-barec-optimized-v8", "marbert": "marbertv2-reg"}
WORD3 = {"wordCE": 1.0, "camelbert": 1.0, "qarib8": 1.0}
TUNED = {"wordCE": 0.306, "camelbert": 0.231, "qarib8": 0.231, "marbert": 0.231}


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def load(split):
    return {m: z(np.load(S / f"{stem}_{split}.npy")) for m, stem in STEMS.items()}


def blend(zs, w):
    return sum(w[m] * zs[m] for m in w) / sum(w.values())


def main():
    val, test = load_2026("validation"), load_2026("test")
    yva = np.array(val[TARGET].to_list(), int)
    yte = np.array(test[TARGET].to_list(), int)
    zv, zt = load("validation"), load("test")
    rng = np.random.default_rng(20260725)

    def fit_cuts(w, n_cand):
        return OptimizedRounder(n_candidates=n_cand).fit(blend(zv, w), yva).cutpoints_

    print("=" * 76)
    print("A) ROUNDER GRID NOISE: identical data, only n_candidates changes")
    for label, w in (("wordonly3", WORD3), ("all4 tuned", TUNED)):
        qs = []
        for n in (150, 200, 250, 300, 350, 400, 450, 500):
            q = official_qwk(yte, apply_cutpoints(blend(zt, w), fit_cuts(w, n))) * 100
            qs.append(q)
        qs = np.array(qs)
        print(f"   {label:11s} " + " ".join(f"{q:.2f}" for q in qs))
        print(f"   {'':11s} mean {qs.mean():.3f}  sd {qs.std():.3f}  "
              f"range {qs.max()-qs.min():.3f}")

    print("\n" + "=" * 76)
    print("B) TEST-SAMPLING NOISE: cutpoints FROZEN (n_cand=300), bootstrap test rows")
    cuts3, cuts4 = fit_cuts(WORD3, 300), fit_cuts(TUNED, 300)
    p3 = apply_cutpoints(blend(zt, WORD3), cuts3)
    p4 = apply_cutpoints(blend(zt, TUNED), cuts4)
    n = len(yte)
    b3, b4 = [], []
    for _ in range(400):
        idx = rng.integers(0, n, n)
        b3.append(official_qwk(yte[idx], p3[idx]) * 100)
        b4.append(official_qwk(yte[idx], p4[idx]) * 100)
    b3, b4 = np.array(b3), np.array(b4)
    print(f"   wordonly3  mean {b3.mean():.3f}  sd {b3.std():.3f}  "
          f"95% CI [{np.percentile(b3,2.5):.2f}, {np.percentile(b3,97.5):.2f}]")
    print(f"   all4 tuned mean {b4.mean():.3f}  sd {b4.std():.3f}  "
          f"95% CI [{np.percentile(b4,2.5):.2f}, {np.percentile(b4,97.5):.2f}]")

    print("\n" + "=" * 76)
    print("D) PAIRED BOOTSTRAP ON THE DELTA (all4_tuned - wordonly3), same resamples")
    d = b4 - b3
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    print(f"   point estimate {p4.mean()*0 + (official_qwk(yte,p4)-official_qwk(yte,p3))*100:+.3f}")
    print(f"   bootstrap mean {d.mean():+.3f}  sd {d.std():.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"   P(delta > 0) = {(d > 0).mean()*100:.1f}%")
    print(f"   VERDICT: {'CI excludes 0, real effect' if lo > 0 else 'CI includes 0, NOT distinguishable from noise'}")

    print("\n" + "=" * 76)
    print("C) FOLD-SEED NOISE in the honest val nested-OOF, and the honest marbert effect")
    for seed in (20260714, 1, 7, 42, 2026):
        f = fold_of_rows(build_document_fold_map(yva, val[GROUP].to_list(), n_splits=5,
                                                 seed=seed), val[GROUP].to_list())
        q3 = nested_oof_qwk(blend(zv, WORD3), yva, f, n_candidates=300)[0] * 100
        q4 = nested_oof_qwk(blend(zv, TUNED), yva, f, n_candidates=300)[0] * 100
        print(f"   seed {seed:<9} wordonly3 {q3:.3f}   all4tuned {q4:.3f}   delta {q4-q3:+.3f}")


if __name__ == "__main__":
    main()
