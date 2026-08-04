"""OptimizedRounder unit tests: monotonicity, clipping, QWK improvement over
naive rounding, and a valid nested-OOF number."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import (  # noqa: E402
    OptimizedRounder,
    apply_cutpoints,
    nested_oof_qwk,
    quantile_cutpoints,
)


def _synthetic(n=4000, seed=0):
    """True ordinal levels 1..19 with a distorted, noisy continuous score, so
    naive round(score) is misaligned and a tuned rounder should recover QWK."""
    rng = np.random.default_rng(seed)
    y = rng.integers(1, 20, size=n)
    # score = a squashed, shifted, noisy transform of the true level
    score = 2.0 + 0.7 * y + 0.03 * (y - 10) ** 2 + rng.normal(0, 1.2, size=n)
    return y, score


def test_quantile_seed_is_monotone():
    y = np.array([1, 1, 5, 10, 10, 12, 12, 14, 19])
    cut = quantile_cutpoints(y)
    assert len(cut) == 18
    assert np.all(np.diff(cut) > 0), "cutpoints must be strictly increasing"


def test_apply_clips_to_1_19():
    cut = np.arange(1.5, 19.5, 1.0)  # 18 cutpoints
    assert len(cut) == 18
    out = apply_cutpoints([-100.0, 100.0, 5.2], cut)
    assert out.min() >= 1 and out.max() <= 19
    assert out[0] == 1 and out[1] == 19


def test_rounder_beats_naive():
    y, score = _synthetic()
    naive = np.clip(np.rint(score), 1, 19).astype(int)
    naive_qwk = fold_qwk(y, naive)
    r = OptimizedRounder(n_candidates=120).fit(score, y)
    tuned_qwk = fold_qwk(y, r.predict(score))
    assert np.all(np.diff(r.cutpoints_) > 0), "fitted cutpoints not monotone"
    assert tuned_qwk >= naive_qwk - 1e-9, f"tuned {tuned_qwk} < naive {naive_qwk}"
    # on this misaligned score the tuned rounder should be a clear improvement
    assert tuned_qwk > naive_qwk + 0.01, f"expected real gain, got {tuned_qwk-naive_qwk:.4f}"
    print(f"naive={naive_qwk:.4f} tuned={tuned_qwk:.4f} gain=+{tuned_qwk-naive_qwk:.4f}")


def test_nested_oof_is_valid():
    y, score = _synthetic(n=3000, seed=1)
    fold = np.random.default_rng(2).integers(0, 5, size=len(y))
    q, preds = nested_oof_qwk(score, y, fold, n_candidates=80)
    assert preds.min() >= 1 and preds.max() <= 19
    assert -1.0 <= q <= 1.0
    # nested (honest) QWK should be a touch below in-sample tuned QWK, not above
    in_sample = fold_qwk(y, OptimizedRounder(n_candidates=80).fit(score, y).predict(score))
    assert q <= in_sample + 1e-6, f"nested {q} implausibly above in-sample {in_sample}"
    print(f"nested_oof_qwk={q:.4f} in_sample={in_sample:.4f}")


if __name__ == "__main__":
    test_quantile_seed_is_monotone()
    test_apply_clips_to_1_19()
    test_rounder_beats_naive()
    test_nested_oof_is_valid()
    print("ROUNDER TESTS PASSED")
