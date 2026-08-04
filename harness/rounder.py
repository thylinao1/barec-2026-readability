"""OptimizedRounder: map a continuous score to an integer level 1..19 via 18
monotone cutpoints t1 < ... < t18, chosen by coordinate ascent to maximize QWK.

Primary search is monotone coordinate ascent (Nelder-Mead can return crossed,
non-monotone thresholds in 18 dims and is only a fallback cross-check). Seeded
from equal-frequency quantiles of the label distribution. Output clipped to
[1,19]. Tuned on grouped-OOF continuous predictions ONLY; never on train or dev.

nested_oof_qwk is the HONEST go/no-go number: cutpoints for each fold are fit on
the OTHER folds' OOF scores, so the QWK is never measured on the same rows the
thresholds were tuned on (out-of-fold-of-OOF).
"""
from __future__ import annotations

import numpy as np

from .metrics import fold_qwk

N_LEVELS = 19
N_CUTS = N_LEVELS - 1  # 18


def _enforce_monotone(c, eps: float = 1e-6) -> np.ndarray:
    c = np.array(c, dtype=float)
    for i in range(1, len(c)):
        if c[i] <= c[i - 1]:
            c[i] = c[i - 1] + eps
    return c


def quantile_cutpoints(y_levels) -> np.ndarray:
    """Seed 18 cutpoints from equal-frequency quantiles of the label dist."""
    y = np.asarray([int(v) for v in y_levels], dtype=float)
    qs = np.quantile(y, [k / N_LEVELS for k in range(1, N_LEVELS)])
    return _enforce_monotone(qs)


def apply_cutpoints(scores, cutpoints) -> np.ndarray:
    """level = 1 + (# cutpoints strictly below the score), clipped to [1,19]."""
    scores = np.asarray(scores, dtype=float)
    cutpoints = np.asarray(cutpoints, dtype=float)
    lvl = 1 + np.searchsorted(cutpoints, scores, side="right")
    return np.clip(lvl, 1, N_LEVELS).astype(int)


class OptimizedRounder:
    def __init__(self, n_candidates: int = 200, max_rounds: int = 20):
        self.n_candidates = n_candidates
        self.max_rounds = max_rounds
        self.cutpoints_ = None
        self.oof_qwk_ = None

    def fit(self, scores, y_true, init=None):
        scores = np.asarray(scores, dtype=float)
        y_true = np.asarray([int(v) for v in y_true])
        lo, hi = float(scores.min()), float(scores.max())
        if hi <= lo:
            hi = lo + 1.0
        cut = quantile_cutpoints(y_true) if init is None else np.asarray(init, float)
        cut = _enforce_monotone(np.clip(cut, lo, hi))
        best = fold_qwk(y_true, apply_cutpoints(scores, cut))
        grid = np.linspace(lo, hi, self.n_candidates)
        for _ in range(self.max_rounds):
            improved = False
            for j in range(len(cut)):
                low = cut[j - 1] + 1e-6 if j > 0 else lo
                high = cut[j + 1] - 1e-6 if j < len(cut) - 1 else hi
                cands = grid[(grid > low) & (grid < high)]
                for c in cands:
                    trial = cut.copy()
                    trial[j] = c
                    q = fold_qwk(y_true, apply_cutpoints(scores, trial))
                    if q > best + 1e-9:
                        best, cut, improved = q, trial, True
            if not improved:
                break
        self.cutpoints_ = cut
        self.oof_qwk_ = best
        return self

    def predict(self, scores) -> np.ndarray:
        if self.cutpoints_ is None:
            raise RuntimeError("OptimizedRounder is not fit")
        return apply_cutpoints(scores, self.cutpoints_)


def nested_oof_qwk(scores, y_true, fold_idx, n_candidates: int = 200):
    """Honest rounder QWK: for each fold f, fit cutpoints on all OTHER folds'
    OOF scores and predict fold f. Returns (qwk, preds). This is THE go/no-go
    number for the rounder (thresholds never see the rows they score)."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray([int(v) for v in y_true])
    fold_idx = np.asarray(fold_idx, dtype=int)
    preds = np.empty_like(y_true)
    for f in np.unique(fold_idx):
        tr, te = fold_idx != f, fold_idx == f
        r = OptimizedRounder(n_candidates=n_candidates).fit(scores[tr], y_true[tr])
        preds[te] = r.predict(scores[te])
    return fold_qwk(y_true, preds), preds
