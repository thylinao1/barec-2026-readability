"""e1: fitted linear weights on the FLEET14 members (NNLS / shrunk NNLS / OLS).

Usage: python scripts/endgame/e1_fitted_weights.py <config>
Configs: nnls | shrunk25 | shrunk50 | shrunk75 | ols | ols_clip

Honest number: C.honest_val_oof_weighted (weights fitted per-fold on the other
folds only -> OOF blend -> nested rounder). Oracle: C.test_oracle with the
DEPLOYMENT weights (fitted on FULL val). Nothing is ever fitted on test/blind.

Centering note: fit_fns center X and y internally, so the fit is invariant to
the target mean; blend() normalizes weights by their sum, so only direction
matters. Reported weights are normalized to mean 1.0.
"""
import sys
import time

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, "scripts/endgame")
import common as C  # noqa: E402


def _center(X, y):
    return X - X.mean(axis=0), np.asarray(y, float) - np.mean(y)


def fit_nnls(X, y):
    Xc, yc = _center(X, y)
    w, _ = nnls(Xc, yc)
    return w


def make_shrunk_nnls(g):
    def fit(X, y):
        w = fit_nnls(X, y)
        m = w.mean()
        wn = w / m if m > 0 else np.ones(len(w))
        return (1.0 - g) * wn + g

    fit.__name__ = f"shrunk_nnls_g{g}"
    return fit


def fit_ols(X, y):
    Xc, yc = _center(X, y)
    w, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
    return w


def fit_ols_clip(X, y):
    return np.clip(fit_ols(X, y), 0.0, None)


CONFIGS = {
    "nnls": fit_nnls,
    "shrunk25": make_shrunk_nnls(0.25),
    "shrunk50": make_shrunk_nnls(0.50),
    "shrunk75": make_shrunk_nnls(0.75),
    "ols": fit_ols,
    "ols_clip": fit_ols_clip,
}


def main(name):
    fit = CONFIGS[name]
    t0 = time.time()
    yva, fold, _, _ = C.data()
    X = np.column_stack([C.member_z(m, "val") for m in C.FLEET14])
    print(f"[{name}] data loaded X={X.shape}  t={time.time()-t0:.0f}s", flush=True)

    (hm, hs), _oof, per_fold = C.honest_val_oof_weighted(C.FLEET14, fit)
    print(f"[{name}] HONEST {hm:.3f} +- {hs:.3f}  t={time.time()-t0:.0f}s", flush=True)

    np.set_printoptions(precision=3, suppress=True, linewidth=250)
    pf = np.array([w / (w.mean() if w.mean() else 1.0) for w in per_fold])
    print(f"[{name}] per-fold weights (normalized to mean 1):")
    for i, w in enumerate(pf):
        print(f"  fold{i}: {w}")
    print(f"[{name}] across-fold SD of normalized weights: {pf.std(axis=0)}")
    print(f"[{name}] max across-fold SD: {pf.std(axis=0).max():.3f}")

    w_full = np.asarray(fit(X, yva), float)
    w_dep = w_full / w_full.mean() if w_full.mean() else w_full.copy()
    print(f"[{name}] full-val deployment weights (mean 1): {w_dep}")
    orc = C.test_oracle(C.FLEET14, w_full)
    print(f"[{name}] ORACLE {orc:.3f}  t={time.time()-t0:.0f}s", flush=True)
    print("RESULT\t%s\t%.3f\t%.3f\t%.3f\t%s"
          % (name, hm, hs, orc, ",".join(f"{v:.6f}" for v in w_dep)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
