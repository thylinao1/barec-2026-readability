"""e5: fitting-free robust aggregation over FLEET14 z-scores (row-wise).

Configs:
  (a) median
  (b) trimmed mean dropping single min + max          (trim1)
  (c) trimmed mean dropping 2 lowest + 2 highest      (trim2)
  (d) mean of per-member rank transforms (rankdata/n per member,
      then z of the row-mean)                          (rankmean)
  (e) 0.5*mean + 0.5*median                            (meanmed)

No fitting anywhere -> honest_val on the aggregated val scores IS the honest
number. Oracle replicates test_oracle with the aggregator swapped in for the
weighted mean: sv = agg(val z), st = agg(test z),
pred = apply_cutpoints(st, deploy_cuts(sv, st)), fold_qwk vs yte * 100.
Baselines (verified, not recomputed): fleet14 equal-weight mean
honest 85.281 +-0.051, oracle 86.705.
"""
import sys
import time

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "scripts/endgame")
import common as C  # noqa: E402


def zv(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() or 1.0)


def agg_median(X):
    return np.median(X, axis=1)


def agg_trim(X, k):
    Xs = np.sort(X, axis=1)
    return Xs[:, k:X.shape[1] - k].mean(axis=1)


def agg_rankmean(X):
    R = np.column_stack([rankdata(X[:, j]) / X.shape[0] for j in range(X.shape[1])])
    return zv(R.mean(axis=1))


CONFIGS = [
    ("a_median",   agg_median),
    ("b_trim1",    lambda X: agg_trim(X, 1)),
    ("c_trim2",    lambda X: agg_trim(X, 2)),
    ("d_rankmean", agg_rankmean),
    ("e_meanmed",  lambda X: 0.5 * X.mean(axis=1) + 0.5 * np.median(X, axis=1)),
]


def main():
    members = C.FLEET14
    _, _, _, yte = C.data()
    Xv = np.column_stack([C.member_z(m, "val") for m in members])
    Xt = np.column_stack([C.member_z(m, "test") for m in members])
    print(f"val matrix {Xv.shape}, test matrix {Xt.shape}", flush=True)

    for name, fn in CONFIGS:
        t0 = time.time()
        sv = fn(Xv)
        st = fn(Xt)
        hm, hs = C.honest_val(sv)
        pred = C.apply_cutpoints(st, C.deploy_cuts(sv, st))
        orc = float(C.fold_qwk(yte, pred)) * 100
        print(f"{name:12s} honest {hm:.3f} +-{hs:.3f}  oracle {orc:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
