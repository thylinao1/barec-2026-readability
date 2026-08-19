"""e3: expanded member pool under NNLS.

Pool: from members_with_test() keep the 4 champion members, 'marbert', every
avg-* member, and singles whose family (name minus -sNN suffix) has no avg-*
member. Per-seed singles of families that already have an avg are dropped.

Configs:
  1. NNLS on the pool.
  2. Shrunk NNLS, g in {0.25, 0.5} (toward equal weight over the pool).
  3. NNLS restricted to the nonzero-support of config 1's full-val fit, refit.

Honest numbers via C.honest_val_oof_weighted (weights fitted per fold, OOF
blend, nested rounder). Oracles via C.test_oracle with FULL-VAL fitted weights
(deployment mirror; test never fitted on).
"""
import json
import re
import sys
import time

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, "scripts/endgame")
import common as C

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ---------------- pool construction ----------------
ALL = C.members_with_test()
SPECIAL_KEEP = {"wordCE", "camelbert", "qarib", "reg_myd3", "marbert"}
avgs = [m for m in ALL if m.startswith("avg-")]
avg_stems = [a[len("avg-"):] for a in avgs]


def family(name):
    return re.sub(r"-s\d+$", "", name)


def has_avg(fam):
    return any(fam == s or fam.startswith(s + "-") for s in avg_stems)


pool = []
for m in ALL:
    if m in SPECIAL_KEEP or m.startswith("avg-"):
        pool.append(m)
    elif not has_avg(family(m)):
        pool.append(m)

log(f"POOL ({len(pool)} members, canonical members_with_test order):")
for m in pool:
    print("   ", m, flush=True)

K = len(pool)
yva, fold, ytr, yte = C.data()
Xv = np.column_stack([C.member_z(m, "val") for m in pool])


# ---------------- fitters ----------------
def fit_nnls(X, y):
    Xc = X - X.mean(axis=0)
    yc = y.astype(float) - y.mean()
    w, _ = nnls(Xc, yc)
    if w.sum() <= 0:
        w = np.ones(X.shape[1])
    return w


def make_shrunk(g):
    def fit(X, y):
        w = fit_nnls(X, y)
        w = w / w.mean()  # normalize to mean 1 over the pool
        return (1 - g) * w + g * np.ones_like(w)
    return fit


def norm1(w):
    w = np.asarray(w, float)
    return w / w.mean()


def report(name, members, honest, oracle, w_full):
    nz = [(m, round(float(wi), 4)) for m, wi in zip(members, norm1(w_full)) if wi > 1e-10]
    log(f"RESULT {name}: honest {honest[0]:.3f} +- {honest[1]:.3f} | oracle {oracle:.3f} "
        f"| nonzero {len(nz)}/{len(members)}")
    print("    weights(mean-1 norm):", nz, flush=True)


results = {}

# ---------------- config 1: NNLS on pool ----------------
log("config 1: NNLS on pool - honest (nested OOF weight fitting) ...")
(h_m, h_s), oof, pf = C.honest_val_oof_weighted(pool, fit_nnls)
nz_counts = [int((np.asarray(w) > 1e-10).sum()) for w in pf]
log(f"  per-fold nonzero counts: {nz_counts}")
w1_full = fit_nnls(Xv, yva)
orc1 = C.test_oracle(pool, w1_full)
report("nnls_pool", pool, (h_m, h_s), orc1, w1_full)
results["nnls_pool"] = dict(honest=(h_m, h_s), oracle=orc1, w=norm1(w1_full).tolist())

# ---------------- config 2: shrunk NNLS ----------------
for g in (0.25, 0.5):
    log(f"config 2: shrunk NNLS g={g} - honest ...")
    (h_m, h_s), _, _ = C.honest_val_oof_weighted(pool, make_shrunk(g))
    wg_full = make_shrunk(g)(Xv, yva)
    orc = C.test_oracle(pool, wg_full)
    report(f"nnls_shrunk_g{g}", pool, (h_m, h_s), orc, wg_full)
    results[f"nnls_shrunk_g{g}"] = dict(honest=(h_m, h_s), oracle=orc,
                                        w=norm1(wg_full).tolist())

# ---------------- config 3: restricted refit ----------------
support = [m for m, wi in zip(pool, w1_full) if wi > 1e-10]
log(f"config 3: NNLS restricted to config-1 full-val support ({len(support)} members): {support}")
(h_m, h_s), _, pf3 = C.honest_val_oof_weighted(support, fit_nnls)
Xr = np.column_stack([C.member_z(m, "val") for m in support])
w3_full = fit_nnls(Xr, yva)
orc3 = C.test_oracle(support, w3_full)
report("nnls_restricted", support, (h_m, h_s), orc3, w3_full)
results["nnls_restricted"] = dict(honest=(h_m, h_s), oracle=orc3,
                                  members=support, w=norm1(w3_full).tolist())

out = C.ROOT / "artifacts" / "endgame" / "e3_results.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(dict(pool=pool, results=results), f, indent=1)
log(f"saved {out}")
log("DONE")
