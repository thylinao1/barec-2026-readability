"""e2: direct-QWK coordinate ascent on FLEET14 weights.

fit_fn(X_tr, y_tr): start w = ones(14); objective(w) = fold_qwk(y_tr,
apply_cutpoints(s, cuts)) with s = X_tr @ w / w.sum() and cuts =
np.quantile(s, np.cumsum(freq)[:-1]), freq = empirical label distribution
of y_tr over levels 1..19. Coordinate ascent: for each member (FLEET14
order) try w_i in {0, 0.25, ..., 2.0}, keep the best; up to 3 full passes
or until a pass yields no improvement.

Configs:
  (a) e2_ascent_raw    - raw ascent weights
  (b) e2_ascent_shrunk - w = 0.5 * w/mean(w) + 0.5

Honest via C.honest_val_oof_weighted (per-fold refit); oracle via
C.test_oracle with FULL-val-fitted weights. Nothing fitted on test/blind.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "scripts/endgame")
import common as C  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import apply_cutpoints  # noqa: E402

GRID = np.arange(0.0, 2.0 + 1e-9, 0.25)  # 0, 0.25, ..., 2.0
EPS = 1e-12
MEMBERS = list(C.FLEET14)
K = len(MEMBERS)


def make_objective(X, y):
    freq = np.array([(y == k).mean() for k in range(1, 20)])
    qcum = np.cumsum(freq)[:-1]

    def obj(w):
        ws = w.sum()
        if ws <= 0:
            return -1.0
        s = X @ w / ws
        cuts = np.quantile(s, qcum)
        return fold_qwk(y, apply_cutpoints(s, cuts))

    return obj


def ascent(X, y, tag="", verbose=False):
    obj = make_objective(X, y)
    w = np.ones(K)
    best = obj(w)
    traj = [round(best, 6)]
    for p in range(3):
        improved = False
        for i in range(K):
            cur = w[i]
            best_v, best_q = cur, best
            for v in GRID:
                if v == cur:
                    continue
                w[i] = v
                q = obj(w)
                if q > best_q + EPS:
                    best_q, best_v = q, v
            w[i] = best_v
            if best_q > best + EPS:
                best = best_q
                improved = True
        traj.append(round(best, 6))
        if verbose:
            print(f"    [{tag}] pass {p + 1}: obj={best:.5f} "
                  f"w={np.round(w, 2).tolist()}", flush=True)
        if not improved:
            break
    return w.copy(), best, traj


_fit_cache = {}


def raw_fit_cached(X_tr, y_tr):
    key = (X_tr.shape[0], round(float(X_tr.sum()), 6), int(y_tr.sum()))
    if key not in _fit_cache:
        w, best, traj = ascent(X_tr, y_tr, tag=f"fold n={X_tr.shape[0]}")
        print(f"    fold fit n={X_tr.shape[0]}: obj {traj[0]:.5f} -> "
              f"{best:.5f} traj={traj} w={np.round(w, 2).tolist()}",
              flush=True)
        _fit_cache[key] = w
    return _fit_cache[key]


def shrunk_fit(X_tr, y_tr):
    w = raw_fit_cached(X_tr, y_tr)
    return 0.5 * w / w.mean() + 0.5


def norm1(w):
    w = np.asarray(w, float)
    return w / w.mean()


def main():
    t0 = time.time()
    yva, fold, _, _ = C.data()
    Xva = np.column_stack([C.member_z(m, "val") for m in MEMBERS])
    print(f"val n={len(yva)}, members={K}", flush=True)

    # ---- full-val ascent (deployment weights) ----
    print("[1/5] full-val coordinate ascent ...", flush=True)
    w_full, obj_full, traj_full = ascent(Xva, yva, tag="FULL", verbose=True)
    w_raw = w_full
    w_shr = 0.5 * w_full / w_full.mean() + 0.5
    print(f"  full-val objective trajectory: {traj_full}")
    print(f"  raw deployment weights   : {np.round(w_raw, 4).tolist()}")
    print(f"  raw normalized (mean 1)  : {np.round(norm1(w_raw), 4).tolist()}")
    print(f"  shrunk deployment weights: {np.round(w_shr, 4).tolist()}")
    print(f"  shrunk normalized        : {np.round(norm1(w_shr), 4).tolist()}")
    print(f"  elapsed {time.time() - t0:.0f}s", flush=True)

    # ---- honest (a): raw ascent, per-fold refit ----
    print("[2/5] honest_val_oof_weighted raw ascent ...", flush=True)
    (m_a, s_a), _, pf_a = C.honest_val_oof_weighted(MEMBERS, raw_fit_cached)
    print(f"  e2_ascent_raw honest: {m_a:.3f} +- {s_a:.3f}")
    for i, w in enumerate(pf_a):
        print(f"    fold{i} w={np.round(w, 2).tolist()}")
    print(f"  elapsed {time.time() - t0:.0f}s", flush=True)

    # ---- honest (b): shrunk, per-fold refit (cache hits, no re-ascent) ----
    print("[3/5] honest_val_oof_weighted shrunk ...", flush=True)
    (m_b, s_b), _, pf_b = C.honest_val_oof_weighted(MEMBERS, shrunk_fit)
    print(f"  e2_ascent_shrunk honest: {m_b:.3f} +- {s_b:.3f}")
    for i, w in enumerate(pf_b):
        print(f"    fold{i} w={np.round(w, 3).tolist()}")
    print(f"  elapsed {time.time() - t0:.0f}s", flush=True)

    # ---- oracles with full-val-fitted deployment weights ----
    print("[4/5] test_oracle raw ...", flush=True)
    or_a = C.test_oracle(MEMBERS, w_raw)
    print(f"  e2_ascent_raw oracle: {or_a:.3f}", flush=True)
    print("[5/5] test_oracle shrunk ...", flush=True)
    or_b = C.test_oracle(MEMBERS, w_shr)
    print(f"  e2_ascent_shrunk oracle: {or_b:.3f}", flush=True)

    print("\n===== SUMMARY =====")
    print(f"baseline fleet14 equal : honest 85.281 +- 0.051 | oracle 86.705")
    print(f"e2_ascent_raw          : honest {m_a:.3f} +- {s_a:.3f} | "
          f"oracle {or_a:.3f}")
    print(f"e2_ascent_shrunk       : honest {m_b:.3f} +- {s_b:.3f} | "
          f"oracle {or_b:.3f}")
    print(f"full-val obj trajectory: {traj_full}")
    print(f"raw w (norm mean1)     : {np.round(norm1(w_raw), 6).tolist()}")
    print(f"shrunk w (norm mean1)  : {np.round(norm1(w_shr), 6).tolist()}")
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
