"""e4: per-member isotonic calibration on FLEET14.

Configs measured (honest = manual per-fold nesting of the isotonic fits,
then C.honest_val on the OOF blend; oracle = full-val isotonic fits, exact
deployment mirror via deploy_cuts on test):

  A. e4A_iso_eq   : per-member IsotonicRegression(increasing=True,
                    out_of_bounds='clip') on (member_z, yva), equal average
                    of the 14 calibrated members.
  B. e4B_iso_raw  : 0.5 * z(config-A blend) + 0.5 * z(raw equal z blend).
  C. e4C_iso_nnls : per-member isotonic then NNLS weights on the calibrated
                    members (isotonic AND weights refit per fold for honest).

Nothing is fitted on test or blind anywhere in this file.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def fit_iso(x, y):
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)


yva, fold, ytr, yte = C.data()
M = C.FLEET14
K = len(M)
Xv = np.column_stack([C.member_z(m, "val") for m in M])
Xt = np.column_stack([C.member_z(m, "test") for m in M])
folds = np.unique(fold)
log(f"data: val={len(yva)} test={len(yte)} members={K}")

# ---------- nested per-member isotonic OOF (backbone for A and B) ----------
oof_cal = np.empty_like(Xv)
for f in folds:
    tr, te = fold != f, fold == f
    for j in range(K):
        ir = fit_iso(Xv[tr, j], yva[tr])
        oof_cal[te, j] = ir.predict(Xv[te, j])
    log(f"nested isotonic: fold {f} done")

# ---------- full-val isotonic (deployment mirror for A and B) ----------
cal_v = np.empty_like(Xv)
cal_t = np.empty_like(Xt)
for j in range(K):
    ir = fit_iso(Xv[:, j], yva)
    cal_v[:, j] = ir.predict(Xv[:, j])
    cal_t[:, j] = ir.predict(Xt[:, j])
log("full-val isotonic done")

results = []

# ---------- Config A: equal average of calibrated members ----------
oof_A = oof_cal.mean(axis=1)
hA_mean, hA_sd = C.honest_val(oof_A)
log(f"A e4A_iso_eq   honest {hA_mean:.3f} +- {hA_sd:.3f}")
sv_A = cal_v.mean(axis=1)
st_A = cal_t.mean(axis=1)
oA = float(C.fold_qwk(yte, C.apply_cutpoints(st_A, C.deploy_cuts(sv_A, st_A)))) * 100
log(f"A e4A_iso_eq   oracle {oA:.3f}")
results.append(("e4A_iso_eq", hA_mean, hA_sd, oA))

# ---------- Config B: half raw z blend, half calibrated blend ----------
raw_v = Xv.mean(axis=1)
raw_t = Xt.mean(axis=1)
oof_B = 0.5 * C.z(oof_A) + 0.5 * C.z(raw_v)
hB_mean, hB_sd = C.honest_val(oof_B)
log(f"B e4B_iso_raw  honest {hB_mean:.3f} +- {hB_sd:.3f}")
sv_B = 0.5 * C.z(sv_A) + 0.5 * C.z(raw_v)
st_B = 0.5 * C.z(st_A) + 0.5 * C.z(raw_t)
oB = float(C.fold_qwk(yte, C.apply_cutpoints(st_B, C.deploy_cuts(sv_B, st_B)))) * 100
log(f"B e4B_iso_raw  oracle {oB:.3f}")
results.append(("e4B_iso_raw", hB_mean, hB_sd, oB))

# ---------- Config C: isotonic + NNLS weights, fully nested ----------
oof_C = np.empty(len(yva))
wC_folds = []
for f in folds:
    tr, te = fold != f, fold == f
    cal_in = np.empty((int(tr.sum()), K))
    cal_out = np.empty((int(te.sum()), K))
    for j in range(K):
        ir = fit_iso(Xv[tr, j], yva[tr])
        cal_in[:, j] = ir.predict(Xv[tr, j])
        cal_out[:, j] = ir.predict(Xv[te, j])
    w, _ = nnls(cal_in, yva[tr].astype(float))
    if w.sum() == 0:
        w = np.ones(K)
    wC_folds.append(w / w.sum())
    oof_C[te] = cal_out @ w / w.sum()
    log(f"C nested fold {f}: nnz={int((w > 1e-9).sum())}")
hC_mean, hC_sd = C.honest_val(oof_C)
log(f"C e4C_iso_nnls honest {hC_mean:.3f} +- {hC_sd:.3f}")
w_full, _ = nnls(cal_v, yva.astype(float))
if w_full.sum() == 0:
    w_full = np.ones(K)
w_full_n = w_full / w_full.sum()
sv_C = cal_v @ w_full_n
st_C = cal_t @ w_full_n
oC = float(C.fold_qwk(yte, C.apply_cutpoints(st_C, C.deploy_cuts(sv_C, st_C)))) * 100
log(f"C e4C_iso_nnls oracle {oC:.3f}")
results.append(("e4C_iso_nnls", hC_mean, hC_sd, oC))

# ---------- summary ----------
log("==== SUMMARY (baseline fleet14 equal: honest 85.281 +-0.051, oracle 86.705) ====")
for name, hm, hs, orc in results:
    log(f"{name:14s} honest {hm:.3f} +- {hs:.3f}   oracle {orc:.3f}")
log("full-val NNLS weights on calibrated members (mean-1.0 normalized):")
w_mean1 = w_full_n * K
for m, w in zip(M, w_mean1):
    log(f"  {m:28s} {w:.6f}")
log("per-fold NNLS weight vectors (sum-1 normalized):")
for f, w in zip(folds, wC_folds):
    log(f"  fold {f}: " + " ".join(f"{x:.4f}" for x in w))
