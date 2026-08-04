"""Phase 2 - confidence-weighted ensemble of the public checkpoint members.

CONTAMINATION NOTE (the harness caught this): the public checkpoints were trained
on the BAREC corpus, which is exactly the 2026 TRAIN split, so their train
predictions are in-sample (solo train QWK ~89-95 vs ~82-84 honest). TRAIN IS
UNUSABLE for tuning the checkpoint blend/rounder. The checkpoints treat
validation and test as clean holdouts, so:
  - TUNE everything (member selection, weights, rounder, corrections) on VAL,
    using grouped-OOF within val (by Document) for honesty.
  - EVALUATE once on TEST (held out, never tuned on). Test QWK = the honest
    number and the expected open-test board score.

Every keep/drop decision is gated on val grouped-OOF QWK. Test is touched once.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/phase2_ensemble.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.cv import build_document_fold_map, fold_of_rows  # noqa: E402
from harness.data import GROUP, ID, TARGET, load_2026  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints, nested_oof_qwk  # noqa: E402
from harness.submission import write_codabench_zip  # noqa: E402

SCORES = ROOT / "artifacts" / "scores"
SUBS = ROOT / "submissions" / "phase2"
MEMBERS = ["arabertv2-d3tok-reg", "arabertv2-d3tok-CE", "arabertv02-word-CE", "camelbert-word-CE"]
NC = 200
SEED = 20260714


def load_members():
    have, va, te = [], {}, {}
    for m in MEMBERS:
        if (SCORES / f"{m}_validation.npy").exists() and (SCORES / f"{m}_test.npy").exists():
            have.append(m)
            va[m] = np.load(SCORES / f"{m}_validation.npy")
            te[m] = np.load(SCORES / f"{m}_test.npy")
    return have, va, te


def standardize(names, va, te):
    stats = {m: (float(va[m].mean()), float(va[m].std()) or 1.0) for m in names}
    Zva = {m: (va[m] - stats[m][0]) / stats[m][1] for m in names}
    Zte = {m: (te[m] - stats[m][0]) / stats[m][1] for m in names}
    return Zva, Zte


def blend(names, Z, w):
    return sum(w[i] * Z[m] for i, m in enumerate(names))


def nested(score, y, fold):
    return nested_oof_qwk(score, y, fold, n_candidates=NC)[0]


def _cap(preds, hi):
    out = preds.copy(); out[out > hi] = hi
    return out


CORRECTIONS = [("cap@18", lambda p: _cap(p, 18)), ("cap@17", lambda p: _cap(p, 17))]


def search_corrections(nested_preds, y):
    kept, preds, base = [], nested_preds.copy(), fold_qwk(y, nested_preds)
    for name, fn in CORRECTIONS:
        q = fold_qwk(y, fn(preds))
        if q > base + 1e-6:
            preds, base, _ = fn(preds), q, kept.append((name, fn))
            print(f"  keep {name}: val nested-OOF -> {q*100:.4f}%")
        else:
            print(f"  drop {name}: {q*100:.4f}% (no lift)")
    return kept, base


def apply_corrections(kept, preds):
    for _, fn in kept:
        preds = fn(preds)
    return preds


def main():
    SUBS.mkdir(parents=True, exist_ok=True)
    names, va, te = load_members()
    if not names:
        print("no members cached - run scripts/phase2_infer.py first"); return
    val, test = load_2026("validation"), load_2026("test")
    yva = np.array(val[TARGET].to_list(), dtype=int)
    yte = np.array(test[TARGET].to_list(), dtype=int)

    # grouped folds WITHIN val (by Document) for honest tuning
    vmap = build_document_fold_map(yva, val[GROUP].to_list(), n_splits=5, seed=SEED)
    vfold = fold_of_rows(vmap, val[GROUP].to_list())
    Zva, Zte = standardize(names, va, te)

    # ---- solo members: honest val nested-OOF + val/test single-rounder ----
    print("=== solo members ===")
    print(f"  {'member':24s} {'val nOOF':>9s} {'test':>8s}")
    solo = {}
    for m in names:
        solo[m] = nested(Zva[m], yva, vfold)
        r = OptimizedRounder(n_candidates=NC).fit(Zva[m], yva)  # fit on val, apply to test
        qte = fold_qwk(yte, apply_cutpoints(Zte[m], r.cutpoints_))
        print(f"  {m:24s} {solo[m]*100:8.4f}% {qte*100:7.4f}%")
    print("pairwise val-score corr:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {a} ~ {b}: {np.corrcoef(va[a], va[b])[0,1]:.3f}")

    # ---- forward selection by val nested-OOF (equal weight) ----
    order = sorted(names, key=lambda m: -solo[m])
    selected, best = [order[0]], solo[order[0]]
    print(f"\n=== forward selection (val nested-OOF, equal weight) ===\n  start {selected} -> {best*100:.4f}%")
    improved = True
    while improved and len(selected) < len(names):
        improved = False
        cb, cm = best, None
        for m in order:
            if m in selected:
                continue
            trial = selected + [m]
            w = np.ones(len(trial)) / len(trial)
            q = nested(blend(trial, Zva, w), yva, vfold)
            if q > cb + 1e-4:
                cb, cm = q, m
        if cm:
            selected.append(cm); best = cb; improved = True
            print(f"  + {cm} -> {best*100:.4f}%")
    print(f"  selected: {selected}")

    # ---- weighting scheme (val nested-OOF) ----
    k = len(selected)
    w_eq = np.ones(k) / k
    rv = np.array([np.var(va[m] - yva) for m in selected]); w_iv = (1 / rv) / (1 / rv).sum()
    A = np.stack([Zva[m] for m in selected], axis=1)
    w_n, _ = nnls(A, yva.astype(float)); w_n = w_n / w_n.sum() if w_n.sum() > 0 else w_eq
    best_nm, best_w, best_q = "equal", w_eq, nested(blend(selected, Zva, w_eq), yva, vfold)
    print("\n=== weighting (val nested-OOF) ===")
    for nm, w in {"equal": w_eq, "inv_var": w_iv, "nnls": w_n}.items():
        q = nested(blend(selected, Zva, w), yva, vfold)
        print(f"  {nm:8s} w={np.round(w,3)} -> {q*100:.4f}%")
        if q > best_q + 1e-4:
            best_nm, best_w, best_q = nm, w, q
    print(f"  chosen: {best_nm}")

    # ---- final rounder on ALL val -> apply to test; corrections on val nested ----
    s_va = blend(selected, Zva, best_w)
    s_te = blend(selected, Zte, best_w)
    rounder = OptimizedRounder(n_candidates=300).fit(s_va, yva)
    val_honest, nested_preds = nested_oof_qwk(s_va, yva, vfold, n_candidates=300)
    te_pred = apply_cutpoints(s_te, rounder.cutpoints_)
    print("\n=== distribution-correction pass (val nested-OOF gated) ===")
    kept, val_corr = search_corrections(nested_preds, yva)
    te_pred = apply_corrections(kept, te_pred)

    print("\n=== PHASE 2 RESULT (honest) ===")
    print(f"  val nested-OOF QWK:            {val_honest*100:.4f}%")
    print(f"  + corrections {[k for k,_ in kept] or 'none'}: {val_corr*100:.4f}%")
    print(f"  TEST QWK (held out, primary):  {fold_qwk(yte, te_pred)*100:.4f}%   <- expected board score")

    np.save(SCORES / "phase2_blend_val.npy", s_va)
    np.save(SCORES / "phase2_blend_test.npy", s_te)
    zp = write_codabench_zip(SUBS, test[ID].to_list(), te_pred)
    print(f"\nwrote {zp}  (NOT uploaded - human gate)")


if __name__ == "__main__":
    main()
