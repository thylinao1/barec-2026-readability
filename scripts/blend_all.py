"""Generalized final blender - the reusable 'push' step.

Auto-discovers every cached member (the 4 CAMeL-Lab checkpoints + any vetted
pub_* public checkpoints + optional classical/kaggle members), tunes on VAL
(grouped-OOF within val, by Document), evaluates once on TEST (= the board, now
proven local==board), forward-selects by val nested-OOF, compares weight schemes,
one rounder, ablated corrections, writes the submission.

Contamination discipline stays: everything tuned on val, test touched once. Since
local==board is confirmed, the reported TEST QWK is the exact board score.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blend_all.py [--include-classical]
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
SUBS = ROOT / "submissions" / "blend_all"
BASE = ["arabertv2-d3tok-reg", "arabertv2-d3tok-CE", "arabertv02-word-CE", "camelbert-word-CE"]
NC = 200
SEED = 20260714


def discover(include_classical: bool):
    """Any member with both <name>_validation.npy and <name>_test.npy: the 4
    checkpoints, pub_* public members, and Kaggle members (e.g. marbertv2-reg).
    Internal artifacts use different suffixes (phase1_val, phase2_blend_val) so
    they are not matched."""
    names = []
    for p in sorted(SCORES.glob("*_validation.npy")):
        m = p.name[: -len("_validation.npy")]
        if (SCORES / f"{m}_test.npy").exists() and m not in names:
            names.append(m)
    # deterministic order: base checkpoints first, then the rest
    names = [m for m in BASE if m in names] + [m for m in names if m not in BASE]
    if include_classical and (SCORES / "phase1_val.npy").exists():
        names.append("__classical__")
    return names


def load(m):
    if m == "__classical__":
        return np.load(SCORES / "phase1_val.npy"), np.load(SCORES / "phase1_test.npy")
    return np.load(SCORES / f"{m}_validation.npy"), np.load(SCORES / f"{m}_test.npy")


def main():
    include_classical = "--include-classical" in sys.argv
    SUBS.mkdir(parents=True, exist_ok=True)
    names = discover(include_classical)
    print(f"members ({len(names)}): {names}")
    val, test = load_2026("validation"), load_2026("test")
    yva = np.array(val[TARGET].to_list(), dtype=int)
    yte = np.array(test[TARGET].to_list(), dtype=int)
    vmap = build_document_fold_map(yva, val[GROUP].to_list(), n_splits=5, seed=SEED)
    vfold = fold_of_rows(vmap, val[GROUP].to_list())

    raw = {m: load(m) for m in names}
    stats = {m: (raw[m][0].mean(), raw[m][0].std() or 1.0) for m in names}
    Zv = {m: (raw[m][0] - stats[m][0]) / stats[m][1] for m in names}
    Zt = {m: (raw[m][1] - stats[m][0]) / stats[m][1] for m in names}

    def bl(sel, w):
        return sum(w[i] * Zv[m] for i, m in enumerate(sel)), sum(w[i] * Zt[m] for i, m in enumerate(sel))

    def nested_v(sel, w):
        sv, _ = bl(sel, w)
        return nested_oof_qwk(sv, yva, vfold, n_candidates=NC)[0]

    solo = {m: nested_v([m], [1.0]) for m in names}
    print("solo val nested-OOF:")
    for m in sorted(names, key=lambda x: -solo[x]):
        print(f"  {m:52s} {solo[m]*100:.3f}")

    order = sorted(names, key=lambda m: -solo[m])
    sel, best = [order[0]], solo[order[0]]
    print(f"\nforward selection: start {sel[0]} -> {best*100:.3f}")
    improved = True
    while improved and len(sel) < len(names):
        improved = False
        cb, cm = best, None
        for m in order:
            if m in sel:
                continue
            w = np.ones(len(sel) + 1) / (len(sel) + 1)
            q = nested_v(sel + [m], w)
            if q > cb + 1e-4:
                cb, cm = q, m
        if cm:
            sel.append(cm); best = cb; improved = True
            print(f"  + {cm} -> {best*100:.3f}")
    print(f"selected ({len(sel)}): {sel}")

    k = len(sel)
    A = np.stack([Zv[m] for m in sel], axis=1)
    w_n, _ = nnls(A, yva.astype(float)); w_n = w_n / w_n.sum() if w_n.sum() > 0 else np.ones(k) / k
    rv = np.array([np.var(raw[m][0] - yva) for m in sel]); w_iv = (1 / rv) / (1 / rv).sum()
    best_w, best_q = np.ones(k) / k, nested_v(sel, np.ones(k) / k)
    print("\nweighting:")
    for nm, w in {"equal": np.ones(k) / k, "inv_var": w_iv, "nnls": w_n}.items():
        q = nested_v(sel, w)
        print(f"  {nm:8s} -> {q*100:.3f}")
        if q > best_q + 1e-4:
            best_w, best_q = w, q

    sv, st = bl(sel, best_w)
    rounder = OptimizedRounder(n_candidates=300).fit(sv, yva)
    val_honest = nested_oof_qwk(sv, yva, vfold, n_candidates=300)[0]
    te_pred = apply_cutpoints(st, rounder.cutpoints_)

    # ablated corrections (val nested gated)
    _, npreds = nested_oof_qwk(sv, yva, vfold, n_candidates=300)
    base = fold_qwk(yva, npreds); kept = []
    for name, hi in [("cap@18", 18), ("cap@17", 17)]:
        c = npreds.copy(); c[c > hi] = hi
        if fold_qwk(yva, c) > base + 1e-6:
            base = fold_qwk(yva, c); npreds = c; kept.append(hi)
    for hi in kept:
        te_pred[te_pred > hi] = hi

    test_qwk = fold_qwk(yte, te_pred)
    print("\n=== BLEND_ALL RESULT ===")
    print(f"  val nested-OOF:  {val_honest*100:.4f}%")
    print(f"  TEST QWK (=board): {test_qwk*100:.4f}%   corrections={kept or 'none'}")
    zp = write_codabench_zip(SUBS, test[ID].to_list(), te_pred)
    print(f"  wrote {zp}")


if __name__ == "__main__":
    main()
