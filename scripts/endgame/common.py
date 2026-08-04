"""Shared infrastructure for the 2026-08-03 endgame experiments.

Two instruments, both board-calibrated:

1. honest_val(blend_val_scores): nested-OOF QWK on validation (document folds,
   seed 20260714, grids 200/300/400). Same number eval_candidates.py produces;
   fleet14 = 85.281 +-0.051 on it, board 85.3.

2. test_oracle(members, weights): full deployment pipeline (per-split self-z,
   weighted blend, rounder fit on VAL, half-match at 0.50 toward the train
   prior using TEST scores) evaluated against test gold. Test is EVALUATION
   ONLY - nothing may ever be fitted on it. This mirrors val->blind deployment
   exactly, so an aggregation scheme that beats equal weight here generalizes
   the same way the blind submission will.

Caveat recorded: reg_myd3's test stand-in is the GOLD-D3Tok test score
(arabertv2-d3tok-reg_test.npy); its blind file is my-D3Tok. Constant across
aggregator comparisons, so A/B deltas are clean; absolute test QWK is ~0.1-0.2
optimistic for that one member.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from harness.cv import build_document_fold_map, fold_of_rows  # noqa: E402
from harness.data import GROUP, TARGET, load_2026  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints, nested_oof_qwk  # noqa: E402
from blind_blend import REG  # noqa: E402  (the blender's member registry: name -> (val, blind))

S = ROOT / "artifacts" / "scores"
F = S / "fleet"
CT = S / "cluster_test"
FOLD_SEED = 20260714
GRIDS = (200, 300, 400)
FRAC = 0.50  # half-match ratio, board-validated optimum. Do not sweep.

FLEET14 = ["wordCE", "camelbert", "qarib", "reg_myd3",
           "avg-arabertv2-reg-d3tok", "avg-arabertv02-ce-word",
           "avg-camelbert-ce-word", "avg-qarib-ce-raw",
           "avg-alarge02-reg-word", "avg-araelectra-reg-word",
           "avg-camelbert-reg-word", "avg-marbert-ce-raw",
           "avg-arabertv2-ce-d3tok", "xlmrL-reg-raw-s42"]

# test-score resolution for the non-fleet members
_TEST_OVERRIDES = {
    "wordCE": S / "arabertv02-word-CE_test.npy",
    "camelbert": S / "camelbert-word-CE_test.npy",
    "qarib": S / "pub_AymanTarig__qarib-barec-optimized-v8_test.npy",
    "reg_myd3": S / "arabertv2-d3tok-reg_test.npy",   # gold-D3Tok stand-in, see docstring
    "dtCE_myd3": S / "arabertv2-d3tok-CE_test.npy",   # gold-D3Tok stand-in
    "marbert": S / "marbertv2-reg_test.npy",
}


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def test_path(name):
    if name in _TEST_OVERRIDES:
        p = _TEST_OVERRIDES[name]
        return p if p.exists() else None
    for p in (F / f"{name}_test.npy", CT / name / f"{name}_test.npy"):
        if p.exists():
            return p
    return None


_cache = {}


def data():
    """(yva, fold, ytr, yte) - cached."""
    if "d" not in _cache:
        val = load_2026("validation")
        yva = np.array(val[TARGET].to_list(), int)
        fold = fold_of_rows(
            build_document_fold_map(yva, val[GROUP].to_list(), n_splits=5, seed=FOLD_SEED),
            val[GROUP].to_list())
        train = load_2026("train")
        ytr = np.array(train[TARGET].to_list(), int)
        test = load_2026("test")
        yte = np.array(test[TARGET].to_list(), int)
        _cache["d"] = (yva, fold, ytr, yte)
    return _cache["d"]


def member_z(name, split):
    """z-scored member scores for split in {'val','blind','test'}."""
    key = (name, split)
    if key not in _cache:
        if split == "test":
            p = test_path(name)
            if p is None:
                raise FileNotFoundError(f"no test scores for member {name}")
        else:
            p = REG[name][0 if split == "val" else 1]
        _cache[key] = z(np.load(p))
    return _cache[key]


def members_with_test():
    return [m for m in REG if test_path(m) is not None
            and REG[m][0].exists() and REG[m][1].exists()]


def blend(members, weights, split):
    w = np.asarray(weights, float)
    return sum(wi * member_z(m, split) for wi, m in zip(w, members)) / w.sum()


def honest_val(blend_val_scores):
    """(mean, sd) over the three rounder grids - THE staging-decision number."""
    yva, fold, _, _ = data()
    qs = np.array([nested_oof_qwk(blend_val_scores, yva, fold, n_candidates=g)[0] * 100
                   for g in GRIDS])
    return qs.mean(), qs.std()


def deploy_cuts(blend_val_scores, blend_target_scores, n_candidates=300):
    """Exact deployment calibration: rounder fit on val, half-matched toward the
    train prior using the target split's scores. Returns sorted cutpoints."""
    yva, _, ytr, _ = data()
    r = OptimizedRounder(n_candidates=n_candidates).fit(blend_val_scores, yva)
    prior = np.array([(ytr == k).mean() for k in range(1, 20)])
    return np.sort((1 - FRAC) * r.cutpoints_
                   + FRAC * np.quantile(blend_target_scores, np.cumsum(prior)[:-1]))


def test_oracle(members, weights):
    """Deployment-mirrored QWK on the test split. Nothing fitted on test."""
    _, _, _, yte = data()
    sv = blend(members, weights, "val")
    st = blend(members, weights, "test")
    pred = apply_cutpoints(st, deploy_cuts(sv, st))
    return float(fold_qwk(yte, pred)) * 100


def honest_val_oof_weighted(members, fit_weights_fn):
    """Nested weight fitting: for each val fold, fit weights on the OTHER folds
    and blend the held-out fold with them -> OOF blend scores that never saw
    their own rows during weight fitting. Then nested rounder QWK on top.
    fit_weights_fn(X_tr [n,k], y_tr) -> weights [k].
    Returns ((mean, sd), oof_scores, per-fold weights)."""
    yva, fold, _, _ = data()
    X = np.column_stack([member_z(m, "val") for m in members])
    oof = np.empty(len(yva), float)
    per_fold = []
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        w = np.asarray(fit_weights_fn(X[tr], yva[tr]), float)
        per_fold.append(w)
        s = X[te] @ w / w.sum() if w.sum() else X[te].mean(axis=1)
        oof[te] = s
    qs = np.array([nested_oof_qwk(oof, yva, fold, n_candidates=g)[0] * 100
                   for g in GRIDS])
    return (qs.mean(), qs.std()), oof, per_fold
