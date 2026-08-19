"""Resolve WHICH standardization protocol produced the run log 4b/4c table.

blend4_validate.py reproduced corr(marbert, blend) = 0.9450 exactly (so member loading is
right) but got wordonly3 = 85.28 vs the documented 85.07, and "+marbert equal" = 85.28 vs the
documented 85.48. The likely fork is standardization:

  SELF : z(test) uses TEST's own mean/std      <- what blind_distmatch.py does (the 84.5 champion)
  VAL  : z(test) uses VAL's mean/std           <- what marbert_local_infer.py's docstring claims

This matters because the rounder cutpoints are fit in val-z space and then applied to test-z
space; if the two splits are centred differently, the two protocols give different QWK. The
whole P0 case rests on "+0.41 from marbert", so we need to know which protocol that came from
and whether the gain survives in the protocol the CHAMPION actually uses.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blend4_forktest.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.cv import build_document_fold_map, fold_of_rows  # noqa: E402
from harness.data import GROUP, TARGET, load_2026  # noqa: E402
from harness.metrics import official_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints, nested_oof_qwk  # noqa: E402

S = ROOT / "artifacts" / "scores"
STEMS = {"wordCE": "arabertv02-word-CE", "camelbert": "camelbert-word-CE",
         "qarib8": "pub_AymanTarig__qarib-barec-optimized-v8", "marbert": "marbertv2-reg"}
TUNED = {"wordCE": 0.306, "camelbert": 0.231, "qarib8": 0.231, "marbert": 0.231}
WORD3 = {"wordCE": 1.0, "camelbert": 1.0, "qarib8": 1.0}
ALL4 = {**WORD3, "marbert": 1.0}


def raw(split):
    return {m: np.load(S / f"{stem}_{split}.npy").ravel().astype(float)
            for m, stem in STEMS.items()}


def doc_shrink(sc, docs, a):
    if not a:
        return sc
    out = np.asarray(sc, float).copy()
    d = np.asarray([str(x) for x in docs])
    mu = np.empty_like(out)
    for u in np.unique(d):
        m = d == u
        mu[m] = out[m].mean()
    return (1 - a) * out + a * mu


def main():
    val, test = load_2026("validation"), load_2026("test")
    yva = np.array(val[TARGET].to_list(), int)
    yte = np.array(test[TARGET].to_list(), int)
    rv, rt = raw("validation"), raw("test")

    def std_self(a):
        return (a - a.mean()) / (a.std() or 1)

    def build(protocol, weights, split_raw, ref_raw):
        """protocol SELF: standardize by own stats. VAL: standardize by ref (val) stats."""
        tot = sum(weights.values())
        acc = 0.0
        for m, w in weights.items():
            a = split_raw[m]
            if protocol == "SELF":
                zz = std_self(a)
            else:
                mu, sd = ref_raw[m].mean(), (ref_raw[m].std() or 1)
                zz = (a - mu) / sd
            acc = acc + w * zz
        return acc / tot

    def ev(protocol, weights, alpha=0.0, n_cand=300):
        sv = build(protocol, weights, rv, rv)
        st = build(protocol, weights, rt, rv)
        if alpha:
            sv = doc_shrink(sv, val[GROUP].to_list(), alpha)
            st = doc_shrink(st, test[GROUP].to_list(), alpha)
        r = OptimizedRounder(n_candidates=n_cand).fit(sv, yva)
        return official_qwk(yte, apply_cutpoints(st, r.cutpoints_)) * 100

    print("=" * 78)
    print("SOLO test QWK          run log: wordCE 84.52 camelbert 83.34 qarib8 82.66 marbert 81.96")
    print(f"{'member':12s}{'SELF':>10s}{'VAL':>10s}")
    for m in STEMS:
        print(f"{m:12s}{ev('SELF', {m: 1.0}):10.2f}{ev('VAL', {m: 1.0}):10.2f}")

    print("\n" + "=" * 78)
    print("BLENDS                       run log SELF? VAL?")
    rows = [("wordonly3 equal      [85.07]", WORD3, 0.0),
            ("+marbert equal       [85.48]", ALL4, 0.0),
            ("+marbert tuned       [85.59]", TUNED, 0.0),
            ("+marbert tuned a=.1  [85.69]", TUNED, 0.1)]
    for label, w, a in rows:
        print(f"{label:30s}{ev('SELF', w, a):10.4f}{ev('VAL', w, a):10.4f}")

    print("\n" + "=" * 78)
    print("Is '+marbert equal == wordonly3' a real tie or a bug? (4 dp, SELF protocol)")
    print(f"   wordonly3      {ev('SELF', WORD3):.4f}")
    print(f"   +marbert equal {ev('SELF', ALL4):.4f}")
    print(f"   n_candidates sensitivity (SELF, ALL4): "
          f"200->{ev('SELF', ALL4, n_cand=200):.4f}  300->{ev('SELF', ALL4, n_cand=300):.4f}  "
          f"400->{ev('SELF', ALL4, n_cand=400):.4f}")

    print("\n" + "=" * 78)
    print("HONEST val nested-OOF (rounder never scores rows it was tuned on)")
    vfold = fold_of_rows(build_document_fold_map(yva, val[GROUP].to_list(), n_splits=5,
                                                seed=20260714), val[GROUP].to_list())
    for label, w in (("wordonly3", WORD3), ("all4 equal", ALL4), ("all4 tuned", TUNED)):
        sv = build("SELF", w, rv, rv)
        q = nested_oof_qwk(sv, yva, vfold, n_candidates=300)[0] * 100
        print(f"   {label:12s} val nested-OOF {q:.3f}")

    print("\n" + "=" * 78)
    print("Marbert weight sweep (SELF, wordCE .306 / camelbert .231 / qarib8 .231 fixed)")
    for wm in (0.0, 0.05, 0.10, 0.15, 0.231, 0.30, 0.40):
        w = {"wordCE": 0.306, "camelbert": 0.231, "qarib8": 0.231, "marbert": wm}
        print(f"   marbert w={wm:<6} a=0.0 {ev('SELF', w):7.3f}   a=0.1 {ev('SELF', w, 0.1):7.3f}")


if __name__ == "__main__":
    main()
