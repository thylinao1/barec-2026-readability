"""Can the D3Tok members be REVIVED for blind by standardizing them consistently?

Run log 4d excluded arabertv2-d3tok-reg and -d3tok-CE from the blind blend because our locally
computed D3Tok only matches the license-gated LDC gold at ~76%, which dropped the `reg` member
from 84.68 to 82.75 on blind.

But that comparison mixed preprocessing regimes: the member was STANDARDIZED using val scores
computed from GOLD D3Tok while being APPLIED to blind scores computed from MY D3Tok. The blend
z-scores each member, so a systematic shift between the two regimes is exactly the kind of
error that standardization is supposed to absorb, and cannot, if the two arrays come from
different preprocessing.

We already have the fix cached and unused: `blind/reg_valmyd3.npy` and `blind/dtCE_valmyd3.npy`
are the VAL scores under MY D3Tok. Pairing those with the my-D3Tok blind scores makes each
member internally consistent, at zero new compute.

Honest criterion: val nested-OOF (the rounder never scores rows it was tuned on). We CANNOT use
test here (no my-D3Tok test scores exist), so val nested-OOF is the only honest number
available, and the run log 4b figure of 85.85 for this direction is NOT comparable (it used
gold D3Tok).

Noise axis: n_candidates, NOT the fold seed. Measured 2026-07-25: two different fold seeds gave
bit-identical nested-OOF (84.364 / 84.562), because the rounder snaps cutpoints to a fixed grid
of `n_candidates` values and any 80% subset of val selects the same grid points. Varying the
grid resolution IS a live noise axis (it moved the 4-member test QWK by sd 0.094, range 0.345),
so that is what we vary here.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_myd3_members.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.cv import build_document_fold_map, fold_of_rows  # noqa: E402
from harness.data import GROUP, TARGET, load_2026  # noqa: E402
from harness.rounder import nested_oof_qwk  # noqa: E402

S = ROOT / "artifacts" / "scores"
B = S / "blind"
FOLD_SEED = 20260714          # canonical, frozen; varying it is a no-op (see docstring)
GRIDS = (200, 300, 400)       # the live noise axis

# val score file per member. The two D3Tok members use the MY-D3TOK val scores so that
# val and blind come from the same preprocessing regime.
VAL = {
    "wordCE": S / "arabertv02-word-CE_validation.npy",
    "camelbert": S / "camelbert-word-CE_validation.npy",
    "qarib": S / "pub_AymanTarig__qarib-barec-optimized-v8_validation.npy",
    "marbert": S / "marbertv2-reg_validation.npy",
    "reg_myd3": B / "reg_valmyd3.npy",
    "dtCE_myd3": B / "dtCE_valmyd3.npy",
    # gold-D3Tok versions, for contrast only; NOT blind-safe
    "reg_gold": S / "arabertv2-d3tok-reg_validation.npy",
    "dtCE_gold": S / "arabertv2-d3tok-CE_validation.npy",
}
BLINDF = {"wordCE": "wordCE_blind.npy", "camelbert": "camelbert_blind.npy",
          "qarib": "qarib_blind.npy", "marbert": "marbert_blind.npy",
          "reg_myd3": "reg_blind.npy", "dtCE_myd3": "dtCE_blind.npy"}

COMBOS = [
    ("wordonly3 (champion members)", ["wordCE", "camelbert", "qarib"]),
    ("+ marbert", ["wordCE", "camelbert", "qarib", "marbert"]),
    ("+ reg_myd3", ["wordCE", "camelbert", "qarib", "reg_myd3"]),
    ("+ dtCE_myd3", ["wordCE", "camelbert", "qarib", "dtCE_myd3"]),
    ("+ reg_myd3 + dtCE_myd3", ["wordCE", "camelbert", "qarib", "reg_myd3", "dtCE_myd3"]),
    ("+ marbert + reg_myd3", ["wordCE", "camelbert", "qarib", "marbert", "reg_myd3"]),
    ("+ marbert + reg_myd3 + dtCE_myd3",
     ["wordCE", "camelbert", "qarib", "marbert", "reg_myd3", "dtCE_myd3"]),
    ("[contrast, NOT blind-safe] + reg_gold + dtCE_gold",
     ["wordCE", "camelbert", "qarib", "reg_gold", "dtCE_gold"]),
]


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def main():
    val = load_2026("validation")
    yva = np.array(val[TARGET].to_list(), int)
    groups = val[GROUP].to_list()
    zv = {}
    for m, p in VAL.items():
        if not p.exists():
            print(f"   (skip {m}: {p.name} not found)")
            continue
        a = np.load(p).ravel()
        if len(a) != len(yva):
            print(f"   (skip {m}: length {len(a)} != val {len(yva)})")
            continue
        zv[m] = z(a)

    print(f"loaded members: {sorted(zv)}\n")
    print("-- pairwise correlation on val (diversity check)")
    ms = [m for m in VAL if m in zv]
    print("             " + "".join(f"{m[:9]:>11s}" for m in ms))
    for a in ms:
        print(f"  {a:11s}" + "".join(f"{np.corrcoef(zv[a], zv[b])[0,1]:11.3f}" for b in ms))

    fold = fold_of_rows(build_document_fold_map(yva, groups, n_splits=5, seed=FOLD_SEED), groups)

    print(f"\n-- honest val nested-OOF, mean over n_candidates in {GRIDS}")
    print(f"   {'combo':46s}{'mean':>8s}{'sd':>7s}   per-grid")
    base = None
    results = []
    for label, members in COMBOS:
        if any(m not in zv for m in members):
            print(f"   {label:46s}  SKIPPED (missing member)")
            continue
        sv = sum(zv[m] for m in members) / len(members)
        qs = np.array([nested_oof_qwk(sv, yva, fold, n_candidates=g)[0] * 100 for g in GRIDS])
        if base is None:
            base = qs.mean()
        results.append((label, qs.mean(), qs.std()))
        print(f"   {label:46s}{qs.mean():8.3f}{qs.std():7.3f}   "
              f"{' '.join(f'{q:.2f}' for q in qs)}  delta {qs.mean() - base:+.3f}")

    print("\n-- ranked by honest mean val nested-OOF")
    for label, mu, sd in sorted(results, key=lambda r: -r[1]):
        print(f"   {mu:7.3f} +-{sd:.3f}  {label}")

    print(f"\nNOTE: the champion baseline is the first row. A gain matters only if it clearly "
          f"exceeds the grid sd. Anything within ~1 sd is indistinguishable from noise and must "
          f"NOT go to the Strict track, where a tie costs the #1 spot (our 84.5 is timestamped "
          f"07-24 07:38, ahead of the tied entry's 13:35; a new upload resets that).")


if __name__ == "__main__":
    main()
