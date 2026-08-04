"""Evaluate the train+test-retrained (tt) members: solo val quality per seat,
then fleet14 with tt seats swapped in (full swap + each seat alone).

Honest val nested-OOF remains valid: no tt member ever saw validation.
The test oracle is DEAD for tt members (test is in their training data).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "endgame"))
import common as C  # noqa: E402
from common import nested_oof_qwk  # noqa: E402

F = ROOT / "artifacts" / "scores" / "fleet"
yva, fold, _, _ = C.data()

SWAPS = {  # old seat -> new tt member
    "avg-alarge02-reg-word": "avg-alarge02-reg-word-tt",
    "avg-marbert-ce-raw": "avg-marbert-ce-raw-tt",
    "avg-arabertv2-reg-d3tok": "avg-arabertv2-reg-d3tok-tt",
    "avg-arabertv02-ce-word": "avg-arabertv02-ce-word-tt",
    "xlmrL-reg-raw-s42": "xlmrL-reg-raw-tt-s42",
}


def z(p):
    a = np.load(p).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def val_z(name):
    return z(F / f"{name}_val.npy") if (F / f"{name}_val.npy").exists() \
        else C.member_z(name, "val")


def honest(members):
    sv = sum(val_z(m) for m in members) / len(members)
    qs = np.array([nested_oof_qwk(sv, yva, fold, n_candidates=g)[0] * 100
                   for g in C.GRIDS])
    return qs.mean(), qs.std()


def solo_val(name):
    q = nested_oof_qwk(val_z(name), yva, fold, n_candidates=300)[0] * 100
    return q


def main():
    print("SOLO val nested-rounder QWK, old seat vs tt seat:")
    for old, new in SWAPS.items():
        print(f"  {old:28s} {solo_val(old):6.3f}  ->  {new:30s} {solo_val(new):6.3f}")

    print("\nBLEND honest val nested-OOF (fleet14 baseline 85.281 +-0.051):")
    full = [SWAPS.get(m, m) for m in C.FLEET14]
    mu, sd = honest(full)
    print(f"  fleet14tt (all 5 seats)        {mu:7.3f} +-{sd:.3f}  delta {mu - 85.281:+.3f}")
    for old, new in SWAPS.items():
        members = [new if m == old else m for m in C.FLEET14]
        mu, sd = honest(members)
        print(f"  swap {old:28s} {mu:7.3f} +-{sd:.3f}  delta {mu - 85.281:+.3f}")


if __name__ == "__main__":
    main()
