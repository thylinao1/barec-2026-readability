"""Gates B and C + test-oracle A/B for the D3Tok hybrid upgrade of reg_myd3.

Inputs (produced by score_reg_texts.py into artifacts/scores/blind/):
  regv2_{val,test,blind}.npy   - reg checkpoint on hybrid-A my-D3Tok v2 text
  regold_{val,test}.npy        - reg checkpoint on TODAY's old-pipeline text

Gate B (member level): solo quality of regv2 vs regold(today) vs July cache
  (reg_valmyd3.npy) on val, and vs the gold-text test file on test.
Gate C (blend level): honest val nested-OOF of fleet14 with reg_myd3's val
  score file swapped to regv2_val. Must beat 85.281 +-0.051.
Oracle A/B: deployment-mirrored test QWK of fleet14 with the reg member as
  (a) original: July val + gold-test stand-in (baseline 86.705)
  (b) old-today: regold_val + regold_test
  (c) new: regv2_val + regv2_test
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "endgame"))
import common as C  # noqa: E402
from common import OptimizedRounder, apply_cutpoints, fold_qwk, nested_oof_qwk, z  # noqa: E402

B = ROOT / "artifacts" / "scores" / "blind"
yva, fold, ytr, yte = C.data()


def solo_val(scores, label):
    qs = [nested_oof_qwk(scores, yva, fold, n_candidates=g)[0] * 100 for g in C.GRIDS]
    print(f"  {label:34s} val nested-rounder QWK {np.mean(qs):6.3f}  "
          f"corr(y) {np.corrcoef(scores, yva)[0, 1]:.4f}")


def solo_test_deploy(sv, st, label):
    pred = apply_cutpoints(z(st), C.deploy_cuts(z(sv), z(st)))
    print(f"  {label:34s} test deploy-QWK {float(fold_qwk(yte, pred)) * 100:6.3f}  "
          f"corr(y) {np.corrcoef(st, yte)[0, 1]:.4f}")


def blend_scores(reg_val, split, reg_split_scores):
    others = [m for m in C.FLEET14 if m != "reg_myd3"]
    s = sum(C.member_z(m, split) for m in others) + z(reg_split_scores)
    return s / 14.0


def main():
    rv2_val = np.load(B / "regv2_val.npy")
    rv2_test = np.load(B / "regv2_test.npy")          # train+test lex: self-lookup, UPPER BOUND
    rv2m_test = np.load(B / "regv2m_test.npy")        # train+val lex: honest blind-regime mirror
    rold_val = np.load(B / "regold_val.npy")
    rold_test = np.load(B / "regold_test.npy")
    rjul_val = np.load(B / "reg_valmyd3.npy")
    rgold_test = np.load(ROOT / "artifacts/scores/arabertv2-d3tok-reg_test.npy")

    print("cross-checks:")
    print(f"  corr(regold_val today, July reg_valmyd3) {np.corrcoef(rold_val, rjul_val)[0, 1]:.4f}")
    print(f"  corr(regv2_val, July reg_valmyd3)        {np.corrcoef(rv2_val, rjul_val)[0, 1]:.4f}")
    print(f"  corr(regv2_test, gold-text test)         {np.corrcoef(rv2_test, rgold_test)[0, 1]:.4f}")

    print("\nGATE B - member solo quality on val:")
    solo_val(rjul_val, "July my-D3Tok (~76% text)")
    solo_val(rold_val, "today old pipeline (63.7% text)")
    solo_val(rv2_val, "hybrid v2A (79.1% text)")
    print("GATE B - member solo on test (deploy-style, val-fitted):")
    solo_test_deploy(rold_val, rold_test, "today old pipeline")
    solo_test_deploy(rv2_val, rv2m_test, "hybrid v2 MIRROR (blind-regime)")
    solo_test_deploy(rv2_val, rv2_test, "hybrid v2 self-lookup (upper bd)")
    solo_test_deploy(np.load(ROOT / "artifacts/scores/arabertv2-d3tok-reg_validation.npy"),
                     rgold_test, "gold text (upper ref)")

    print("\nGATE C - fleet14 blend honest val nested-OOF (baseline 85.281 +-0.051):")
    sv_new = blend_scores(rv2_val, "val", rv2_val)
    qs = np.array([nested_oof_qwk(sv_new, yva, fold, n_candidates=g)[0] * 100 for g in C.GRIDS])
    print(f"  fleet14[reg->v2]                   {qs.mean():7.3f} +-{qs.std():.3f}   "
          f"delta {qs.mean() - 85.281:+.3f}")

    print("\nORACLE A/B - fleet14 deployment-mirrored test QWK (baseline (a) = 86.705):")
    for label, rv, rt in (("(b) old-today val+test", rold_val, rold_test),
                          ("(c) hybrid v2 MIRROR (decision)", rv2_val, rv2m_test),
                          ("(d) hybrid v2 self-lookup (upper)", rv2_val, rv2_test)):
        sv = blend_scores(rv, "val", rv)
        st = blend_scores(rt, "test", rt)
        pred = apply_cutpoints(st, C.deploy_cuts(sv, st))
        print(f"  {label:34s} {float(fold_qwk(yte, pred)) * 100:7.3f}")


if __name__ == "__main__":
    main()
