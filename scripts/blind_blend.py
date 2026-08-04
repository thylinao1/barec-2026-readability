"""Stage a blind submission from an arbitrary member set. Supersedes blind_blend4.py.

Everything except the member list is copied verbatim from scripts/blind_reweighted_prior.py,
whose `raw50_reference` output was verified byte-identical to ~/Desktop/BLIND_halfmatch.zip
(the 84.5 champion): per-split self-standardization, OptimizedRounder(n_candidates=300) fit on
val, and the raw-train-prior half-match calibration at frac=0.50.

REGRESSION GATE (`--gate`): rebuilds the champion from its three members at equal weight and
asserts byte-identity with BLIND_halfmatch.zip. Verified PASS 2026-07-25. Run it after any edit
to this file -- if it fails, the code path has drifted and nothing staged here is trustworthy.

MEMBER NOTE -- why `reg_myd3` exists and `reg_gold` must never be used for blind:
our locally computed D3Tok matches the license-gated LDC gold at only ~76%. HANDOFF 4d
concluded the D3Tok members were therefore unusable on blind, but that test standardized the
member with GOLD-D3Tok val scores while applying it to MY-D3Tok blind scores -- two different
preprocessing regimes, which z-scoring cannot reconcile. Pairing my-D3Tok val scores
(`blind/reg_valmyd3.npy`) with my-D3Tok blind scores makes the member self-consistent.
Measured honest val nested-OOF, mean over n_candidates in (200,300,400), 2026-07-25:

    wordonly3 (champion)              84.422 +-0.057    baseline
    + marbert                         84.517 +-0.021    +0.095
    + reg_myd3                        84.508 +-0.019    +0.086
    + dtCE_myd3                       84.367 +-0.029    -0.055   <- hurts, excluded
    + marbert + reg_myd3              84.828 +-0.019    +0.406   <- BEST blind-safe
    + marbert + reg_myd3 + dtCE_myd3  84.811 +-0.041    +0.389
    + reg_gold + dtCE_gold            85.009 +-0.030    +0.587   <- NOT blind-safe, contrast only

marbert and reg_myd3 are the two least-correlated members (0.919-0.941 and 0.922-0.960 against
the rest) and their gains are super-additive: +0.406 together vs +0.095/+0.086 alone.

Calibration is EXHAUSTED as a lever (board: 0% 84.1 | 50% 84.5 | 60% 84.4 | 100% 84.0 |
50% reweighted-prior 83.9). Do not sweep --frac.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_blend.py --gate
    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_blend.py \
        --members wordCE,camelbert,qarib,marbert,reg_myd3 --tag blend5
"""
import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.data import ID, TARGET, load_2026  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints  # noqa: E402
from harness.submission import write_codabench_zip  # noqa: E402

S = ROOT / "artifacts" / "scores"
B = S / "blind"
OUT = ROOT / "submissions" / "blind_blend"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"
CHAMPION = Path.home() / "Desktop" / "BLIND_halfmatch.zip"
N_CAND = 300
CHAMPION_MEMBERS = ["wordCE", "camelbert", "qarib"]  # order is load-bearing for bit-exactness

# member -> (val score path, blind score path). The D3Tok members deliberately take their VAL
# scores from the my-D3Tok files so val and blind share one preprocessing regime.
REG = {
    "wordCE":    (S / "arabertv02-word-CE_validation.npy",                    B / "wordCE_blind.npy"),
    "camelbert": (S / "camelbert-word-CE_validation.npy",                     B / "camelbert_blind.npy"),
    "qarib":     (S / "pub_AymanTarig__qarib-barec-optimized-v8_validation.npy", B / "qarib_blind.npy"),
    "marbert":   (S / "marbertv2-reg_validation.npy",                         B / "marbertv2_blind.npy"),
    "reg_myd3":  (B / "reg_valmyd3.npy",                                      B / "reg_blind.npy"),
    "dtCE_myd3": (B / "dtCE_valmyd3.npy",                                     B / "dtCE_blind.npy"),
}
# fleet members landed by scripts/fleet_land.py register themselves by file pair
for _vp in sorted((S / "fleet").glob("*_val.npy")) if (S / "fleet").exists() else []:
    _name = _vp.name[:-8]  # strip _val.npy
    _bp = S / "fleet" / f"{_name}_blind.npy"
    if _bp.exists():
        REG[_name] = (_vp, _bp)


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def weighted(members, weights, which):
    """sum(w_i*z(x_i))/sum(w). Equal weights reproduce the champion's sum(z)/n bit-exactly."""
    idx = 0 if which == "val" else 1
    return sum(weights[m] * z(np.load(REG[m][idx])) for m in members) / sum(weights.values())


def doc_shrink(scores, docs, alpha):
    """score := (1-a)*score + a*document_mean(score). Fragile: HANDOFF 4c measured the test
    peak at a=0.1 and a collapse to 82.15 by a=0.5. Blind has 212 docs, median 40 sentences."""
    if not alpha:
        return scores
    out = np.asarray(scores, float).copy()
    d = np.asarray([str(x) for x in docs])
    mu = np.empty_like(out)
    for u in np.unique(d):
        m = d == u
        mu[m] = out[m].mean()
    return (1 - alpha) * out + alpha * mu


def champion_predictions():
    with zipfile.ZipFile(CHAMPION) as zf:
        lines = zf.read("prediction").decode().splitlines()
    return {r.split(",")[0]: int(r.split(",")[1]) for r in lines[1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="wordCE,camelbert,qarib,marbert,reg_myd3")
    ap.add_argument("--weights", default="equal",
                    help="'equal' or a comma list matching --members")
    ap.add_argument("--alpha", type=float, default=0.0, help="doc shrinkage; 0.1 max")
    ap.add_argument("--frac", type=float, default=0.50, help="half-match ratio; 0.50 is optimal")
    ap.add_argument("--tag", default="blend")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--stab-boots", type=int, default=0,
                    help="document-bootstrap-average the val-tuned cutpoints over B "
                         "resamples before half-match (variance reduction; 0 = off)")
    ap.add_argument("--no-shrink-val", action="store_true",
                    help="shrink blind only, leaving the rounder in unshrunk val space")
    a = ap.parse_args()

    members = CHAMPION_MEMBERS if a.gate else [m.strip() for m in a.members.split(",")]
    if a.gate:
        a.weights, a.alpha, a.frac, a.tag = "equal", 0.0, 0.50, "GATE_champion_rebuild"

    unknown = [m for m in members if m not in REG]
    if unknown:
        sys.exit(f"unknown members {unknown}. known: {sorted(REG)}")
    absent = [m for m in members if not REG[m][0].exists() or not REG[m][1].exists()]
    if absent:
        sys.exit(f"missing score files for {absent}.\n"
                 f"  marbert needs scripts/marbert_local_infer.py to have been run.\n"
                 f"  reg_myd3/dtCE_myd3 need artifacts/scores/blind/*_valmyd3.npy + *_blind.npy")

    if a.weights == "equal":
        weights = {m: 1.0 for m in members}
    else:
        vals = [float(x) for x in a.weights.split(",")]
        if len(vals) != len(members):
            sys.exit(f"--weights has {len(vals)} values for {len(members)} members")
        weights = dict(zip(members, vals))

    train = load_2026("train"); ytr = np.array(train[TARGET].to_list(), int)
    val = load_2026("validation"); yva = np.array(val[TARGET].to_list(), int)
    blind = pl.read_parquet(BLIND); bids = [str(i) for i in blind[ID].to_list()]

    sv = weighted(members, weights, "val")
    sb = weighted(members, weights, "blind")
    if a.alpha:
        # Shrink BOTH sides. The rounder's cutpoints live in val-score space, so shrinking only
        # blind would narrow the blind distribution while leaving the cutpoints wide -- a
        # different transform from the one measured offline (+0.22 on test, which shrank both).
        sb = doc_shrink(sb, blind["Document"].to_list(), a.alpha)
        if not a.no_shrink_val:
            sv = doc_shrink(sv, val["Document"].to_list(), a.alpha)

    rounder = OptimizedRounder(n_candidates=N_CAND).fit(sv, yva)
    val_cuts = rounder.cutpoints_
    if a.stab_boots:
        # Bootstrap DOCUMENTS (the CV grouping unit) so resamples respect topic
        # structure; average the tuned cutpoints. Attacks the grid-snap variance
        # (sd 0.094) without changing the transform the board already validated.
        rng = np.random.default_rng(20260714)
        docs = np.asarray([str(x) for x in val["Document"].to_list()])
        uniq = np.unique(docs)
        boots = []
        for b in range(a.stab_boots):
            take = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(docs == d) for d in take])
            boots.append(OptimizedRounder(n_candidates=N_CAND)
                         .fit(sv[idx], yva[idx], init=val_cuts).cutpoints_)
        val_cuts = np.sort(np.mean(boots, axis=0))
        print(f"stabilized cutpoints over {a.stab_boots} document bootstraps "
              f"(mean |shift| {np.abs(val_cuts - rounder.cutpoints_).mean():.4f})")
    prior = np.array([(ytr == k).mean() for k in range(1, 20)])
    cut = np.sort((1 - a.frac) * val_cuts
                  + a.frac * np.quantile(sb, np.cumsum(prior)[:-1]))
    pred = apply_cutpoints(sb, cut)

    print(f"members      {members}")
    print(f"weights      {weights}")
    print(f"alpha {a.alpha}   frac {a.frac}")
    print(f"val rounder  in-sample QWK {rounder.oof_qwk_*100:.3f}")
    print("pred dist    " + " ".join(f"{k}:{(pred == k).mean()*100:.1f}" for k in range(8, 16)))

    zpath = write_codabench_zip(OUT / a.tag, bids, pred)
    print(f"staged       {zpath}")
    print(f"sha256[:16]  {hashlib.sha256(Path(zpath).read_bytes()).hexdigest()[:16]}")

    champ = champion_predictions()
    mine = dict(zip(bids, pred.tolist()))
    if set(champ) != set(bids):
        sys.exit("!! champion ID set differs from blind IDs -- investigate before uploading")
    diff = [i for i in bids if champ[i] != mine[i]]

    if a.gate:
        print("\n" + "=" * 72)
        if not diff:
            print(f"GATE PASS: rebuilt champion identical to BLIND_halfmatch.zip ({len(bids)} rows).")
            return
        print(f"GATE FAIL: {len(diff)}/{len(bids)} predictions differ. Code path DRIFTED.")
        sys.exit(1)

    shifts = {}
    for i in diff:
        s = mine[i] - champ[i]
        shifts[s] = shifts.get(s, 0) + 1
    print(f"\nvs champion  {len(diff)}/{len(bids)} = {len(diff)/len(bids)*100:.1f}% changed")
    print(f"shift hist   {dict(sorted(shifts.items()))}   max |shift| "
          f"{max((abs(k) for k in shifts), default=0)}")
    dest = Path.home() / "Desktop" / f"BLIND_{a.tag}.zip"
    shutil.copy(zpath, dest)
    print(f"copied to    {dest}")
    print("\nUPLOAD TO THE OPEN TRACK (16544) FIRST. It reads the same blind gold and we lead it\n"
          "by 3.3 QWK, so a probe there is free. Promote to STRICT (16545) ONLY if the live Open\n"
          "score STRICTLY exceeds 84.5 -- our Strict 84.5 is timestamped 07-24 07:38, ahead of\n"
          "the tied entry's 13:35, and any new Strict upload resets that timestamp and forfeits the tie.")


if __name__ == "__main__":
    main()
