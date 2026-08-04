"""Stage the 4-member blind submission (wordCE + camelbert + qarib8 + MARBERTv2).

Extends the 84.5 champion by ONE variable: a fourth member. Everything else -- per-split
self-standardization, OptimizedRounder(n_candidates=300) fit on val, and the raw-train-prior
half-match calibration -- is copied verbatim from scripts/blind_reweighted_prior.py, whose
`raw50_reference` output was verified byte-identical to ~/Desktop/BLIND_halfmatch.zip.

REGRESSION GATE: run with `--gate` and the script rebuilds the champion by setting the marbert
weight to 0. The result must be byte-identical to BLIND_halfmatch.zip. If it is not, the code
path has drifted and NOTHING produced here can be trusted -- stop and fix it.

Calibration is EXHAUSTED as a lever (board evidence: 0% 84.1 | 50% 84.5 | 60% 84.4 | 100% 84.0
| 50% reweighted-prior 83.9). Do not sweep `--frac` looking for gains; 0.50 is the optimum.

    # gate first, always
    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_blend4.py --gate
    # then stage
    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_blend4.py \
        --marbert-weight 0.231 --alpha 0.1 --tag blend4_a01
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
OUT = ROOT / "submissions" / "blind_blend4"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"
CHAMPION = Path.home() / "Desktop" / "BLIND_halfmatch.zip"
N_CAND = 300

# order matters for bit-exact reproduction of the champion: the first three, in this order,
# are what blind_reweighted_prior.py summed.
VAL = {"wordCE": S / "arabertv02-word-CE_validation.npy",
       "camelbert": S / "camelbert-word-CE_validation.npy",
       "qarib": S / "pub_AymanTarig__qarib-barec-optimized-v8_validation.npy",
       "marbert": S / "marbertv2-reg_validation.npy"}
BLD = {"wordCE": B / "wordCE_blind.npy", "camelbert": B / "camelbert_blind.npy",
       "qarib": B / "qarib_blind.npy", "marbert": B / "marbert_blind.npy"}


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def weighted(paths, weights):
    """sum(w_i * z(x_i)) / sum(w). With w=(1,1,1,0) this is bit-identical to the
    champion's sum(z(x_i))/3 -- adding 0.0*z is an exact no-op in IEEE754."""
    return sum(weights[m] * z(np.load(paths[m])) for m in weights) / sum(weights.values())


def doc_shrink(scores, docs, alpha):
    """score := (1-a)*score + a*document_mean(score). Real but fragile: HANDOFF 4c measured
    the test-set peak at a=0.1 and a collapse to 82.15 by a=0.5. Never exceed 0.2."""
    if not alpha:
        return scores
    out = np.asarray(scores, float).copy()
    d = np.asarray([str(x) for x in docs])
    mu = np.empty_like(out)
    for u in np.unique(d):
        m = d == u
        mu[m] = out[m].mean()
    return (1 - alpha) * out + alpha * mu


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def champion_predictions():
    """The 84.5 file's predictions, keyed by ID, read straight out of the zip."""
    with zipfile.ZipFile(CHAMPION) as zf:
        lines = zf.read("prediction").decode().splitlines()
    return {r.split(",")[0]: int(r.split(",")[1]) for r in lines[1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marbert-weight", type=float, default=0.231)
    ap.add_argument("--alpha", type=float, default=0.0, help="doc shrinkage; 0.1 max recommended")
    ap.add_argument("--frac", type=float, default=0.50, help="half-match ratio; 0.50 is optimal")
    ap.add_argument("--tag", default="blend4")
    ap.add_argument("--gate", action="store_true",
                    help="rebuild the champion (marbert w=0) and assert byte-identity")
    ap.add_argument("--shrink-val", action="store_true",
                    help="also shrink val before fitting the rounder (matches offline tests)")
    a = ap.parse_args()

    if a.gate:
        a.marbert_weight, a.alpha, a.frac, a.tag = 0.0, 0.0, 0.50, "GATE_champion_rebuild"

    missing = [m for m, p in BLD.items() if not p.exists()]
    if a.marbert_weight == 0:
        missing = [m for m in missing if m != "marbert"]
    if missing:
        sys.exit(f"missing blind scores for {missing}. Run scripts/marbert_local_infer.py first.")

    weights = {"wordCE": 0.306, "camelbert": 0.231, "qarib": 0.231, "marbert": a.marbert_weight}
    if a.gate:  # the champion was EQUAL-weight over the three word members
        weights = {"wordCE": 1.0, "camelbert": 1.0, "qarib": 1.0, "marbert": 0.0}

    train = load_2026("train"); ytr = np.array(train[TARGET].to_list(), int)
    val = load_2026("validation"); yva = np.array(val[TARGET].to_list(), int)
    blind = pl.read_parquet(BLIND); bids = [str(i) for i in blind[ID].to_list()]

    val_paths = {m: VAL[m] for m in weights}
    bld_paths = {m: BLD[m] for m in weights}
    if a.marbert_weight == 0:  # do not even touch the file when it is unused
        val_paths = {m: p for m, p in val_paths.items() if m != "marbert"}
        bld_paths = {m: p for m, p in bld_paths.items() if m != "marbert"}
        weights = {m: w for m, w in weights.items() if m != "marbert"}

    sv = weighted(val_paths, weights)
    sb = weighted(bld_paths, weights)
    if a.alpha:
        sb = doc_shrink(sb, blind["Document"].to_list(), a.alpha)
        if a.shrink_val:
            sv = doc_shrink(sv, val["Document"].to_list(), a.alpha)

    rounder = OptimizedRounder(n_candidates=N_CAND).fit(sv, yva)
    prior = np.array([(ytr == k).mean() for k in range(1, 20)])
    q = np.quantile(sb, np.cumsum(prior)[:-1])
    cut = np.sort((1 - a.frac) * rounder.cutpoints_ + a.frac * q)
    pred = apply_cutpoints(sb, cut)

    print(f"weights      {weights}")
    print(f"alpha        {a.alpha}  (shrink_val={a.shrink_val})   frac {a.frac}")
    print(f"val rounder  in-sample QWK {rounder.oof_qwk_*100:.3f}")
    print(f"pred dist    " + " ".join(f"{k}:{(pred == k).mean()*100:.1f}" for k in range(8, 16)))

    zpath = write_codabench_zip(OUT / a.tag, bids, pred)
    print(f"staged       {zpath}  sha {sha(zpath)}")

    champ = champion_predictions()
    if set(champ) != set(bids):
        print("!! champion ID set differs from blind IDs -- investigate")
    diff = [i for i in bids if champ[i] != pred[list(bids).index(i)]] if a.gate else None

    if a.gate:
        mine = dict(zip(bids, pred.tolist()))
        same = all(champ[i] == mine[i] for i in bids)
        print("\n" + "=" * 70)
        if same:
            print("GATE PASS: rebuilt champion is IDENTICAL to BLIND_halfmatch.zip "
                  f"({len(bids)} rows). The 4-member code path is trustworthy.")
        else:
            n = sum(champ[i] != mine[i] for i in bids)
            print(f"GATE FAIL: {n}/{len(bids)} predictions differ from the champion. "
                  "The code path has DRIFTED -- do not stage anything until this is fixed.")
            sys.exit(1)
        return

    # not the gate: report how far this moves from the 84.5 file
    mine = dict(zip(bids, pred.tolist()))
    d = [i for i in bids if champ[i] != mine[i]]
    shifts = {}
    for i in d:
        s = mine[i] - champ[i]
        shifts[s] = shifts.get(s, 0) + 1
    print(f"\nvs champion   {len(d)}/{len(bids)} = {len(d)/len(bids)*100:.1f}% changed")
    print(f"shift hist    {dict(sorted(shifts.items()))}")
    print(f"max |shift|   {max((abs(k) for k in shifts), default=0)}")
    dest = Path.home() / "Desktop" / f"BLIND_{a.tag}.zip"
    shutil.copy(zpath, dest)
    print(f"\ncopied to     {dest}")
    print("Upload to the OPEN track (16544) FIRST -- it has a 3.3 QWK cushion and reads the "
          "same blind gold, so it is a free oracle. Only promote to STRICT (16545) if the "
          "live Open score STRICTLY exceeds 84.5.")


if __name__ == "__main__":
    main()
