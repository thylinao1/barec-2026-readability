"""Rank cluster-fleet members and propose candidate blends, WITHOUT landing weights.

Consumes the small artifacts synced down from the cluster (runs/*/meta.json +
*_validation.npy + *_test.npy) plus the existing local members, and runs the exact
honest-evaluation protocol of scripts/blind_myd3_members.py: per-member z-score on
val, equal-weight average, nested-OOF QWK with Document-grouped folds, mean over
n_candidates grids (200,300,400).

Two outputs:
  1. member table: solo val nested-OOF + max corr vs the champion members
     (decorrelation is where blend points come from)
  2. greedy forward selection from the champion base, printing the path and the
     honest delta at each step

Seed groups (same recipe, -s1..-s5) are pre-averaged into one synthetic member
named avg-<recipe> (mean of z-scored member scores), the proven variance cut.

NOTE the noise floor: offline deltas under ~0.3 are noise (HANDOFF 0d). This
script COARSELY ranks candidates for board probes; it does not pick winners.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/fleet_rank.py \
        [--fleet-dir artifacts/fleet/runs] [--max-add 4]
"""
import argparse
import json
import re
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
FOLD_SEED = 20260714
GRIDS = (200, 300, 400)

CHAMPION = ["wordCE", "camelbert", "qarib", "reg_myd3"]  # blend4reg, 84.8 on both boards
EXISTING_VAL = {
    "wordCE": S / "arabertv02-word-CE_validation.npy",
    "camelbert": S / "camelbert-word-CE_validation.npy",
    "qarib": S / "pub_AymanTarig__qarib-barec-optimized-v8_validation.npy",
    "marbert": S / "marbertv2-reg_validation.npy",
    "reg_myd3": B / "reg_valmyd3.npy",
    "dtCE_myd3": B / "dtCE_valmyd3.npy",
}


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def load_members(fleet_dir, n_val):
    zv, meta = {}, {}
    for m, p in EXISTING_VAL.items():
        if p.exists() and len(np.load(p).ravel()) == n_val:
            zv[m] = z(np.load(p))
            meta[m] = {"origin": "existing"}
    for run in sorted(Path(fleet_dir).glob("*/")):
        name = run.name
        vp = run / f"{name}_validation.npy"
        mj = run / "meta.json"
        if not vp.exists():
            continue
        a = np.load(vp).ravel()
        if len(a) != n_val:
            print(f"  (skip {name}: val len {len(a)})")
            continue
        zv[name] = z(a)
        meta[name] = json.loads(mj.read_text()) if mj.exists() else {}
        meta[name]["origin"] = "fleet"
    # public members scored by score_public_member.py (contaminated ones are
    # quarantined with a .CONTAMINATED suffix and never appear here)
    pub = S / "fleet"
    if pub.exists():
        for vp in sorted(pub.glob("*_val.npy")):
            name = vp.name[:-8]
            if name in zv or not (pub / f"{name}_blind.npy").exists():
                continue
            a = np.load(vp).ravel()
            if len(a) != n_val:
                continue
            zv[name] = z(a)
            meta[name] = {"origin": "public"}
    return zv, meta


def average_seed_groups(zv, meta):
    """avg-<recipe> := mean of z(seed members); members stay available too."""
    groups = {}
    for name in list(zv):
        m = re.match(r"^(.*)-s(\d+)$", name)
        if m and meta[name].get("origin") == "fleet":
            groups.setdefault(m.group(1), []).append(name)
    for recipe, names in groups.items():
        if len(names) >= 3:
            zv[f"avg-{recipe}"] = z(np.mean([zv[n] for n in names], axis=0))
            meta[f"avg-{recipe}"] = {"origin": "seed-avg", "members": sorted(names)}
    return zv, meta


def honest(sv, yva, fold):
    qs = np.array([nested_oof_qwk(sv, yva, fold, n_candidates=g)[0] * 100 for g in GRIDS])
    return qs.mean(), qs.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet-dir", default=str(ROOT / "artifacts/fleet/runs"))
    ap.add_argument("--max-add", type=int, default=4)
    ap.add_argument("--base", default=None,
                    help="comma list overriding the champion as the greedy start")
    ap.add_argument("--solo", action="store_true", help="also print solo nested-OOF per member (slow)")
    a = ap.parse_args()
    global CHAMPION
    if a.base:
        CHAMPION = [m.strip() for m in a.base.split(",")]

    val = load_2026("validation")
    yva = np.array(val[TARGET].to_list(), int)
    groups = val[GROUP].to_list()
    fold = fold_of_rows(build_document_fold_map(yva, groups, n_splits=5, seed=FOLD_SEED), groups)

    zv, meta = load_members(a.fleet_dir, len(yva))
    zv, meta = average_seed_groups(zv, meta)
    missing = [m for m in CHAMPION if m not in zv]
    if missing:
        sys.exit(f"champion members missing from val scores: {missing}")

    fleet = [m for m in zv if meta[m]["origin"] != "existing"]
    print(f"members: {len(zv)} total, {len(fleet)} fleet/seed-avg\n")

    champ_z = {m: zv[m] for m in CHAMPION}
    print(f"{'member':34s}{'maxcorr_champ':>14s}{'val_naive':>10s}  origin")
    rows = []
    for m in sorted(fleet):
        mc = max(abs(np.corrcoef(zv[m], cz)[0, 1]) for cz in champ_z.values())
        naive = meta[m].get("val_naive_qwk", "")
        rows.append((m, mc))
        print(f"{m:34s}{mc:14.4f}{str(naive):>10s}  {meta[m]['origin']}")

    base_sv = sum(zv[m] for m in CHAMPION) / len(CHAMPION)
    base_mu, base_sd = honest(base_sv, yva, fold)
    print(f"\nchampion blend4reg honest val nested-OOF: {base_mu:.3f} +-{base_sd:.3f}")

    # greedy forward selection from the champion
    current = list(CHAMPION)
    cur_mu = base_mu
    print("\n-- greedy forward selection (honest nested-OOF; noise floor ~0.3 applies)")
    for step in range(a.max_add):
        best = None
        for m in sorted(zv):
            if m in current:
                continue
            # skip raw seeds when their seed-avg exists (avg is strictly better-behaved)
            if re.match(r"^(.*)-s(\d+)$", m) and f"avg-{re.match(r'^(.*)-s(\d+)$', m).group(1)}" in zv:
                continue
            sv = sum(zv[x] for x in current + [m]) / (len(current) + 1)
            mu, sd = honest(sv, yva, fold)
            if best is None or mu > best[1]:
                best = (m, mu, sd)
        if best is None:
            break
        m, mu, sd = best
        print(f"  step {step+1}: + {m:30s} {mu:8.3f} +-{sd:.3f}  delta {mu - cur_mu:+.3f}")
        if mu <= cur_mu:
            print("  (no member improves further; stopping)")
            break
        current.append(m)
        cur_mu = mu

    print(f"\nproposed blend: {current}")
    print(f"honest val nested-OOF {cur_mu:.3f} (champion {base_mu:.3f}, delta {cur_mu-base_mu:+.3f})")
    print("\nNEXT: land ONLY the proposed additions with scripts/fleet_land.py, score blind,")
    print("stage with blind_blend.py (gate must PASS), probe on OPEN first.")


if __name__ == "__main__":
    main()
