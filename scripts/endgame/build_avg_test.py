"""Reconstruct each avg-* member's seed list (verified numerically against the
stored avg val array) and synthesize the matching avg-*_test.npy from the
cluster-pulled per-seed test scores.

avg member := mean of z-scored seed arrays (make_avg_member.py definition).
The seed list was never recorded, so we recover it by search: group fleet
seed files into families (name minus -s<k>), and for each avg member find the
family + seed subset whose z-mean reproduces the stored avg val array to
allclose. All-seeds is tried first, then leave-one-out subsets.

Writes artifacts/scores/fleet/avg-*_test.npy and a manifest JSON.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/endgame/build_avg_test.py
"""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "artifacts" / "scores" / "fleet"
CT = ROOT / "artifacts" / "scores" / "cluster_test"


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def main():
    seed_files = sorted(p.name[:-8] for p in F.glob("*_val.npy")
                        if re.search(r"-s\d+$", p.name[:-8]))
    families = {}
    for s in seed_files:
        families.setdefault(re.sub(r"-s\d+$", "", s), []).append(s)

    avgs = sorted(p.name[:-8] for p in F.glob("avg-*_val.npy"))
    manifest, failures = {}, []
    for avg in avgs:
        target = np.load(F / f"{avg}_val.npy").ravel()
        stem = avg[4:]  # strip 'avg-'
        # candidate families: exact stem, or stem is a prefix (regime suffix dropped)
        cands = [k for k in families if k == stem] or \
                [k for k in families if k.startswith(stem)]
        found = None
        for fam in cands:
            seeds = sorted(families[fam])
            subsets = [seeds] + [list(c) for r in (len(seeds) - 1, len(seeds) - 2)
                                 if r >= 2
                                 for c in itertools.combinations(seeds, r)]
            for sub in subsets:
                m = np.mean([z(np.load(F / f"{s}_val.npy")) for s in sub], axis=0)
                if m.shape == target.shape and np.allclose(m, target, atol=1e-10):
                    found = (fam, sub)
                    break
            if found:
                break
        if not found:
            failures.append(avg)
            print(f"  {avg:32s} NO MATCH (families tried: {cands})")
            continue
        fam, sub = found
        test_paths = [CT / s / f"{s}_test.npy" for s in sub]
        missing = [p for p in test_paths if not p.exists()]
        if missing:
            failures.append(avg)
            print(f"  {avg:32s} seeds {sub} OK but missing cluster test npy: {missing}")
            continue
        out = np.mean([z(np.load(p)) for p in test_paths], axis=0)
        np.save(F / f"{avg}_test.npy", out)
        manifest[avg] = sub
        print(f"  {avg:32s} = {len(sub)} seeds  -> test written  ({sub})")

    (F / "avg_test_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} avg test members written, {len(failures)} failures")
    if failures:
        sys.exit(f"FAILURES: {failures}")


if __name__ == "__main__":
    main()
