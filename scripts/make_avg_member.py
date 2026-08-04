"""Synthesize a seed-averaged member from landed per-seed score files.

avg member := mean of z-scored seed arrays, computed separately for val and
blind (matches fleet_rank.py's avg definition). Writes
artifacts/scores/fleet/avg-<recipe>_{val,blind}.npy for blind_blend.py.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/make_avg_member.py \
        avg-arabertv02-ce-word=arabertv02-ce-word-s1,arabertv02-ce-word-s2,...
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "artifacts" / "scores" / "fleet"


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: make_avg_member.py <avgname>=<seed1,seed2,...> [...]")
    for spec in sys.argv[1:]:
        name, seeds = spec.split("=")
        seeds = seeds.split(",")
        for split in ("val", "blind"):
            arrs = []
            for s in seeds:
                p = F / f"{s}_{split}.npy"
                if not p.exists():
                    sys.exit(f"missing {p} (land it first)")
                arrs.append(z(np.load(p)))
            out = np.mean(arrs, axis=0)
            np.save(F / f"{name}_{split}.npy", out)
        print(f"{name}: {len(seeds)} seeds -> val+blind written")


if __name__ == "__main__":
    main()
