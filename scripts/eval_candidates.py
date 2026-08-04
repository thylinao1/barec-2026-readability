"""Evaluate hand-picked candidate blends under the honest protocol, using the
EXACT val score files the blender itself consumes (blind_blend.py's REG map,
incl. auto-discovered fleet members). This is the staging-decision number;
fleet_rank.py's greedy uses cluster-side val files and is only a coarse guide.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/eval_candidates.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.cv import build_document_fold_map, fold_of_rows  # noqa: E402
from harness.data import GROUP, TARGET, load_2026  # noqa: E402
from harness.rounder import nested_oof_qwk  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from blind_blend import REG  # noqa: E402  (the blender's own member registry)

FOLD_SEED = 20260714
GRIDS = (200, 300, 400)

CANDIDATES = [
    ("champion blend4reg (board 84.8)", ["wordCE", "camelbert", "qarib", "reg_myd3"]),
    ("+ avg-arabertv2-reg-d3tok", ["wordCE", "camelbert", "qarib", "reg_myd3",
                                   "avg-arabertv2-reg-d3tok"]),
    ("+ avgd3 + avg-arabertv02", ["wordCE", "camelbert", "qarib", "reg_myd3",
                                  "avg-arabertv2-reg-d3tok", "avg-arabertv02-ce-word"]),
    ("+ avgd3 + avg-arabertv02 + avg-camelbert",
     ["wordCE", "camelbert", "qarib", "reg_myd3", "avg-arabertv2-reg-d3tok",
      "avg-arabertv02-ce-word", "avg-camelbert-ce-word"]),
    ("+ all4 avgs", ["wordCE", "camelbert", "qarib", "reg_myd3",
                     "avg-arabertv2-reg-d3tok", "avg-arabertv02-ce-word",
                     "avg-camelbert-ce-word", "avg-qarib-ce-raw"]),
    ("+ all4 avgs + alarge02-reg", ["wordCE", "camelbert", "qarib", "reg_myd3",
                                    "avg-arabertv2-reg-d3tok", "avg-arabertv02-ce-word",
                                    "avg-camelbert-ce-word", "avg-qarib-ce-raw",
                                    "alarge02-reg-word-s42"]),
    ("+ all4 avgs + alarge02 + monda_emd + oll15",
     ["wordCE", "camelbert", "qarib", "reg_myd3", "avg-arabertv2-reg-d3tok",
      "avg-arabertv02-ce-word", "avg-camelbert-ce-word", "avg-qarib-ce-raw",
      "alarge02-reg-word-s42", "monda_d3tok_emd", "monda_sent_oll15"]),
    ("+ all4 avgs + alarge02 + qarib_mbv2 + araelectra",
     ["wordCE", "camelbert", "qarib", "reg_myd3", "avg-arabertv2-reg-d3tok",
      "avg-arabertv02-ce-word", "avg-camelbert-ce-word", "avg-qarib-ce-raw",
      "alarge02-reg-word-s42", "qarib_marbertv2_v1", "araelectra-reg-word-s42"]),
    ("kitchen sink 13", ["wordCE", "camelbert", "qarib", "reg_myd3",
                         "avg-arabertv2-reg-d3tok", "avg-arabertv02-ce-word",
                         "avg-camelbert-ce-word", "avg-qarib-ce-raw",
                         "alarge02-reg-word-s42", "monda_d3tok_emd", "monda_sent_oll15",
                         "qarib_marbertv2_v1", "araelectra-reg-word-s42"]),
    ("marbert variant (board-refuted solo; recheck in big ctx)",
     ["wordCE", "camelbert", "qarib", "reg_myd3", "avg-arabertv2-reg-d3tok",
      "avg-arabertv02-ce-word", "marbert"]),
]


def z(a):
    a = np.asarray(a).ravel().astype(float)
    return (a - a.mean()) / (a.std() or 1)


def main():
    val = load_2026("validation")
    yva = np.array(val[TARGET].to_list(), int)
    fold = fold_of_rows(build_document_fold_map(yva, val[GROUP].to_list(),
                                                n_splits=5, seed=FOLD_SEED),
                        val[GROUP].to_list())
    zv = {}
    for m, (vp, bp) in REG.items():
        if vp.exists() and bp.exists():
            a = np.load(vp).ravel()
            if len(a) == len(yva):
                zv[m] = z(a)

    print(f"blender registry: {len(zv)} members with val+blind ready\n")
    base = None
    rows = []
    for label, members in CANDIDATES:
        missing = [m for m in members if m not in zv]
        if missing:
            print(f"  {label:55s} SKIP (missing {missing})")
            continue
        sv = sum(zv[m] for m in members) / len(members)
        qs = np.array([nested_oof_qwk(sv, yva, fold, n_candidates=g)[0] * 100
                       for g in GRIDS])
        if base is None:
            base = qs.mean()
        rows.append((label, qs.mean(), qs.std(), len(members)))
        print(f"  {label:55s} {qs.mean():7.3f} +-{qs.std():.3f}  "
              f"delta {qs.mean()-base:+.3f}  ({len(members)} members)")

    print("\nranked:")
    for label, mu, sd, n in sorted(rows, key=lambda r: -r[1]):
        print(f"  {mu:7.3f} +-{sd:.3f}  ({n:2d}m)  {label}")


if __name__ == "__main__":
    main()
