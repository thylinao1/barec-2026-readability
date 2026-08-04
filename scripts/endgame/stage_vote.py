"""Stage a per-row vote over board-measured blind prediction files.

Reads the EXACT prediction files the board scored (submissions/blind_blend/*),
votes per row, anchors half-integer medians to the incumbent (fleet14), and
stages through write_codabench_zip. No score files, no fitting, no drift.

    python stage_vote.py median vote6med fleet14 fleet14d3v2 fleet14avgv2 \
        fleet14bothv2 fleet14w25 fleet10strict
"""
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.submission import write_codabench_zip  # noqa: E402

SUB = ROOT / "submissions" / "blind_blend"


def preds_of(tag):
    z = zipfile.ZipFile(SUB / tag / "prediction.zip")
    rows = z.read("prediction").decode().splitlines()[1:]
    return {r.split(",")[0]: int(r.split(",")[1]) for r in rows}


def main():
    mode, out_tag, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    anchor = tags[0]
    files = {t: preds_of(t) for t in tags}
    ids = list(files[anchor])
    assert all(set(files[t]) == set(ids) for t in tags), "ID sets differ"
    M = np.array([[files[t][i] for t in tags] for i in ids], float)

    if mode == "median":
        med = np.median(M, axis=1)
        a = np.array([files[anchor][i] for i in ids], float)
        # half-integer medians: side with the incumbent
        out = np.where(np.abs(med - np.round(med)) > 1e-9,
                       np.where(a >= med, np.ceil(med), np.floor(med)),
                       np.round(med)).astype(int)
    else:
        out = np.rint(M.mean(axis=1)).astype(int)
    out = np.clip(out, 1, 19)

    diff = int((out != np.array([files[anchor][i] for i in ids])).sum())
    print(f"{out_tag}: {mode} of {len(tags)} files; {diff}/{len(ids)} rows "
          f"differ from {anchor} ({diff / len(ids) * 100:.1f}%)")
    zpath = write_codabench_zip(SUB / out_tag, ids, out)
    import shutil
    dest = Path.home() / "Desktop" / f"BLIND_{out_tag}.zip"
    shutil.copy(zpath, dest)
    print(f"staged {zpath}\ncopied {dest}")


if __name__ == "__main__":
    main()
