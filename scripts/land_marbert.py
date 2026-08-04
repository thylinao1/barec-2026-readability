"""Find, validate and install the fine-tuned MARBERTv2 checkpoint downloaded from Kaggle.

Handles either shape of download: a `model/` folder (Kaggle CLI) or a zip (browser Output tab).
Searches ~/Downloads and the project root, installs to models/marbertv2_finetuned/, and copies
this run's val/test .npy alongside as `.npy.kaggle` cross-check references.

THE GUARD THAT MATTERS: the .npy files must come from the SAME run as the weights. A previous
download turned out to be byte-identical to the cached July-14 scores, i.e. the OLD notebook's
output, which never called save_pretrained. This script fails loudly on that case, because
pairing new weights with old predictions would either trip the correlation gate for the wrong
reason or -- worse -- silently mis-standardize the member.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/land_marbert.py
    # add --run to chain straight into scoring val+test+blind
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "marbertv2_finetuned"
SCORES = ROOT / "artifacts" / "scores"
BACKUP = SCORES / "_backup_pre_marbert_rerun"
SEARCH = [Path.home() / "Downloads", Path.home() / "Downloads" / "marbert_out",
          Path.home() / "Desktop", ROOT]
NEEDED_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def find_model_dir():
    """A directory containing config.json plus a weights file."""
    for base in SEARCH:
        if not base.exists():
            continue
        for cfg in list(base.glob("**/config.json"))[:400]:
            d = cfg.parent
            if any((d / w).exists() for w in NEEDED_WEIGHTS):
                return d
    return None


def find_zip():
    cands = []
    for base in SEARCH:
        if not base.exists():
            continue
        for z in list(base.glob("*.zip"))[:200]:
            try:
                with zipfile.ZipFile(z) as zf:
                    names = zf.namelist()
                if any(n.endswith("config.json") for n in names) and \
                   any(n.endswith(w) for n in names for w in NEEDED_WEIGHTS):
                    cands.append((z.stat().st_size, z))
            except zipfile.BadZipFile:
                continue
    return max(cands)[1] if cands else None


def install_from_zip(zpath):
    stage = ROOT / ".marbert_unzip"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    print(f"extracting {zpath} ({zpath.stat().st_size/1e6:.0f} MB) ...")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(stage)
    for cfg in stage.glob("**/config.json"):
        if any((cfg.parent / w).exists() for w in NEEDED_WEIGHTS):
            return cfg.parent
    sys.exit(f"zip {zpath} had no usable model dir after extraction")


def check_npy():
    """Locate this run's val/test npy and refuse the stale July-14 ones."""
    found = {}
    for split, n in (("validation", 7310), ("test", 7286)):
        best = None
        for base in SEARCH:
            if not base.exists():
                continue
            for p in base.glob(f"**/marbertv2-reg_{split}*.npy"):
                try:
                    a = np.load(p).ravel()
                except Exception:
                    continue
                if len(a) == n:
                    mt = p.stat().st_mtime
                    if best is None or mt > best[0]:
                        best = (mt, p, a)
        found[split] = best
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="chain into marbert_local_infer.py after installing")
    ap.add_argument("--allow-stale-npy", action="store_true",
                    help="override the stale-npy guard (you almost never want this)")
    a = ap.parse_args()

    src = find_model_dir()
    if src is None:
        z = find_zip()
        if z is None:
            sys.exit("No checkpoint found. Looked for config.json + model.safetensors under:\n  "
                     + "\n  ".join(str(s) for s in SEARCH)
                     + "\n\nDownload the Output-tab `model/` folder (or its zip) first.")
        src = install_from_zip(z)

    print(f"found checkpoint at {src}")
    cfg = json.loads((src / "config.json").read_text())
    nl, pt = cfg.get("num_labels", cfg.get("id2label") and len(cfg["id2label"])), cfg.get("problem_type")
    arch = cfg.get("architectures")
    print(f"  architectures {arch}   num_labels {nl}   problem_type {pt}")
    if nl != 1:
        sys.exit(f"GATE FAIL: num_labels={nl}, expected 1. This looks like the BASE model, not "
                 f"the fine-tuned regression checkpoint. Do not proceed.")
    if arch and "ForSequenceClassification" not in arch[0]:
        sys.exit(f"GATE FAIL: architecture {arch} is not a sequence-classification head.")
    print("  GATE PASS: regression head with num_labels=1")

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy(f, DEST / f.name)
            print(f"  installed {f.name}  {f.stat().st_size:,} bytes")

    npys = check_npy()
    stale = []
    for split, best in npys.items():
        if best is None:
            print(f"  !! no marbertv2-reg_{split}.npy found -- the cross-check will be skipped")
            continue
        _, p, arr = best
        old = BACKUP / f"marbertv2-reg_{split}.npy"
        if old.exists() and np.array_equal(arr, np.load(old).ravel()):
            stale.append((split, p))
        shutil.copy(p, SCORES / f"marbertv2-reg_{split}.npy.kaggle")
        print(f"  cross-check ref <- {p.name}  (n={len(arr)}, mean={arr.mean():.3f})")

    if stale and not a.allow_stale_npy:
        for split, p in stale:
            print(f"\n!! {p} is BYTE-IDENTICAL to the cached July-14 {split} scores.")
        sys.exit(
            "\nGATE FAIL: those .npy files are the OLD notebook's output, not this run's.\n"
            "The weights you just installed came from a run that also wrote fresh\n"
            "marbertv2-reg_validation.npy / _test.npy -- download THOSE two files from the same\n"
            "Output tab and re-run. (Override with --allow-stale-npy only if you are certain the\n"
            "weights and these predictions came from the same training run.)")

    print(f"\ninstalled to {DEST}")
    cmd = ["scripts/marbert_local_infer.py", "models/marbertv2_finetuned", "marbertv2-reg"]
    if not a.run:
        print("next:\n  PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u " + " ".join(cmd))
        return
    print("\nscoring val + test + blind from one code path ...\n" + "=" * 72)
    subprocess.run([sys.executable, "-u", *cmd], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
