"""Cache each candidate member's BLIND scores (and the my-D3Tok VAL scores for the
d3tok members) so blend variants can be generated instantly without re-inference.
Run once. Saves to artifacts/scores/blind/.
    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blind_cache_scores.py
"""
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.data import ID, load_2026  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)
OUT = ROOT / "artifacts" / "scores" / "blind"
OUT.mkdir(parents=True, exist_ok=True)
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"

# (name, repo, tok_repo, head, input)
MEMBERS = [
    ("reg", "CAMeL-Lab/readability-arabertv2-d3tok-reg", None, "reg", "d3tok"),
    ("dtCE", "CAMeL-Lab/readability-arabertv2-d3tok-CE", None, "ce", "d3tok"),
    ("wordCE", "CAMeL-Lab/readability-arabertv02-word-CE", None, "ce", "word"),
    ("camelbert", "CAMeL-Lab/readability-camelbert-word-CE", None, "ce", "word"),
    ("qarib", "AymanTarig/qarib-barec-optimized-v8", "qarib/bert-base-qarib", "ce", "raw"),
]


def _levels(cfg):
    n = cfg.num_labels
    id2 = getattr(cfg, "id2label", None) or {}
    out = []
    for i in range(n):
        lab = str(id2.get(i, id2.get(str(i), "")))
        out.append(i + 1 if (lab == "" or lab.upper().startswith("LABEL_"))
                   else (int("".join(c for c in lab if c.isdigit())) if any(c.isdigit() for c in lab) else i + 1))
    return np.asarray(out, float)


@torch.no_grad()
def infer(repo, tokr, head, texts):
    tok = AutoTokenizer.from_pretrained(tokr or repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).eval().to(DEVICE)
    reg = (model.config.num_labels == 1) or head == "reg"
    lv = None if reg else _levels(model.config)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty(len(texts), float)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors="pt")
        lg = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits.float().cpu()
        out[idx] = lg.squeeze(-1).numpy() if reg else (torch.softmax(lg, -1).numpy() @ lv)
    del model, tok
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return out


def cmap(path, col):
    return {r["ID"]: r[col] for r in csv.DictReader(open(path))}


def main():
    blind = pl.read_parquet(BLIND)
    bids = [str(i) for i in blind[ID].to_list()]
    braw = {str(i): (s or "") for i, s in zip(blind[ID].to_list(), blind["Sentence"].to_list())}
    bword = cmap("/tmp/blind_word_mine.csv", "Word")
    bd3 = cmap("/tmp/blind_d3tok_mine.csv", "D3Tok")
    binp = {"raw": braw, "word": bword, "d3tok": bd3}

    val = load_2026("validation")
    vids = [str(i) for i in val[ID].to_list()]
    vraw = {str(i): (s or "") for i, s in zip(val[ID].to_list(), val["Sentence"].to_list())}
    vd3 = cmap("/tmp/val_d3tok_mine.csv", "D3Tok") if Path("/tmp/val_d3tok_mine.csv").exists() else None

    for name, repo, tokr, head, src in MEMBERS:
        bp = OUT / f"{name}_blind.npy"
        if not bp.exists():
            b = infer(repo, tokr, head, [binp[src].get(i, "") for i in bids])
            np.save(bp, b)
            print(f"  {name} blind: mean={b.mean():.2f} saved", flush=True)
        # d3tok members also need my-D3Tok VAL scores for standardization/rounder
        if src == "d3tok":
            vp = OUT / f"{name}_valmyd3.npy"
            if not vp.exists() and vd3 is not None:
                v = infer(repo, tokr, head, [vd3.get(i, "") for i in vids])
                np.save(vp, v)
                print(f"  {name} val(myD3): mean={v.mean():.2f} saved", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
