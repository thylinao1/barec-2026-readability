"""Score a d3tok-regime checkpoint on arbitrary D3Tok text CSVs (MPS, one at a
time), mirroring fleet_land.py's inference exactly (reg head -> raw logit;
CE head -> posterior mean; batch 32, length-sorted, max_len 128).

  python score_reg_texts.py <model_dir_or_hf_id> <loss> <out_prefix> \
      <split>=<d3tok_csv> [...]

Writes artifacts/scores/blind/<out_prefix>_<split>.npy  (row order = the
canonical split ID order: val/test from load_2026, blind from the parquet).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.data import ID, load_2026  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)
LEVELS = np.arange(1, 20, dtype=float)
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"
OUT = ROOT / "artifacts" / "scores" / "blind"


def ids_for(split):
    if split == "blind":
        return [str(i) for i in pl.read_parquet(BLIND)[ID].to_list()]
    name = {"val": "validation", "test": "test"}[split]
    return [str(i) for i in load_2026(name)[ID].to_list()]


@torch.no_grad()
def infer(model, tok, texts, loss):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty(len(texts), float)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=128, return_tensors="pt")
        lg = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits.float().cpu()
        if loss == "reg":
            out[idx] = lg.squeeze(-1).numpy()
        else:
            out[idx] = torch.softmax(lg, -1).numpy() @ LEVELS
    return out


def main():
    model_id, loss, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    jobs = [a.split("=", 1) for a in sys.argv[4:]]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval().to(DEVICE)
    print(f"{model_id}: num_labels={model.config.num_labels} loss={loss}")
    for split, csvpath in jobs:
        m = {r["ID"]: r["D3Tok"] for r in csv.DictReader(open(csvpath))}
        idl = ids_for(split)
        missing = sum(1 for i in idl if i not in m)
        assert missing == 0, f"{split}: {missing} IDs missing from {csvpath}"
        s = infer(model, tok, [m[i] or "" for i in idl], loss)
        p = OUT / f"{prefix}_{split}.npy"
        np.save(p, s)
        print(f"  {split}: {len(idl)} rows scored from {csvpath} -> {p.name} "
              f"(mean {s.mean():.3f} sd {s.std():.3f})")


if __name__ == "__main__":
    main()
