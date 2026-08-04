"""Score val + test + BLIND with a fine-tuned Kaggle checkpoint downloaded locally.

Why recompute val/test locally instead of reusing the Kaggle .npy: the blend standardizes
each member using its VAL mean/std, so val/test/blind must come from ONE identical code path
and one identical set of weights. Recomputing all three here removes any fp16/kernel/version
mismatch between Kaggle's GPU and this machine's MPS. The Kaggle .npy files are then used
only as a cross-check (correlation should be ~1.0; a low value means the wrong checkpoint).

Input text is RAW `Sentence` -- MARBERTv2/AraELECTRA were fine-tuned on raw text
(TEXT_COL="Sentence" in kaggle_finetune.py), so NO camel-tools preprocessing here. Do not
"helpfully" add Word/D3Tok preprocessing: it would silently wreck the member.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u \
        scripts/marbert_local_infer.py models/marbertv2_finetuned marbertv2-reg
"""
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
SCORES = ROOT / "artifacts" / "scores"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"
MAX_LEN = 128


@torch.no_grad()
def infer(model, tok, texts):
    """Length-sorted batching for speed; results restored to original order."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty(len(texts), float)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        lg = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits.float().cpu()
        out[idx] = lg.squeeze(-1).numpy()
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: marbert_local_infer.py <model_dir> <out_name>   "
                 "e.g. models/marbertv2_finetuned marbertv2-reg")
    mdir, name = sys.argv[1], sys.argv[2]

    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).eval().to(DEVICE)
    nl = model.config.num_labels
    print(f"loaded {mdir}  num_labels={nl}  device={DEVICE}")
    if nl != 1:
        print(f"  !! WARNING: expected a regression head (num_labels=1), got {nl}. "
              f"This is probably the BASE model, not the fine-tuned checkpoint.")

    for split in ("validation", "test"):
        df = load_2026(split)
        sc = infer(model, tok, [s or "" for s in df["Sentence"].to_list()])
        np.save(SCORES / f"{name}_{split}.npy", sc)
        old = SCORES / f"{name}_{split}.npy.kaggle"
        print(f"  {split}: n={len(sc)} mean={sc.mean():.3f} sd={sc.std():.3f} -> saved")
        if old.exists():  # cross-check against the Kaggle-produced scores
            o = np.load(old).ravel()
            c = np.corrcoef(sc, o)[0, 1]
            print(f"    cross-check vs Kaggle npy: corr={c:.6f}"
                  f"{'  OK' if c > 0.99 else '  !! MISMATCH - wrong checkpoint?'}")

    blind = pl.read_parquet(BLIND)
    bs = infer(model, tok, [s or "" for s in blind["Sentence"].to_list()])
    outp = SCORES / "blind" / f"{name.split('-')[0]}_blind.npy"
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, bs)
    print(f"  blind: n={len(bs)} mean={bs.mean():.3f} sd={bs.std():.3f} -> {outp}")

    del model, tok
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    print("\ndone. next: rebuild the blend with this member and stage a submission.")


if __name__ == "__main__":
    main()
