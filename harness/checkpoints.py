"""Phase 2 - inference over the public CAMeL-Lab readability checkpoints.

ONE model at a time on the 8GB M2: load, run forward passes (max_len 128,
torch.no_grad, CPU, threads capped), decode to a continuous per-row score in
level space, cache to .npy, then unload before the next. Never two at once.

Decoding (Section 3.4 of the plan):
- regression head (num_labels==1): raw predicted score.
- classification head: expected value E[level] = sum_k level_k * softmax_k, NOT
  argmax (argmax throws away the ordinal information a QWK blend needs). level_k
  is read from config.id2label; if labels are generic (LABEL_0..), level = k+1.

The 4 public checkpoints and their heads:
  readability-arabertv2-d3tok-reg   -> reg,  D3Tok  input
  readability-arabertv2-d3tok-CE    -> CE,   D3Tok  input
  readability-arabertv02-word-CE    -> CE,   Word   input (raw Sentence)
  readability-camelbert-word-CE     -> CE,   Word   input (raw Sentence)
"""
from __future__ import annotations

import gc

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINTS = {
    "arabertv2-d3tok-reg": {"repo": "CAMeL-Lab/readability-arabertv2-d3tok-reg", "head": "reg", "tok": "d3tok"},
    "arabertv2-d3tok-CE": {"repo": "CAMeL-Lab/readability-arabertv2-d3tok-CE", "head": "ce", "tok": "d3tok"},
    "arabertv02-word-CE": {"repo": "CAMeL-Lab/readability-arabertv02-word-CE", "head": "ce", "tok": "word"},
    "camelbert-word-CE": {"repo": "CAMeL-Lab/readability-camelbert-word-CE", "head": "ce", "tok": "word"},
}

torch.set_num_threads(4)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _level_vector(config) -> np.ndarray:
    """Map class index -> integer readability level from config.id2label.

    The public readability CE checkpoints ship GENERIC HF labels
    (LABEL_0..LABEL_18) that encode the class INDEX, not the level: class k is
    readability level k+1 (levels 1..19). So generic labels map to i+1. Only a
    non-generic label that literally spells a level (e.g. "12") is parsed as-is.
    """
    n = config.num_labels
    id2label = getattr(config, "id2label", None) or {}
    levels = []
    for i in range(n):
        lab = str(id2label.get(i, id2label.get(str(i), "")))
        if lab == "" or lab.upper().startswith("LABEL_"):
            levels.append(i + 1)
            continue
        s = "".join(ch for ch in lab if ch.isdigit() or ch == "-")
        levels.append(int(s) if s not in ("", "-") else i + 1)
    return np.asarray(levels, dtype=float)


@torch.no_grad()
def run_checkpoint(repo: str, texts, head: str, max_len: int = 128, batch_size: int = 32,
                   verbose: bool = True) -> np.ndarray:
    """Return one continuous score per input text (level space). Runs on MPS
    when available; batches are length-sorted to cut padding, then unsorted back
    to the caller's order. Scores are generated fresh (no external byte-parity
    reference), so device numerical differences are immaterial as long as a
    member's train/val/test all use the same device (they do)."""
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).eval().to(DEVICE)
    cfg = model.config
    is_reg = (cfg.num_labels == 1) or head == "reg"
    levels = None if is_reg else _level_vector(cfg)
    if verbose:
        print(f"    {repo}: device={DEVICE} num_labels={cfg.num_labels} reg={is_reg} "
              f"id2label={getattr(cfg,'id2label',None)}", flush=True)

    texts = [str(t) for t in texts]
    order = np.argsort([len(t) for t in texts], kind="stable")  # short -> long
    out = np.empty(len(texts), dtype=float)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        chunk = [texts[i] for i in idx]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(**enc).logits.float().cpu()
        if is_reg:
            out[idx] = logits.squeeze(-1).numpy()
        else:
            out[idx] = (torch.softmax(logits, dim=-1).numpy() @ levels)
    del model, tok
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return out
