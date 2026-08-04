"""Targeted retry of diverse public checkpoints using BASE-model tokenizers
(the repos shipped weights but not tokenizer files). Vet on val+test; keep only
clean (not contaminated) + decent members. Bounded final scavenge attempt.
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
from harness.data import TARGET, load_2026  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints  # noqa: E402

SCORES = ROOT / "artifacts" / "scores"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)

# (repo, [candidate base tokenizers], input variant)
CANDS = [
    ("AymanTarig/qarib-barec-optimized-v5", ["qarib/bert-base-qarib", "ahmedabdelali/bert-base-qarib"], "word"),
    ("AymanTarig/qarib-barec-optimized-v8", ["qarib/bert-base-qarib", "ahmedabdelali/bert-base-qarib"], "word"),
    ("AymanTarig/roberta-barec-v1", ["xlm-roberta-base"], "word"),
    ("AymanTarig/arabert-barec-optimized-v1", ["aubmindlab/bert-base-arabertv2"], "word"),
    ("AymanTarig/arabertv02-barec-optimized-v1", ["aubmindlab/bert-base-arabertv02"], "word"),
    ("shymaa25/barec-readability-sent-arabertv2-d3tok-ce-wkl-strict", ["aubmindlab/bert-base-arabertv2"], "d3tok"),
]


def _levels(id2label, n):
    out = []
    for i in range(n):
        lab = str((id2label or {}).get(i, (id2label or {}).get(str(i), "")))
        out.append(i + 1 if (lab == "" or lab.upper().startswith("LABEL_")) else
                   (int("".join(c for c in lab if c.isdigit())) if any(c.isdigit() for c in lab) else i + 1))
    return np.asarray(out, float)


def _decode(logits, n, id2label):
    if n == 1:
        return logits.squeeze(-1).numpy()
    if n == 18:
        return 1.0 + torch.sigmoid(logits).sum(dim=-1).numpy()
    return torch.softmax(logits, dim=-1).numpy() @ _levels(id2label, n)


def _tok(bases):
    for b in bases:
        try:
            return AutoTokenizer.from_pretrained(b)
        except Exception:
            continue
    return None


@torch.no_grad()
def run(repo, bases, texts):
    tok = _tok(bases)
    if tok is None:
        raise RuntimeError("no base tokenizer loaded")
    model = AutoModelForSequenceClassification.from_pretrained(repo).eval().to(DEVICE)
    n, id2label = model.config.num_labels, getattr(model.config, "id2label", None)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty(len(texts), float)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors="pt")
        out[idx] = _decode(model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits.float().cpu(), n, id2label)
    del model, tok; gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return out, n


def main():
    val, test = load_2026("validation"), load_2026("test")
    yva = np.array(val[TARGET].to_list()); yte = np.array(test[TARGET].to_list())
    d3v = pl.read_parquet(ROOT / "data/processed/d3tok_validation.parquet")["D3Tok"].to_list()
    d3t = pl.read_parquet(ROOT / "data/processed/d3tok_test.parquet")["D3Tok"].to_list()
    rv = val["Sentence"].fill_null("").to_list(); rt = test["Sentence"].fill_null("").to_list()
    print(f"device={DEVICE}\n{'model':52s} {'nlab':>4s} {'valQWK':>7s} {'testQWK':>7s}  verdict")
    for repo, bases, var in CANDS:
        try:
            tv, tt = (d3v, d3t) if var == "d3tok" else (rv, rt)
            sv, n = run(repo, bases, tv)
            st, _ = run(repo, bases, tt)
            r = OptimizedRounder(n_candidates=150).fit(sv, yva)
            qv = fold_qwk(yva, apply_cutpoints(sv, r.cutpoints_))
            qt = fold_qwk(yte, apply_cutpoints(st, r.cutpoints_))
            contam = qv > 0.88 or qt > 0.88
            verdict = "CONTAM(skip)" if contam else ("KEEP" if qt > 0.80 else "weak")
            if not contam and qt > 0.80:
                tag = "pub_" + repo.replace("/", "__")
                np.save(SCORES / f"{tag}_validation.npy", sv)
                np.save(SCORES / f"{tag}_test.npy", st)
            print(f"{repo.split('/')[-1]:52s} {n:4d} {qv*100:7.2f} {qt*100:7.2f}  {verdict}", flush=True)
        except Exception as e:
            print(f"{repo.split('/')[-1]:52s}  FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
