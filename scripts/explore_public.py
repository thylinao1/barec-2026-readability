"""Vet additional PUBLIC BAREC checkpoints as free ensemble members.

Beyond the 4 CAMeL-Lab checkpoints, many teams uploaded fine-tuned models. This
inferences each on val + test (MPS, one at a time), decodes to level space, and
reports honest val + test QWK. CONTAMINATION GUARD: a model that scores ~90+ on
a split was trained on that split (memorized) -> its number is a mirage; keep
only models honest (~<88) on BOTH val and test so any blend gain is real and
transfers to the blind test. Good members' scores cache to pub_<tag>_<split>.npy.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/explore_public.py
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

from harness.data import ID, TARGET, load_2026  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints  # noqa: E402

SCORES = ROOT / "artifacts" / "scores"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)

CANDIDATES = [
    "AymanTarig/qarib-barec-optimized-v5",
    "AymanTarig/qarib-barec-optimized-v8",
    "AymanTarig/roberta-barec-v1",
    "AymanTarig/arabert-barec-optimized-v1",
    "AymanTarig/arabertv02-barec-optimized-v1",
    "MoT69420/barec-araelectra-coral-ordinal-regression-finetuned",
    "shymaa25/barec-readability-sent-arabertv2-d3tok-ce-wkl-strict",
    "bensapir/barec-base-finetune-sent",
]


def _levels(id2label, n):
    out = []
    for i in range(n):
        lab = str((id2label or {}).get(i, (id2label or {}).get(str(i), "")))
        if lab == "" or lab.upper().startswith("LABEL_"):
            out.append(i + 1)
        else:
            s = "".join(c for c in lab if c.isdigit() or c == "-")
            out.append(int(s) if s not in ("", "-") else i + 1)
    return np.asarray(out, float)


def _decode(logits, n, id2label):
    if n == 1:
        return logits.squeeze(-1).numpy()
    if n == 18:  # CORAL: K-1 cumulative P(y>k) -> expected level 1 + sum P
        return 1.0 + torch.sigmoid(logits).sum(dim=-1).numpy()
    p = torch.softmax(logits, dim=-1).numpy()
    return p @ _levels(id2label, n)


@torch.no_grad()
def run(repo, texts):
    tok = AutoTokenizer.from_pretrained(repo)
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


def tag(repo):
    return "pub_" + repo.replace("/", "__")


def main():
    val, test = load_2026("validation"), load_2026("test")
    yva = np.array(val[TARGET].to_list()); yte = np.array(test[TARGET].to_list())
    d3v = pl.read_parquet(ROOT / "data/processed/d3tok_validation.parquet")
    d3t = pl.read_parquet(ROOT / "data/processed/d3tok_test.parquet")
    raw_v = val["Sentence"].fill_null("").to_list(); raw_t = test["Sentence"].fill_null("").to_list()
    d3tok_v = d3v["D3Tok"].to_list(); d3tok_t = d3t["D3Tok"].to_list()

    print(f"device={DEVICE}")
    print(f"{'model':60s} {'nlab':>4s} {'valQWK':>7s} {'testQWK':>7s}  verdict")
    for repo in CANDIDATES:
        try:
            d3 = "d3tok" in repo.lower()
            sv, n = run(repo, (d3tok_v if d3 else raw_v))
            st, _ = run(repo, (d3tok_t if d3 else raw_t))
            r = OptimizedRounder(n_candidates=150).fit(sv, yva)   # fit on val, apply to test
            qv = fold_qwk(yva, apply_cutpoints(sv, r.cutpoints_))
            qt = fold_qwk(yte, apply_cutpoints(st, r.cutpoints_))
            contam = qv > 0.88 or qt > 0.88
            verdict = "CONTAM(skip)" if contam else ("keep" if qt > 0.80 else "weak")
            if not contam:
                np.save(SCORES / f"{tag(repo)}_validation.npy", sv)
                np.save(SCORES / f"{tag(repo)}_test.npy", st)
            print(f"{repo:60s} {n:4d} {qv*100:7.2f} {qt*100:7.2f}  {verdict}", flush=True)
        except Exception as e:
            print(f"{repo:60s}  FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
