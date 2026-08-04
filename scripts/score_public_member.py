"""Score a PUBLIC HF checkpoint as a candidate ensemble member (val + blind).

Same conventions as scripts/blind_cache_scores.py: classification heads emit the
posterior mean softmax(logits) @ levels (id2label digits parsed for level order,
LABEL_i -> i+1), regression heads emit the raw logit. Outputs go to
artifacts/scores/fleet/{name}_val.npy + {name}_blind.npy, where blind_blend.py
auto-discovers them.

Input regimes (self-consistency rule -- val and blind must share one regime):
  raw   -> Sentence (val parquet / blind parquet)
  word  -> GOLD Word val (processed parquet) + my-Word blind (proven wordCE pattern)
  d3tok -> my-D3Tok val (/tmp/val_d3tok_mine.csv) + my-D3Tok blind (reg_myd3 pattern)

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/score_public_member.py \
        --repo Monda/bert-base-arabertv2_D3Tok_EMD_19levels --input d3tok --name monda_d3tok_emd \
        [--tok-repo OTHER/tokenizer] [--val-only] [--evict]
"""
import argparse
import csv
import gc
import shutil
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
OUT = ROOT / "artifacts" / "scores" / "fleet"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"
MAX_LEN = 128


def _levels(cfg):
    n = cfg.num_labels
    id2 = getattr(cfg, "id2label", None) or {}
    out = []
    for i in range(n):
        lab = str(id2.get(i, id2.get(str(i), "")))
        out.append(i + 1 if (lab == "" or lab.upper().startswith("LABEL_"))
                   else (int("".join(c for c in lab if c.isdigit())) if any(c.isdigit() for c in lab) else i + 1))
    return np.asarray(out, float)


def cmap(path, col):
    return {r["ID"]: r[col] for r in csv.DictReader(open(path))}


@torch.no_grad()
def infer(model, tok, texts, lv):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty(len(texts), float)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        lg = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits.float().cpu()
        out[idx] = lg.squeeze(-1).numpy() if lv is None else (torch.softmax(lg, -1).numpy() @ lv)
    return out


def get_texts(regime, split):
    if split == "validation":
        df = load_2026("validation")
        ids = [str(i) for i in df[ID].to_list()]
        if regime == "raw":
            return [s or "" for s in df["Sentence"].to_list()]
        if regime == "word":
            gold = pl.read_parquet(ROOT / "data/processed/d3tok_validation.parquet")
            m = dict(zip([str(i) for i in gold[ID].to_list()], gold["Word"].to_list()))
            return [m.get(i, "") or "" for i in ids]
        if regime in ("lex", "d3lex"):
            m = cmap("/tmp/val_lex_mine.csv", {"lex": "Lex", "d3lex": "D3Lex"}[regime])
            return [m.get(i, "") for i in ids]
        m = cmap("/tmp/val_d3tok_mine.csv", "D3Tok")
        return [m.get(i, "") for i in ids]
    b = pl.read_parquet(BLIND)
    bids = [str(i) for i in b[ID].to_list()]
    if regime == "raw":
        return [s or "" for s in b["Sentence"].to_list()]
    src, col = {"word": ("/tmp/blind_word_mine.csv", "Word"),
                "d3tok": ("/tmp/blind_d3tok_mine.csv", "D3Tok"),
                "lex": ("/tmp/blind_lex_mine.csv", "Lex"),
                "d3lex": ("/tmp/blind_lex_mine.csv", "D3Lex")}[regime]
    m = cmap(src, col)
    return [m.get(i, "") for i in bids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tok-repo", default=None)
    ap.add_argument("--input", required=True,
                    choices=["raw", "word", "d3tok", "lex", "d3lex"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--evict", action="store_true", help="purge HF cache for repo after scoring")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tok_repo or a.repo)
    model = AutoModelForSequenceClassification.from_pretrained(a.repo).eval().to(DEVICE)
    nl = model.config.num_labels
    lv = None if nl == 1 else _levels(model.config)
    print(f"[{a.name}] {a.repo} num_labels={nl} "
          f"levels={'reg' if lv is None else lv[:3].astype(int).tolist() + ['...']} input={a.input}")

    OUT.mkdir(parents=True, exist_ok=True)
    sv = infer(model, tok, get_texts(a.input, "validation"), lv)
    np.save(OUT / f"{a.name}_val.npy", sv)
    from sklearn.metrics import cohen_kappa_score
    yva = np.array(load_2026("validation")["Readability_Level_19"].to_list(), int)
    naive = cohen_kappa_score(yva, np.clip(np.round(sv), 1, 19).astype(int),
                              weights="quadratic") * 100
    # CONTAMINATION SCREEN: our members plateau ~76-85 naive on clean val. Anything
    # higher means the checkpoint saw validation rows in training (the v02_Sentence_CE
    # case measured 94.2). Such a member poisons the val-tuned rounder and blend
    # selection and MUST NOT be blended.
    flag = "  !! CONTAMINATED-VAL — DO NOT BLEND" if naive > 86 else ""
    print(f"  val:   n={len(sv)} mean={sv.mean():.3f} sd={sv.std():.3f} "
          f"naiveQWK={naive:.2f}{flag}")
    if naive > 86:
        (OUT / f"{a.name}_val.npy").rename(OUT / f"{a.name}_val.npy.CONTAMINATED")
        print(f"  quarantined {a.name}_val.npy -> .CONTAMINATED (blender will not see it)")
        return
    if not a.val_only:
        sb = infer(model, tok, get_texts(a.input, "blind"), lv)
        np.save(OUT / f"{a.name}_blind.npy", sb)
        print(f"  blind: n={len(sb)} mean={sb.mean():.3f} sd={sb.std():.3f}")

    del model, tok
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    if a.evict:
        for d in (Path.home() / ".cache/huggingface/hub").glob(
                f"models--{a.repo.replace('/', '--')}"):
            shutil.rmtree(d, ignore_errors=True)
            print(f"  evicted {d.name}")


if __name__ == "__main__":
    main()
