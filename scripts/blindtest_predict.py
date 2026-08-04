"""Generate BLIND TEST predictions with the frozen val-tuned pipeline.

The winning 3-member blend, all locally runnable:
  reg   = CAMeL-Lab/readability-arabertv2-d3tok-reg   input=my-D3Tok  (camel-tools)
  wordCE= CAMeL-Lab/readability-arabertv02-word-CE     input=my-Word   (camel-tools)
  qarib = AymanTarig/qarib-barec-optimized-v8          input=raw Sentence (base tokenizer)

Each blind member is standardized with the SAME member's VAL stats (cached val
npy), equal-weight blended; the OptimizedRounder is fit on the VAL blend (nothing
re-tuned on blind); cutpoints applied to the blind blend. Writes prediction.zip.

Preprocessing CSVs (from the camel venv): /tmp/blind_word_mine.csv, /tmp/blind_d3tok_mine.csv
    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/blindtest_predict.py [--wordonly]
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
from harness.data import ID, TARGET, load_2026  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints  # noqa: E402
from harness.submission import write_codabench_zip  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)
SCORES = ROOT / "artifacts" / "scores"
OUT = ROOT / "submissions" / "blindtest"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"

# (name, repo, tokenizer_repo_or_None, head, input_source, val_score_override_or_None)
# val_score_override: use MY-D3Tok val scores for the reg member so its standardization
# and the rounder match what the blind set sees (my D3Tok, not the license-gated gold).
FULL = [
    ("arabertv2-d3tok-reg", "CAMeL-Lab/readability-arabertv2-d3tok-reg", None, "reg", "d3tok", "/tmp/reg_myD3_val.npy"),
    ("arabertv02-word-CE", "CAMeL-Lab/readability-arabertv02-word-CE", None, "ce", "word", None),
    ("pub_AymanTarig__qarib-barec-optimized-v8", "AymanTarig/qarib-barec-optimized-v8",
     "qarib/bert-base-qarib", "ce", "raw", None),
]
WORDONLY = [
    ("arabertv02-word-CE", "CAMeL-Lab/readability-arabertv02-word-CE", None, "ce", "word", None),
    ("camelbert-word-CE", "CAMeL-Lab/readability-camelbert-word-CE", None, "ce", "word", None),
    ("pub_AymanTarig__qarib-barec-optimized-v8", "AymanTarig/qarib-barec-optimized-v8",
     "qarib/bert-base-qarib", "ce", "raw", None),
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
def infer(repo, tok_repo, head, texts):
    tok = AutoTokenizer.from_pretrained(tok_repo or repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).eval().to(DEVICE)
    n = model.config.num_labels
    reg = (n == 1) or head == "reg"
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


def read_csv_map(path, col):
    return {r["ID"]: r[col] for r in csv.DictReader(open(path))}


def main():
    members = WORDONLY if "--wordonly" in sys.argv else FULL
    OUT.mkdir(parents=True, exist_ok=True)
    val = load_2026("validation")
    yva = np.array(val[TARGET].to_list(), dtype=int)
    blind = pl.read_parquet(BLIND)
    bids = [str(i) for i in blind[ID].to_list()]
    raw = {str(i): (s or "") for i, s in zip(blind[ID].to_list(), blind["Sentence"].to_list())}
    myword = read_csv_map("/tmp/blind_word_mine.csv", "Word")
    myd3 = read_csv_map("/tmp/blind_d3tok_mine.csv", "D3Tok") if any(m[4] == "d3tok" for m in members) else {}

    def blind_texts(src):
        return [({"raw": raw, "word": myword, "d3tok": myd3}[src]).get(i, "") for i in bids]

    Zva, Zte = [], []
    print(f"members: {[m[0] for m in members]}  device={DEVICE}", flush=True)
    for name, repo, tokr, head, src, val_override in members:
        vscore = np.load(val_override) if val_override else np.load(SCORES / f"{name}_validation.npy")
        mu, sd = float(vscore.mean()), float(vscore.std()) or 1.0
        bscore = infer(repo, tokr, head, blind_texts(src))
        Zva.append((vscore - mu) / sd)
        Zte.append((bscore - mu) / sd)
        print(f"  {name}: blind n={len(bscore)} mean={bscore.mean():.2f} range=[{bscore.min():.2f},{bscore.max():.2f}]", flush=True)

    sva = sum(Zva) / len(Zva)
    ste = sum(Zte) / len(Zte)
    rounder = OptimizedRounder(n_candidates=300).fit(sva, yva)  # fit on VAL only
    val_qwk = fold_qwk(yva, apply_cutpoints(sva, rounder.cutpoints_))
    bpred = apply_cutpoints(ste, rounder.cutpoints_)
    print(f"\nval blend QWK (in-sample rounder): {val_qwk*100:.3f}")
    print(f"blind prediction distribution: min={bpred.min()} max={bpred.max()} "
          f"mean={bpred.mean():.2f} n={len(bpred)}", flush=True)
    zp = write_codabench_zip(OUT, bids, bpred)
    print(f"wrote {zp}  ({len(bids)} blind predictions)")


if __name__ == "__main__":
    main()
