"""Land selected fleet members: pull weights from the cluster, verify, score blind.

Per member (names as in fleet_manifest.tsv):
  1. rsync soc:~/barec/runs/<name>/model -> models/fleet/<name>/   (skips if present)
  2. GATE 1: config head shape matches the loss (reg->1, corn->18, else 19)
  3. GATE 2: local val inference on the SAME text regime the cluster trained on
     (raw Sentence / gold Word / gold D3Tok) must correlate > 0.99 with the
     cluster's {name}_validation.npy -- proves the weights match the scores.
  4. Blend-side scores, one local code path (scoring convention identical to
     cluster_train.py: reg=logit, corn=1+sum cumprod sigmoid, else posterior mean):
       val:   raw/word -> gate scores reused (same regime as blend standardization)
              d3tok    -> RECOMPUTED on my-D3Tok val text (/tmp/val_d3tok_mine.csv),
                          the proven reg_myd3 pattern (HANDOFF 0e)
       blind: raw -> Sentence | word -> /tmp/blind_word_mine.csv | d3tok ->
              /tmp/blind_d3tok_mine.csv
     Saved to artifacts/scores/fleet/{name}_val.npy + {name}_blind.npy for the blender.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/fleet_land.py \
        arabertv02-ce-word-s1 camelbert-corn-word-s42 ...
"""
import csv
import gc
import json
import subprocess
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
MODELS = ROOT / "models" / "fleet"
FLEET_RUNS = ROOT / "artifacts" / "fleet" / "runs"
OUT = ROOT / "artifacts" / "scores" / "fleet"
BLIND = ROOT / "data/raw/barec-blindtest-sent/data/train-00000-of-00001.parquet"
LEVELS = np.arange(1, 20, dtype=float)
MAX_LEN = 128
HEAD_FOR_LOSS = {"reg": 1, "softqwk": 1, "corn": 18, "ce": 19, "soft": 19, "focal": 19}


def cmap(path, col):
    return {r["ID"]: r[col] for r in csv.DictReader(open(path))}


@torch.no_grad()
def infer(model, tok, texts, loss):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty(len(texts), float)
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        lg = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits.float().cpu()
        if loss in ("reg", "softqwk"):
            out[idx] = lg.squeeze(-1).numpy()
        elif loss == "corn":
            out[idx] = (1.0 + torch.cumprod(torch.sigmoid(lg), dim=-1).sum(-1)).numpy()
        else:
            out[idx] = torch.softmax(lg, -1).numpy() @ LEVELS
    return out


def texts_for(regime, split, ids=None):
    """split in {validation, blind}; regime in {raw, word, d3tok}.
    validation uses GOLD word/d3tok (cluster parity); blind uses my-Word/my-D3Tok."""
    if split == "validation":
        df = load_2026("validation")
        if regime == "raw":
            return [s or "" for s in df["Sentence"].to_list()], [str(i) for i in df[ID].to_list()]
        gold = pl.read_parquet(ROOT / "data/processed/d3tok_validation.parquet")
        col = {"word": "Word", "d3tok": "D3Tok"}[regime]
        m = dict(zip([str(i) for i in gold[ID].to_list()], gold[col].to_list()))
        ids_ = [str(i) for i in df[ID].to_list()]
        return [m.get(i, "") or "" for i in ids_], ids_
    # blind
    b = pl.read_parquet(BLIND)
    bids = [str(i) for i in b[ID].to_list()]
    if regime == "raw":
        return [s or "" for s in b["Sentence"].to_list()], bids
    src = {"word": "/tmp/blind_word_mine.csv", "d3tok": "/tmp/blind_d3tok_mine.csv"}[regime]
    m = cmap(src, {"word": "Word", "d3tok": "D3Tok"}[regime])
    return [m.get(i, "") for i in bids], bids


def land(name):
    run = FLEET_RUNS / name
    meta = json.loads((run / "meta.json").read_text())
    loss, regime = meta["loss"], meta["input"]
    mdir = MODELS / name
    if not (mdir / "config.json").exists():
        mdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-az", "-e", "ssh",
                        f"soc:~/barec/runs/{name}/model/", str(mdir) + "/"], check=True)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).eval().to(DEVICE)

    nl = model.config.num_labels
    want = HEAD_FOR_LOSS[loss]
    assert nl == want, f"{name}: GATE1 FAIL num_labels={nl}, want {want} for loss={loss}"

    vx, _ = texts_for(regime, "validation")
    sv = infer(model, tok, vx, loss)
    ref = np.load(run / f"{name}_validation.npy").ravel()
    c = np.corrcoef(sv, ref)[0, 1]
    print(f"  [{name}] GATE2 corr vs cluster val npy: {c:.6f}"
          f"{'  OK' if c > 0.99 else '  !! FAIL'}")
    assert c > 0.99, f"{name}: GATE2 FAIL corr={c:.4f} (wrong weights?)"

    if regime == "d3tok":  # blend val must be my-D3Tok (reg_myd3 pattern)
        myd3 = cmap("/tmp/val_d3tok_mine.csv", "D3Tok")
        vdf = load_2026("validation")
        vids = [str(i) for i in vdf[ID].to_list()]
        sv = infer(model, tok, [myd3.get(i, "") for i in vids], loss)

    bx, _ = texts_for(regime, "blind")
    sb = infer(model, tok, bx, loss)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}_val.npy", sv)
    np.save(OUT / f"{name}_blind.npy", sb)
    print(f"  [{name}] landed: val mean={sv.mean():.2f} blind mean={sb.mean():.2f} "
          f"-> artifacts/scores/fleet/")
    del model, tok
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


def main():
    names = sys.argv[1:]
    if not names:
        sys.exit("usage: fleet_land.py <member-name> [...]  (names from fleet_manifest.tsv)")
    for n in names:
        land(n)
    print("\ndone. blend with blind_blend.py (run --gate first).")


if __name__ == "__main__":
    main()
