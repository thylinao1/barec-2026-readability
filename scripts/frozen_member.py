"""Path 1 - a diverse ensemble member from a FROZEN backbone, entirely on the M2.

No fine-tuning (respects the never-train-locally rule): run the backbone in
inference only (masked mean-pool of the last hidden state), then train a light
Ridge head on top. The head is trained on train (a raw pretrained backbone's
frozen features are not label-memorized, unlike the fine-tuned checkpoints), and
predicts val + test for the blender. Weaker than a Kaggle fine-tune but free,
safe, and Kaggle-independent.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/frozen_member.py <local_model_dir> <member_name> [d3tok|word]
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.data import ID, TARGET, load_2026  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints  # noqa: E402

SCORES = ROOT / "artifacts" / "scores"
FEAT = ROOT / "artifacts" / "feats"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(4)


@torch.no_grad()
def features(model, tok, texts, bs=32):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        hid = model(**enc).last_hidden_state              # (B,T,H)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B,T,1)
        pooled = (hid * mask).sum(1) / mask.sum(1).clamp(min=1)  # masked mean
        out[idx] = pooled.float().cpu().numpy()
    return out


def get_feats(model, tok, split, texts, name):
    FEAT.mkdir(parents=True, exist_ok=True)
    p = FEAT / f"{name}_{split}.npy"
    if p.exists():
        return np.load(p)
    t = time.time()
    f = features(model, tok, texts)
    np.save(p, f)
    print(f"  {split}: features {f.shape} ({time.time()-t:.0f}s)", flush=True)
    return f


def main():
    model_dir, name = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else "word"
    SCORES.mkdir(parents=True, exist_ok=True)
    train, val, test = load_2026("train"), load_2026("validation"), load_2026("test")
    y = np.array(train[TARGET].to_list()); yva = np.array(val[TARGET].to_list()); yte = np.array(test[TARGET].to_list())

    if variant == "d3tok":
        col = lambda s: pl.read_parquet(ROOT / f"data/processed/d3tok_{s}.parquet")["D3Tok"].to_list()
        tr_t, va_t, te_t = col("train"), col("validation"), col("test")
    else:
        tr_t = train["Sentence"].fill_null("").to_list()
        va_t = val["Sentence"].fill_null("").to_list()
        te_t = test["Sentence"].fill_null("").to_list()

    print(f"device={DEVICE} model={model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval().to(DEVICE)
    Ftr = get_feats(model, tok, "train", tr_t, name)
    Fva = get_feats(model, tok, "validation", va_t, name)
    Fte = get_feats(model, tok, "test", te_t, name)
    del model; gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # light head: Ridge on frozen features (standardize implicitly via alpha)
    best = None
    for alpha in [1.0, 10.0, 50.0, 100.0]:
        head = Ridge(alpha=alpha).fit(Ftr, y.astype(float))
        sva = head.predict(Fva)
        r = OptimizedRounder(n_candidates=150).fit(sva, yva)
        qv = fold_qwk(yva, apply_cutpoints(sva, r.cutpoints_))
        if best is None or qv > best[0]:
            best = (qv, alpha, head)
    qv, alpha, head = best
    sva, ste = head.predict(Fva), head.predict(Fte)
    r = OptimizedRounder(n_candidates=200).fit(sva, yva)
    qt = fold_qwk(yte, apply_cutpoints(ste, r.cutpoints_))
    np.save(SCORES / f"{name}_validation.npy", sva)
    np.save(SCORES / f"{name}_test.npy", ste)
    print(f"\nfrozen member '{name}' (alpha={alpha}): val {qv*100:.2f}  test {qt*100:.2f}")
    print(f"saved {name}_validation.npy / {name}_test.npy  -> run scripts/blend_all.py to fold in")


if __name__ == "__main__":
    main()
