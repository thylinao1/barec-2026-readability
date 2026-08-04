"""NUS-cluster fine-tune of ONE ensemble member (BAREC 2026 sentence, 19 levels).

Adapted from notebooks/kaggle_finetune.py (the proven T4 recipe) with the July
mistake fixed: this script ALWAYS save_pretrained's the checkpoint AND writes
{name}_validation.npy / {name}_test.npy. The npy files hold CONTINUOUS member
scores in the exact convention of scripts/blind_cache_scores.py:
  reg          -> raw logit (level scale)
  ce/soft/focal-> posterior mean  softmax(logits) @ [1..19]
  corn         -> 1 + sum_k cumprod(sigmoid(logits))_k
so the local blender consumes them unchanged.

Input regimes (parity with the local pipeline):
  raw   -> Sentence column of the 2026 parquet
  word  -> gold Word column from data/processed/d3tok_<split>.parquet
  d3tok -> gold D3Tok column from the same (blind-side inference then uses
           my-D3Tok + my-D3Tok val standardization, the proven reg_myd3 pattern)

Usage (on a Slurm GPU node):
  python cluster_train.py --backbone aubmindlab/bert-base-arabertv02 \
      --loss ce --input word --seed 1 --name arabertv02-ce-word-s1
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

LEVELS = np.arange(1, 20, dtype=float)  # 19 readability levels
N_CLASS = 19
N_CUTS = 18


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--loss", required=True,
                    choices=["reg", "ce", "corn", "soft", "focal", "softqwk"])
    ap.add_argument("--input", required=True, choices=["raw", "word", "d3tok"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.0, help="warmup ratio (larges: 0.06)")
    ap.add_argument("--soft-sigma", type=float, default=1.0)
    ap.add_argument("--data-dir", default=os.path.expanduser("~/barec/data"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/barec/runs"))
    ap.add_argument("--smoke", action="store_true", help="30 steps only, for env sanity")
    return ap.parse_args()


def load_split(data_dir, split, input_regime):
    df = pd.read_parquet(f"{data_dir}/{split}-00000-of-00001.parquet")
    if input_regime == "raw":
        texts = df["Sentence"].fillna("").tolist()
    else:
        col = {"word": "Word", "d3tok": "D3Tok"}[input_regime]
        gold = pd.read_parquet(f"{data_dir}/d3tok_{split}.parquet")
        gold["ID"] = gold["ID"].astype(str)
        m = dict(zip(gold["ID"], gold[col].fillna("")))
        texts = [m.get(str(i), "") for i in df["ID"]]
        n_empty = sum(1 for t in texts if not t)
        assert n_empty < len(texts) * 0.01, f"{split}: {n_empty} rows missing {col}"
    y = df["Readability_Level_19"].astype(int).values if "Readability_Level_19" in df else None
    return texts, y


class DS(Dataset):
    def __init__(self, texts, tok, max_len, y=None):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
        self.y = y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.y is not None:
            item["labels"] = torch.tensor(float(self.y[i]))  # raw level 1..19; loss fn converts
        return item


class MemberTrainer(Trainer):
    """One compute_loss for all five losses. `labels` arrive as float levels 1..19."""

    def __init__(self, *a, loss_kind="reg", soft_sigma=1.0, focal_alpha=None, **kw):
        super().__init__(*a, **kw)
        self.loss_kind = loss_kind
        self.soft_sigma = soft_sigma
        self.focal_alpha = focal_alpha  # tensor [19] or None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        y = inputs.pop("labels")  # float levels 1..19
        out = model(**inputs)
        logits = out.logits
        if self.loss_kind == "reg":
            loss = F.mse_loss(logits.squeeze(-1), y)
        elif self.loss_kind == "softqwk":
            # differentiable weighted-kappa loss (STBW two-phase, phase 2): soft-assign
            # the continuous prediction to the 19 levels with a Gaussian window, build a
            # soft confusion matrix, minimize observed/expected quadratic disagreement
            yhat = logits.squeeze(-1)
            K = torch.arange(1, N_CLASS + 1, device=yhat.device, dtype=yhat.dtype)
            P = F.softmax(-((yhat.unsqueeze(1) - K) ** 2) / (2 * 1.0 ** 2), dim=1)
            Y = F.one_hot((y - 1).long(), N_CLASS).to(yhat.dtype)
            C = Y.t() @ P
            W = ((K.unsqueeze(0) - K.unsqueeze(1)) ** 2) / (N_CLASS - 1) ** 2
            E = Y.sum(0).unsqueeze(1) @ P.sum(0).unsqueeze(0) / y.numel()
            loss = (W * C).sum() / ((W * E).sum() + 1e-8)
        elif self.loss_kind == "ce":
            loss = F.cross_entropy(logits, (y - 1).long())
        elif self.loss_kind == "focal":
            logp = F.log_softmax(logits, dim=-1)
            idx = (y - 1).long()
            logp_t = logp[torch.arange(len(idx), device=idx.device), idx]
            p_t = logp_t.exp()
            alpha_t = self.focal_alpha.to(logits.device)[idx]
            loss = (-alpha_t * (1 - p_t) ** 2 * logp_t).mean()
        elif self.loss_kind == "soft":
            lv = torch.arange(1, N_CLASS + 1, device=logits.device, dtype=logits.dtype)
            q = F.softmax(-((lv.unsqueeze(0) - y.unsqueeze(1)) ** 2)
                          / (2 * self.soft_sigma ** 2), dim=-1)
            loss = -(q * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        elif self.loss_kind == "corn":
            total = logits.new_zeros(())
            count = 0
            for k0 in range(N_CUTS):
                mask = y > k0  # conditional subset: y > threshold-1
                n = int(mask.sum())
                if n == 0:
                    continue
                target = (y[mask] > (k0 + 1)).to(logits.dtype)
                total = total + F.binary_cross_entropy_with_logits(
                    logits[mask, k0], target, reduction="sum")
                count += n
            loss = total / max(count, 1)
        else:
            raise ValueError(self.loss_kind)
        return (loss, out) if return_outputs else loss


@torch.no_grad()
def score(model, tok, texts, loss_kind, max_len, batch=64):
    """Continuous member score, blind_cache_scores.py conventions."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(list(texts[s:s + batch]), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(dev)
        lg = model(**enc).logits.float()
        if loss_kind in ("reg", "softqwk"):
            out.append(lg.squeeze(-1).cpu().numpy())
        elif loss_kind == "corn":
            p_gt = torch.cumprod(torch.sigmoid(lg), dim=-1)  # P(y>k)
            out.append((1.0 + p_gt.sum(-1)).cpu().numpy())
        else:  # ce / soft / focal -> posterior mean
            p = torch.softmax(lg, dim=-1).cpu().numpy()
            out.append(p @ LEVELS)
    return np.concatenate(out)


def quick_qwk(y, s):
    """Rough sanity only (naive rounding, no OptimizedRounder). NOT a go/no-go."""
    from sklearn.metrics import cohen_kappa_score
    pred = np.clip(np.round(s), 1, 19).astype(int)
    return cohen_kappa_score(y, pred, weights="quadratic")


def main():
    a = parse_args()
    set_seed(a.seed)
    t0 = time.time()
    run_dir = Path(a.out_dir) / a.name
    run_dir.mkdir(parents=True, exist_ok=True)

    tr_x, tr_y = load_split(a.data_dir, "train", a.input)
    va_x, va_y = load_split(a.data_dir, "validation", a.input)
    te_x, te_y = load_split(a.data_dir, "test", a.input)
    if a.smoke:  # tiny eval slices so the smoke run finishes in minutes on any device
        va_x, va_y = va_x[:200], va_y[:200]
        te_x, te_y = te_x[:200], te_y[:200]
    print(f"[{a.name}] train={len(tr_x)} val={len(va_x)} test={len(te_x)} "
          f"loss={a.loss} input={a.input} seed={a.seed}", flush=True)

    num_labels = {"reg": 1, "softqwk": 1, "corn": N_CUTS}.get(a.loss, N_CLASS)
    tok = AutoTokenizer.from_pretrained(a.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.backbone, num_labels=num_labels,
        problem_type="regression" if a.loss == "reg" else None)

    focal_alpha = None
    if a.loss == "focal":
        freq = np.array([(np.asarray(tr_y) == k).mean() for k in range(1, 20)])
        w = 1.0 / np.maximum(freq, 1e-4)
        focal_alpha = torch.tensor(w / w.mean(), dtype=torch.float32)

    args = TrainingArguments(
        output_dir=str(run_dir / "hf_out"),
        per_device_train_batch_size=a.batch,
        learning_rate=a.lr, num_train_epochs=a.epochs, seed=a.seed,
        bf16=torch.cuda.is_available(), report_to="none", logging_steps=200,
        save_strategy="no", warmup_ratio=a.warmup, dataloader_num_workers=2,
        max_steps=30 if a.smoke else -1,
        use_cpu=a.smoke and not torch.cuda.is_available(),  # 8GB-Mac smoke safety
    )
    if a.loss == "softqwk":
        # STBW two-phase: MSE warm-start, then the differentiable kappa objective at a
        # lower LR and larger batch (batch-level kappa is noisy on small batches)
        args.num_train_epochs = max(a.epochs - 2, 1)
        MemberTrainer(model=model, args=args,
                      train_dataset=DS(tr_x, tok, a.max_len, tr_y),
                      data_collator=DataCollatorWithPadding(tokenizer=tok),
                      loss_kind="reg").train()
        args2 = TrainingArguments(
            output_dir=str(run_dir / "hf_out2"), per_device_train_batch_size=64,
            learning_rate=a.lr / 2, num_train_epochs=2, seed=a.seed,
            bf16=torch.cuda.is_available(), report_to="none", logging_steps=200,
            save_strategy="no", dataloader_num_workers=2,
            max_steps=10 if a.smoke else -1,
            use_cpu=a.smoke and not torch.cuda.is_available())
        MemberTrainer(model=model, args=args2,
                      train_dataset=DS(tr_x, tok, a.max_len, tr_y),
                      data_collator=DataCollatorWithPadding(tokenizer=tok),
                      loss_kind="softqwk").train()
    else:
        trainer = MemberTrainer(
            model=model, args=args,
            train_dataset=DS(tr_x, tok, a.max_len, tr_y),
            data_collator=DataCollatorWithPadding(tokenizer=tok),
            loss_kind=a.loss, soft_sigma=a.soft_sigma, focal_alpha=focal_alpha)
        trainer.train()

    sv = score(model, tok, va_x, a.loss, a.max_len)
    st = score(model, tok, te_x, a.loss, a.max_len)
    np.save(run_dir / f"{a.name}_validation.npy", sv)
    np.save(run_dir / f"{a.name}_test.npy", st)
    model.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "model")

    vq, tq = quick_qwk(va_y, sv), quick_qwk(te_y, st)
    meta = dict(vars(a), val_naive_qwk=round(float(vq), 5),
                test_naive_qwk=round(float(tq), 5),
                minutes=round((time.time() - t0) / 60, 1))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "DONE").write_text("ok\n")
    print(f"[{a.name}] DONE in {meta['minutes']}m  "
          f"val_naiveQWK={vq:.4f} test_naiveQWK={tq:.4f}", flush=True)


if __name__ == "__main__":
    main()
