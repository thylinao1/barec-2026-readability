"""MARBERTv2 fine-tune + CHECKPOINT EXPORT (paste this whole file into a fresh Kaggle notebook).

Identical to kaggle_finetune.py except it also exports the trained weights to
/kaggle/working/model, which the original never did. That omission is why the fine-tuned
MARBERTv2 could not be used to score the blind set.

Kaggle setup: Accelerator = GPU T4 x2, Internet = On.
Add Input (both):  barec-2026-open      (the BAREC parquet data)
                   marbertv2-model      (the MARBERTv2 base weights)
Then Run All (~30 min). Download from the Output tab:
    model/  +  marbertv2-reg_validation.npy  +  marbertv2-reg_test.npy
"""
import glob
import os
import time

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # avoid the unauthenticated-rate-limit hang

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def find_local_model(hint=""):
    """Prefer a backbone uploaded as a Kaggle dataset (avoids the HF download hang)."""
    cands = []
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        if os.path.exists(os.path.join(d, "pytorch_model.bin")) or glob.glob(f"{d}/*.safetensors"):
            cands.append(d)
    key = hint.split("/")[-1].split("-")[0].lower() if hint else ""
    for d in cands:
        if key and key in d.lower():
            return d
    return cands[0] if cands else None


def fetch(repo, tries=6):
    local = find_local_model(repo)
    if local:
        print(f"using uploaded model at {local}", flush=True)
        return local
    for i in range(tries):
        try:
            return snapshot_download(repo)
        except Exception as e:
            print(f"download retry {i + 1}/{tries}: {e}", flush=True)
            time.sleep(15)
    raise RuntimeError(f"failed to download {repo}")

# ---------------- CONFIG (do NOT change - must match the cached scores) ----------------
BACKBONE = "UBC-NLP/MARBERTv2"
OUT_NAME = "marbertv2-reg"
TEXT_COL = "Sentence"                   # raw text - MARBERTv2 was fine-tuned on raw, not D3Tok
MAX_LEN, BATCH, LR, EPOCHS, SEED = 128, 16, 2e-5, 5, 42
# ---------------------------------------------------------------------------------------

torch.manual_seed(SEED); np.random.seed(SEED)


def read(split):
    hits = glob.glob(f"/kaggle/input/**/{split}-00000-of-00001.parquet", recursive=True)
    assert hits, (f"'{split}' parquet not found under /kaggle/input. Did you 'Add Input' the "
                  f"barec-2026-open dataset? Present: {glob.glob('/kaggle/input/**/*', recursive=True)[:20]}")
    return pd.read_parquet(hits[0])


class DS(Dataset):
    def __init__(self, texts, tok, y=None):
        self.enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding=False)
        self.y = y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.y is not None:
            item["labels"] = torch.tensor(float(self.y[i]))
        return item


@torch.no_grad()
def predict(model, texts, tok):
    model.eval()
    out = []
    for s in range(0, len(texts), 64):
        enc = tok(list(texts[s:s + 64]), truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(model.device)
        out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    tr, va, te = read("train"), read("validation"), read("test")
    local = fetch(BACKBONE)
    tok = AutoTokenizer.from_pretrained(local)
    model = AutoModelForSequenceClassification.from_pretrained(
        local, num_labels=1, problem_type="regression")
    args = TrainingArguments(
        output_dir="/kaggle/working/out", per_device_train_batch_size=BATCH,
        learning_rate=LR, num_train_epochs=EPOCHS, seed=SEED, fp16=True,
        report_to="none", logging_steps=200, save_strategy="no",
    )
    ds = DS(tr[TEXT_COL].values, tok, tr["Readability_Level_19"].values.astype(float))
    collator = DataCollatorWithPadding(tokenizer=tok)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()

    np.save(f"/kaggle/working/{OUT_NAME}_validation.npy", predict(model, va[TEXT_COL].values, tok))
    np.save(f"/kaggle/working/{OUT_NAME}_test.npy", predict(model, te[TEXT_COL].values, tok))

    # ---- THE FIX: export the trained weights (fp32; do NOT .half() - it breaks standardization)
    model.save_pretrained("/kaggle/working/model")
    tok.save_pretrained("/kaggle/working/model")
    print("saved fine-tuned checkpoint to /kaggle/working/model", flush=True)
    for f in sorted(os.listdir("/kaggle/working/model")):
        print("   ", f, os.path.getsize(f"/kaggle/working/model/{f}"), "bytes", flush=True)

    print(f"DONE. Download from the Output tab: model/ + {OUT_NAME}_validation.npy "
          f"+ {OUT_NAME}_test.npy", flush=True)


if __name__ == "__main__":
    main()
