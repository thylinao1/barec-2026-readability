"""Phase 3 - free Kaggle T4 fine-tune of a diverse backbone as an ensemble member.

Paste into a Kaggle notebook (GPU T4 on, Internet on). Trains ONE regression head
on all of train, predicts validation + test, exports 2 .npy. That is all the
blender needs: it tunes on val (grouped-OOF within val) and evaluates on test, so
no train out-of-fold is required. ~30 min per member on a T4.

Regression head on RAW level 1..19 (MSE) - the +3.1-QWK-over-CE lever, and NOT
the sigmoid-normalized target (it saturates). Deterministic (fixed seed).

Set the CONFIG block, Run All, then download the two .npy files and drop them in
~/Developer/barec-2026-sentence-open/artifacts/scores/ . blend_all.py auto-folds
any member named to match; for a custom name, add it to BASE in blend_all.py.
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
    """If the backbone was uploaded as a Kaggle dataset (recommended - avoids the
    HF download hang), return its folder. Looks for a config.json next to model
    weights under /kaggle/input. When several model datasets are attached, prefer
    the folder whose path matches the backbone name (e.g. 'araelectra')."""
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
    """Prefer a locally-uploaded model; else download from HF with retries
    (snapshot_download resumes partial files so a stall recovers, not hangs)."""
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

# ---------------- CONFIG (edit per member) ----------------
BACKBONE = "UBC-NLP/MARBERTv2"          # then: aubmindlab/araelectra-base-discriminator, etc.
OUT_NAME = "marbertv2-reg"              # names the output files
TEXT_COL = "Sentence"                   # raw text (MARBERT/AraELECTRA expect raw, not D3Tok)
MAX_LEN, BATCH, LR, EPOCHS, SEED = 128, 16, 2e-5, 5, 42
# ----------------------------------------------------------

torch.manual_seed(SEED); np.random.seed(SEED)


def read(split):
    # auto-discover under /kaggle/input regardless of dataset slug / subfolder
    hits = glob.glob(f"/kaggle/input/**/{split}-00000-of-00001.parquet", recursive=True)
    assert hits, (f"'{split}' parquet not found under /kaggle/input. Did you 'Add Input' "
                  f"the dataset? Present: {glob.glob('/kaggle/input/**/*', recursive=True)[:20]}")
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
    local = fetch(BACKBONE)  # download with retries and resume, then load locally
    tok = AutoTokenizer.from_pretrained(local)
    model = AutoModelForSequenceClassification.from_pretrained(
        local, num_labels=1, problem_type="regression")
    args = TrainingArguments(
        output_dir="/kaggle/working/out", per_device_train_batch_size=BATCH,
        learning_rate=LR, num_train_epochs=EPOCHS, seed=SEED, fp16=True,
        report_to="none", logging_steps=200, save_strategy="no",
    )
    ds = DS(tr[TEXT_COL].values, tok, tr["Readability_Level_19"].values.astype(float))
    collator = DataCollatorWithPadding(tokenizer=tok)  # pad each batch (DS tokenizes unpadded)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()

    np.save(f"/kaggle/working/{OUT_NAME}_validation.npy", predict(model, va[TEXT_COL].values, tok))
    np.save(f"/kaggle/working/{OUT_NAME}_test.npy", predict(model, te[TEXT_COL].values, tok))
    print(f"DONE: download {OUT_NAME}_validation.npy and {OUT_NAME}_test.npy from the Output tab")


if __name__ == "__main__":
    main()
