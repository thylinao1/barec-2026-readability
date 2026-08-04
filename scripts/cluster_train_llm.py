"""LoRA regression fine-tune of a decoder LLM as an Open-track ensemble member.

Same contract as cluster_train.py: writes {name}_validation.npy / {name}_test.npy
(continuous scores, reg convention) plus the LoRA adapter under runs/<name>/adapter.
Blind inference happens cluster-side via cluster_infer_llm.py (the model is too big
for the 8GB Mac); all three splits therefore share ONE code path on ONE machine,
which satisfies the weights-match-scores gate by construction.

  python cluster_train_llm.py --backbone Qwen/Qwen2.5-7B-Instruct \
      --name qwen7b-reg --lr 1e-4 --epochs 2
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
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--data-dir", default=os.path.expanduser("~/barec/data"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/barec/runs"))
    return ap.parse_args()


class DS(Dataset):
    def __init__(self, texts, tok, max_len, y=None):
        self.enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)
        self.y = y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.y is not None:
            item["labels"] = torch.tensor(float(self.y[i]))
        return item


class RegTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        y = inputs.pop("labels")
        out = model(**inputs)
        loss = F.mse_loss(out.logits.squeeze(-1).float(), y.float())
        return (loss, out) if return_outputs else loss


@torch.no_grad()
def score(model, tok, texts, max_len, batch=32):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(list(texts[s:s + batch]), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(dev)
        out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    a = parse_args()
    set_seed(a.seed)
    t0 = time.time()
    run_dir = Path(a.out_dir) / a.name
    run_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_parquet(f"{a.data_dir}/train-00000-of-00001.parquet")
    va = pd.read_parquet(f"{a.data_dir}/validation-00000-of-00001.parquet")
    te = pd.read_parquet(f"{a.data_dir}/test-00000-of-00001.parquet")
    ycol = "Readability_Level_19"

    tok = AutoTokenizer.from_pretrained(a.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        a.backbone, num_labels=1, problem_type="regression",
        torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.pad_token_id = tok.pad_token_id
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=a.lora_r, lora_alpha=2 * a.lora_r,
                      lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(run_dir / "hf_out"),
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, num_train_epochs=a.epochs, seed=a.seed,
        bf16=True, report_to="none", logging_steps=100,
        save_strategy="no", warmup_ratio=0.03, dataloader_num_workers=2,
        gradient_checkpointing=True)
    RegTrainer(model=model, args=args,
               train_dataset=DS(tr["Sentence"].fillna("").tolist(), tok, a.max_len,
                                tr[ycol].astype(float).values),
               data_collator=DataCollatorWithPadding(tokenizer=tok)).train()

    sv = score(model, tok, va["Sentence"].fillna("").tolist(), a.max_len)
    st = score(model, tok, te["Sentence"].fillna("").tolist(), a.max_len)
    np.save(run_dir / f"{a.name}_validation.npy", sv)
    np.save(run_dir / f"{a.name}_test.npy", st)
    model.save_pretrained(run_dir / "adapter")
    tok.save_pretrained(run_dir / "adapter")

    from sklearn.metrics import cohen_kappa_score
    vq = cohen_kappa_score(va[ycol].astype(int),
                           np.clip(np.round(sv), 1, 19).astype(int), weights="quadratic")
    tq = cohen_kappa_score(te[ycol].astype(int),
                           np.clip(np.round(st), 1, 19).astype(int), weights="quadratic")
    meta = dict(vars(a), loss="reg", input="raw",
                val_naive_qwk=round(float(vq), 5), test_naive_qwk=round(float(tq), 5),
                minutes=round((time.time() - t0) / 60, 1))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "DONE").write_text("ok\n")
    print(f"[{a.name}] DONE {meta['minutes']}m val={vq:.4f} test={tq:.4f}", flush=True)


if __name__ == "__main__":
    main()
