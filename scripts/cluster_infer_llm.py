"""Score a parquet of sentences with a trained LoRA member (cluster-side).

Used for BLIND inference of LLM members that cannot run on the 8GB Mac. The blind
parquet lives in ~/barec-private (chmod 700) and is DELETED after scores land.

  python cluster_infer_llm.py --backbone Qwen/Qwen2.5-7B-Instruct \
      --adapter ~/barec/runs/qwen7b-reg/adapter \
      --parquet ~/barec-private/blind.parquet --out ~/barec/runs/qwen7b-reg/blind.npy
"""
import argparse

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=128)
    a = ap.parse_args()

    df = pd.read_parquet(a.parquet)
    texts = df["Sentence"].fillna("").tolist()
    tok = AutoTokenizer.from_pretrained(a.adapter)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.backbone, num_labels=1, problem_type="regression",
        torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, a.adapter).eval().cuda()

    out = []
    with torch.no_grad():
        for s in range(0, len(texts), 32):
            enc = tok(texts[s:s + 32], truncation=True, max_length=a.max_len,
                      padding=True, return_tensors="pt").to("cuda")
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    sc = np.concatenate(out)
    np.save(a.out, sc)
    print(f"scored {len(sc)} rows -> {a.out}  mean={sc.mean():.3f} sd={sc.std():.3f}")


if __name__ == "__main__":
    main()
