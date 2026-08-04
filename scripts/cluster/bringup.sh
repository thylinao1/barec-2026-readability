#!/bin/bash
# One-shot cluster bring-up + fleet launch. Run FROM the Mac. Idempotent.
# Usage: bash scripts/cluster/bringup.sh [--launch]
#   without --launch: rsync + env + smoke only
#   with    --launch: also sbatch the 50-task array (%10 throttle)
set -euo pipefail
R=$HOME/Developer/barec-2026-sentence-open

echo "== 1/4 rsync code + data =="
rsync -az -e ssh "$R/scripts/cluster_train.py" "$R/scripts/cluster/fleet_manifest.tsv" \
      "$R/scripts/cluster/run_array.sh" soc:~/barec/
rsync -az -e ssh \
      "$R/data/raw/barec-2026-sent/data/train-00000-of-00001.parquet" \
      "$R/data/raw/barec-2026-sent/data/validation-00000-of-00001.parquet" \
      "$R/data/raw/barec-2026-sent/data/test-00000-of-00001.parquet" \
      "$R/data/processed/d3tok_train.parquet" \
      "$R/data/processed/d3tok_validation.parquet" \
      "$R/data/processed/d3tok_test.parquet" \
      soc:~/barec/data/
ssh soc 'mkdir -p ~/barec/logs ~/barec/runs'

echo "== 2/4 venv (idempotent; system python 3.12) =="
ssh soc 'test -x ~/barec/venv/bin/python || python3 -m venv ~/barec/venv
  ~/barec/venv/bin/python -c "import torch, transformers" 2>/dev/null || \
    ~/barec/venv/bin/pip -q install --upgrade pip && \
    ~/barec/venv/bin/pip -q install torch==2.5.1 transformers==4.46.3 accelerate==1.1.1 \
        pandas pyarrow scikit-learn sentencepiece protobuf
  ~/barec/venv/bin/python -c "import torch, transformers, sklearn, pandas; \
    print(\"env OK torch\", torch.__version__, \"transformers\", transformers.__version__)"'

echo "== 3/4 GPU smoke test (camelbert .bin load + 30 steps + save + score) =="
ssh soc 'source ~/barec/venv/bin/activate && cd ~/barec && \
  srun --partition=gpu --gpus=a100-40 --time=00:20:00 \
    python cluster_train.py --backbone CAMeL-Lab/bert-base-arabic-camelbert-msa \
      --loss ce --input word --seed 42 --name SMOKE-camelbert-ce-word --smoke'
ssh soc 'test -f ~/barec/runs/SMOKE-camelbert-ce-word/DONE && echo "SMOKE PASS" || { echo "SMOKE FAIL"; exit 1; }'

if [[ "${1:-}" == "--launch" ]]; then
  echo "== 4/4 launching fleet array 0-49%10 =="
  ssh soc 'cd ~/barec && sbatch --array=0-49%10 run_array.sh && squeue --me'
else
  echo "== 4/4 skipped (pass --launch to fire the array) =="
fi
