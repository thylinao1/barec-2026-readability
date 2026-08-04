#!/bin/bash
#SBATCH --job-name=barec-fleet
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --time=04:00:00
#SBATCH --output=/home/e/e1506804/barec/logs/%x-%A_%a.out
# One task per manifest line. Submit:  sbatch --array=0-49%10 run_array.sh
set -euo pipefail
source ~/barec/venv/bin/activate
cd ~/barec

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" fleet_manifest.tsv)
[ -z "$LINE" ] && { echo "no manifest line for task $SLURM_ARRAY_TASK_ID"; exit 1; }
IFS='|' read -r NAME BACKBONE LOSS INPUT SEED LR WARMUP <<< "$LINE"

if [ -f "runs/$NAME/DONE" ]; then
    echo "[$NAME] already DONE, skipping"
    exit 0
fi

echo "[$NAME] backbone=$BACKBONE loss=$LOSS input=$INPUT seed=$SEED lr=$LR warmup=$WARMUP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python cluster_train.py \
    --backbone "$BACKBONE" --loss "$LOSS" --input "$INPUT" \
    --seed "$SEED" --lr "$LR" --warmup "$WARMUP" \
    --name "$NAME" --data-dir ~/barec/data --out-dir ~/barec/runs
