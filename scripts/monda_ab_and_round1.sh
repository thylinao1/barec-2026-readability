#!/bin/bash
# Phase A: regime A/B for the undocumented Monda Sentence_ family (val-only, both backbones).
# Phase B is fired separately after reading the A/B verdict.
set -uo pipefail
PY=~/mac-ml-setup/.venv/bin/python
cd $HOME/Developer/barec-2026-sentence-open

for regime in raw word d3tok; do
  PYTHONWARNINGS=ignore $PY -u scripts/score_public_member.py \
    --repo Monda/bert-base-arabertv2_Sentence_CE_19levels --input $regime \
    --name AB_v2sent_$regime --val-only
  PYTHONWARNINGS=ignore $PY -u scripts/score_public_member.py \
    --repo Monda/bert-base-arabertv02_Sentence_CE_19levels --input $regime \
    --name AB_v02sent_$regime --val-only
done

PYTHONWARNINGS=ignore $PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, ".")
from harness.data import TARGET, load_2026
from sklearn.metrics import cohen_kappa_score
y = np.array(load_2026("validation")[TARGET].to_list(), int)
print(f"\n{'model':14s}{'regime':8s}{'naiveQWK':>9s}{'corr(y)':>9s}")
for fam in ["v2sent", "v02sent"]:
    for reg in ["raw", "word", "d3tok"]:
        s = np.load(f"artifacts/scores/fleet/AB_{fam}_{reg}_val.npy")
        q = cohen_kappa_score(y, np.clip(np.round(s), 1, 19).astype(int), weights="quadratic")
        print(f"{fam:14s}{reg:8s}{q*100:9.2f}{np.corrcoef(s, y)[0,1]:9.4f}")
EOF
echo AB_COMPLETE
