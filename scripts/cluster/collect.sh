#!/bin/bash
# Pull the fleet's SMALL artifacts down (meta + npy scores, never model weights).
# Idempotent; run any time. Weights come later, selectively, via fleet_land.py.
set -euo pipefail
DEST=$HOME/Developer/barec-2026-sentence-open/artifacts/fleet/runs
mkdir -p "$DEST"
rsync -az -e ssh \
  --include='*/' --include='meta.json' --include='DONE' --include='*_validation.npy' \
  --include='*_test.npy' --exclude='*' \
  soc:~/barec/runs/ "$DEST/"
echo "collected:"
ls "$DEST" | grep -v SMOKE | wc -l
for d in "$DEST"/*/; do
  n=$(basename "$d")
  [ "$n" = "SMOKE-camelbert-ce-word" ] && continue
  [ -f "$d/DONE" ] && echo "  DONE $n"
done
