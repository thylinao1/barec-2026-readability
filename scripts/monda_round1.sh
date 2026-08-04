#!/bin/bash
# Round-1 public-member scoring: 13 members, val+blind, contamination gate armed.
# Regimes per the 2026-07-31 A/B: v2_Sentence_* -> d3tok; v02_Sentence_* presumed
# val-contaminated (their CE sibling measured 94.2 naive val QWK) and NOT queued.
set -uo pipefail
PY=~/mac-ml-setup/.venv/bin/python
cd $HOME/Developer/barec-2026-sentence-open
run() {
  PYTHONWARNINGS=ignore $PY -u scripts/score_public_member.py \
    --repo "$1" --input "$2" --name "$3" --evict ${4:+--tok-repo "$4"}
}
run Monda/bert-base-arabertv2_D3Tok_EMD_19levels        d3tok monda_d3tok_emd
run Monda/bert-base-arabertv2_Word_CE_19levels          word  monda_word_ce
run Monda/bert-base-arabertv2_Sentence_CE_19levels      d3tok monda_sent_ce
run Monda/bert-base-arabertv2_Sentence_OLL15_19levels   d3tok monda_sent_oll15
run Monda/bert-base-arabertv2_Sentence_EMD_19levels     d3tok monda_sent_emd
run Monda/bert-base-arabertv2_Sentence_FocalCB_19levels-2 d3tok monda_sent_focalcb2
run Monda/bert-base-arabertv2_Sentence_REG_19levels     d3tok monda_sent_reg
run Monda/bert-base-arabertv2_D3Tok_REG_19levels        d3tok monda_d3tok_reg
run Monda/bert-base-arabertv2_D3Tok_WCE_19levels        d3tok monda_d3tok_wce
run Monda/bert-base-arabertv2_Word_EMD_19levels         word  monda_word_emd
run AymanTarig/qarib-marbertv2-optimized-v1             raw   qarib_marbertv2_v1 UBC-NLP/MARBERTv2
run AymanTarig/qarib-marbertv2-optimized-v3             raw   qarib_marbertv2_v3 UBC-NLP/MARBERTv2
run Noorrabie/MARBERT                                   raw   noorrabie_marbert
echo ROUND1_COMPLETE
