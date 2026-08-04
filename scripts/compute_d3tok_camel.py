"""Compute D3Tok exactly per barec_analyzer/preprocess.py (run in the camel venv).
Usage: python compute_d3tok_camel.py <in_sent.csv> <out_d3tok.csv>
in CSV needs columns ID,Sentence ; out CSV has ID,D3Tok."""
import sys, csv, re, time
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.transliterate import Transliterator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.utils.dediac import dediac_ar
ARCLEAN="~/Developer/barec-2026-sentence-open/ref/barec_analyzer/arclean_map.json"
tr=Transliterator(CharMapper.mapper_from_json(ARCLEAN))
def clean(t):
    out=tr.transliterate(str(t)); return re.sub(r'(?<=\B)ى(?=\B)','ي',out)
src,dst=sys.argv[1],sys.argv[2]
rows=list(csv.DictReader(open(src)))
print(f"loaded {len(rows)} sentences; loading disambiguator...", flush=True)
bert=BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False, top=1)
simple=[simple_word_tokenize(clean(r["Sentence"]), split_digits=True) for r in rows]
t=time.time(); out=[]
# one sentence per call: avoids the cross-batch torch.cat bug in this camel_tools/torch combo
for i,toks in enumerate(simple):
    if not toks:
        out.append(""); continue
    sent=bert.disambiguate_sentences([toks])[0]
    out.append(" ".join(dediac_ar(d.analyses[0][1]['d3tok']).replace("_+"," +").replace("+_","+ ") for d in sent))
    if (i+1)%500==0 or i+1==len(simple):
        print(f"  {i+1}/{len(simple)} ({time.time()-t:.0f}s)", flush=True)
with open(dst,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["ID","D3Tok"])
    for r,d in zip(rows,out): w.writerow([r["ID"],d])
print(f"wrote {dst}", flush=True)
