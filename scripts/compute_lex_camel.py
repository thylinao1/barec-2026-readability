"""Compute Lex and D3Lex exactly per ref/barec_analyzer/scripts/preprocess.py
(run in the camel venv). Replicates preprocess.py lines 57-73 VERBATIM, including
the loop that leaves d3lex_word holding its previous value when no segment
qualifies. That is what built the corpus columns, so parity requires it.
Usage: python compute_lex_camel.py <in_sent.csv> <out.csv>
out CSV has ID,Lex,D3Lex."""
import csv
import re
import sys
import time

from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.transliterate import Transliterator

ARCLEAN = ("~/Developer/barec-2026-sentence-open/"
           "ref/barec_analyzer/arclean_map.json")
tr = Transliterator(CharMapper.mapper_from_json(ARCLEAN))


def clean(t):
    out = tr.transliterate(str(t))
    return re.sub(r'(?<=\B)ى(?=\B)', 'ي', out)


src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
print(f"loaded {len(rows)} sentences; loading disambiguator...", flush=True)
bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False, top=1)
simple = [simple_word_tokenize(clean(r["Sentence"]), split_digits=True) for r in rows]
t = time.time()
out_lex, out_d3lex = [], []
for i, toks in enumerate(simple):
    if not toks:
        out_lex.append("")
        out_d3lex.append("")
        continue
    sent = bert.disambiguate_sentences([toks])[0]
    lex_sentence = [dediac_ar(d.analyses[0][1]['lex']) for d in sent]
    d3tok_sentence = [dediac_ar(d.analyses[0][1]['d3tok'])
                      .replace("_+", " +").replace("+_", "+ ") for d in sent]
    d3lex_sentence = []
    d3lex_word = ""  # preprocess.py relies on carry-over; start empty for parity
    for j in range(len(d3tok_sentence)):
        d3tok_word = d3tok_sentence[j].split(" ")
        lex_word = lex_sentence[j]
        for segment in d3tok_word:
            if "+" not in segment or segment == "+":
                d3lex_word = d3tok_sentence[j].replace(segment, lex_word)
        d3lex_sentence.append(d3lex_word)
    out_lex.append(" ".join(lex_sentence))
    out_d3lex.append(" ".join(d3lex_sentence))
    if (i + 1) % 500 == 0 or i + 1 == len(simple):
        print(f"  {i+1}/{len(simple)} ({time.time()-t:.0f}s)", flush=True)

with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ID", "Lex", "D3Lex"])
    for r, a, b in zip(rows, out_lex, out_d3lex):
        w.writerow([r["ID"], a, b])
print(f"wrote {dst}", flush=True)
