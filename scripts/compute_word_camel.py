"""Compute the Word input variant exactly per ref/barec_analyzer/scripts/preprocess.py
(run in the camel venv): arclean transliteration + the ى→ي regex, then
simple_word_tokenize(split_digits=True), space-joined.
Usage: python compute_word_camel.py <in_sent.csv> <out_word.csv>
in CSV needs columns ID,Sentence ; out CSV has ID,Word."""
import csv
import re
import sys

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.transliterate import Transliterator

ARCLEAN = ("~/Developer/barec-2026-sentence-open/"
           "ref/barec_analyzer/arclean_map.json")
tr = Transliterator(CharMapper.mapper_from_json(ARCLEAN))


def clean(t):
    out = tr.transliterate(str(t))
    return re.sub(r'(?<=\B)ى(?=\B)', 'ي', out)


src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ID", "Word"])
    for r in rows:
        w.writerow([r["ID"], " ".join(simple_word_tokenize(clean(r["Sentence"]),
                                                           split_digits=True))])
print(f"wrote {dst} ({len(rows)} rows)")
