"""Gold-corpus D3Tok lexicon: word (cleaned Word-token form) -> majority gold
D3Tok segmentation, built from the train+validation+test gold columns by
aligning each sentence's Word tokens to its D3Tok segment groups.

Segment grouping (CAMeL convention after '_+'->' +' / '+_'->'+ ' replacement):
a segment ending in '+' is a proclitic attaching forward; a segment starting
with '+' is an enclitic attaching backward. A group = one original word.

Usage:
  build   -> writes artifacts/scores/blind/d3tok_lexicon.json + stats
  apply <word_csv> <out_csv>  -> D3Tok column via lexicon (OOV: word unchanged)
  evalval -> exact-match of lexicon D3Tok vs gold D3Tok on validation,
             from BOTH gold-Word tokens (ceiling) and my-Word tokens (real)
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LEX = ROOT / "artifacts" / "scores" / "blind" / "d3tok_lexicon.json"


def group_segments(segs):
    groups, cur = [], []
    for s in segs:
        if cur and (cur[-1].endswith("+") or s.startswith("+")):
            cur.append(s)
        else:
            if cur:
                groups.append(cur)
            cur = [s]
    if cur:
        groups.append(cur)
    return groups


def build():
    counts = defaultdict(Counter)
    aligned = skipped = 0
    for split in ("train", "validation", "test"):
        df = pl.read_parquet(ROOT / f"data/processed/d3tok_{split}.parquet")
        for word, d3 in zip(df["Word"].to_list(), df["D3Tok"].to_list()):
            words = (word or "").split()
            groups = group_segments((d3 or "").split())
            if len(groups) != len(words) or not words:
                skipped += 1
                continue
            aligned += 1
            for w, g in zip(words, groups):
                counts[w][" ".join(g)] += 1
    lex = {w: c.most_common(1)[0][0] for w, c in counts.items()}
    # drop identity entries (OOV fallback already yields the word itself)
    lex = {w: d for w, d in lex.items() if d != w}
    LEX.write_text(json.dumps(lex, ensure_ascii=False))
    amb = sum(1 for w, c in counts.items() if len(c) > 1)
    print(f"aligned {aligned} sentences, skipped {skipped} "
          f"({skipped / (aligned + skipped) * 100:.2f}%)")
    print(f"lexicon: {len(counts)} word types, {len(lex)} non-identity entries, "
          f"{amb} ambiguous ({amb / len(counts) * 100:.1f}%)")


def d3_of(tokens, lex):
    return " ".join(lex.get(t, t) for t in tokens)


def apply(src, dst):
    lex = json.loads(LEX.read_text())
    rows = list(csv.DictReader(open(src)))
    total = oov = 0
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "D3Tok"])
        for r in rows:
            toks = r["Word"].split()
            total += len(toks)
            oov += sum(1 for t in toks if t not in lex)
            w.writerow([r["ID"], d3_of(toks, lex)])
    print(f"wrote {dst} ({len(rows)} rows); token OOV-vs-lexicon "
          f"{oov}/{total} = {oov / total * 100:.2f}% (OOV keeps word unchanged)")


def evalval():
    lex = json.loads(LEX.read_text())
    gold = pl.read_parquet(ROOT / "data/processed/d3tok_validation.parquet")
    ids = [str(i) for i in gold["ID"].to_list()]
    gw = dict(zip(ids, gold["Word"].to_list()))
    gd = dict(zip(ids, gold["D3Tok"].to_list()))
    mine = {r["ID"]: r["Word"] for r in csv.DictReader(open("/tmp/val_word_mine.csv"))}
    for label, words in (("gold-Word tokens (ceiling)", gw),
                         ("my-Word tokens (deployment)", mine)):
        hit = sum(d3_of((words[i] or "").split(), lex) == gd[i] for i in ids)
        print(f"{label:28s} exact-sentence match {hit}/{len(ids)} = "
              f"{hit / len(ids) * 100:.2f}%  (old BERT-disambig pipeline: ~76%)")


if __name__ == "__main__":
    {"build": build, "evalval": evalval}.get(
        sys.argv[1], lambda: apply(sys.argv[2], sys.argv[3]))()
