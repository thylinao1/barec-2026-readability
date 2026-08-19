"""Hybrid my-D3Tok v2: gold-corpus lexicon (train+test ONLY - val stays out so
the val regime honestly mirrors blind) with old-BERT-disambiguator fallback.

Per token: lexicon hit -> majority gold d3tok; miss -> the old pipeline's
d3tok for that token (aligned via segment grouping); alignment failure ->
lexicon-or-identity for the whole sentence (counted).

Variant B also routes AMBIGUOUS lexicon types (>1 observed d3tok)
to the old pipeline.

  build_lex                       -> /tmp/d3tok_lex_traintest.json (full counts)
  apply <word_csv> <old_d3_csv> <out_csv> [--ambig-old]
  evalval <variant_out_csv>       -> exact-sentence match vs gold val D3Tok
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LEXPATH = Path("/tmp/d3tok_lex_traintest.json")
sys.path.insert(0, str(ROOT / "scripts" / "endgame"))
from d3tok_lexicon import group_segments  # noqa: E402


def build_lex():
    counts = defaultdict(Counter)
    aligned = skipped = 0
    for split in ("train", "test"):
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
    out = {w: {"top": c.most_common(1)[0][0], "n": len(c)} for w, c in counts.items()}
    LEXPATH.write_text(json.dumps(out, ensure_ascii=False))
    amb = sum(1 for v in out.values() if v["n"] > 1)
    print(f"train+test lexicon: {aligned} sentences aligned ({skipped} skipped), "
          f"{len(out)} types, {amb} ambiguous")


def apply(word_csv, old_csv, out_csv, ambig_old=False):
    lex = json.loads(LEXPATH.read_text())
    words = {r["ID"]: r["Word"] for r in csv.DictReader(open(word_csv))}
    old = {r["ID"]: r["D3Tok"] for r in csv.DictReader(open(old_csv))}
    n_tok = n_lex = n_old = n_alignfail = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "D3Tok"])
        for i, wtext in words.items():
            toks = (wtext or "").split()
            ogroups = group_segments((old.get(i, "") or "").split())
            ok = len(ogroups) == len(toks)
            if not ok:
                n_alignfail += 1
            parts = []
            for j, t in enumerate(toks):
                n_tok += 1
                e = lex.get(t)
                use_lex = e is not None and not (ambig_old and e["n"] > 1)
                if use_lex:
                    parts.append(e["top"])
                    n_lex += 1
                elif ok:
                    parts.append(" ".join(ogroups[j]))
                    n_old += 1
                else:
                    parts.append(e["top"] if e else t)
            w.writerow([i, " ".join(parts)])
    print(f"{out_csv}: {len(words)} rows; tokens {n_tok} -> lexicon {n_lex} "
          f"({n_lex / n_tok * 100:.2f}%), old-pipeline {n_old} "
          f"({n_old / n_tok * 100:.2f}%), align-fail sentences {n_alignfail}")


def evalval(out_csv):
    gold = pl.read_parquet(ROOT / "data/processed/d3tok_validation.parquet")
    ids = [str(i) for i in gold["ID"].to_list()]
    gd = dict(zip(ids, gold["D3Tok"].to_list()))
    mine = {r["ID"]: r["D3Tok"] for r in csv.DictReader(open(out_csv))}
    hit = sum(mine.get(i) == gd[i] for i in ids)
    print(f"{out_csv} vs gold val D3Tok: {hit}/{len(ids)} = {hit / len(ids) * 100:.2f}% "
          f"exact-sentence (old pipeline ~76%, LVO-lexicon-alone 55.8%)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build_lex":
        build_lex()
    elif cmd == "apply":
        apply(sys.argv[2], sys.argv[3], sys.argv[4], "--ambig-old" in sys.argv)
    elif cmd == "evalval":
        evalval(sys.argv[2])
