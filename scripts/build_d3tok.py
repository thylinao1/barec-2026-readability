"""Phase 2 prep: build the D3Tok column for the 2026 splits by joining to the
gold D3Tok shipped in CAMeL-Lab/BAREC-Corpus-v1.0.

Why: the d3tok checkpoints expect D3Tok input, and the corpus was preprocessed
with the license-gated LDC almor-s31 db (the free camel-tools builtin db can
diverge on ambiguous words). The corpus already ships the exact gold D3Tok, so a
join is byte-parity by construction and needs no camel-tools. camel-tools is only
a fallback for any rows the corpus does not cover (e.g. the future Blind Test).

Join strategy: try ID first (cleanest); fall back to normalized Sentence text.
Reports coverage per split and caches data/processed/d3tok_<split>.parquet.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python scripts/build_d3tok.py
"""
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.data import ID, TEXT, load_2026  # noqa: E402

CORPUS_DIR = ROOT / "data" / "raw" / "barec-corpus-v1.0"
OUT = ROOT / "data" / "processed"
HF = str(Path.home() / "mac-ml-setup" / ".venv" / "bin" / "hf")


def ensure_corpus():
    if (CORPUS_DIR / "data").exists():
        return
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [HF, "download", "CAMeL-Lab/BAREC-Corpus-v1.0", "--repo-type", "dataset",
         "--local-dir", str(CORPUS_DIR)],
        check=True,
    )


def load_corpus() -> pl.DataFrame:
    parts = []
    for f in sorted((CORPUS_DIR / "data").glob("*.parquet")):
        parts.append(pl.read_parquet(f))
    df = pl.concat(parts, how="vertical_relaxed")
    assert "D3Tok" in df.columns, f"corpus has no D3Tok column: {df.columns}"
    return df


def _norm(col: str) -> pl.Expr:
    # collapse whitespace so the join is not broken by spacing differences
    return pl.col(col).cast(pl.Utf8).str.replace_all(r"\s+", " ").str.strip_chars()


GOLD_COLS = ["D3Tok", "Word"]  # gold preprocessing variants shipped by the corpus


def build_split(split: str, corpus: pl.DataFrame):
    df = load_2026(split)
    n = df.height

    # 1) join on ID (primary); 2) fill any gaps via normalized Sentence text
    cmap = corpus.select([pl.col(ID).cast(pl.Utf8).alias("_k"), *GOLD_COLS]).unique(subset="_k")
    j = df.with_columns(pl.col(ID).cast(pl.Utf8).alias("_k")).join(cmap, on="_k", how="left")
    cov_id = j["D3Tok"].is_not_null().sum()

    ctext = corpus.select(
        [_norm(TEXT).alias("_t"), *[pl.col(c).alias(f"{c}_t") for c in GOLD_COLS]]
    ).unique(subset="_t")
    jt = df.with_columns(_norm(TEXT).alias("_t")).join(ctext, on="_t", how="left")
    cov_txt = jt["D3Tok_t"].is_not_null().sum()

    merged = j.with_columns(
        [pl.coalesce([pl.col(c), jt[f"{c}_t"]]).alias(c) for c in GOLD_COLS]
    )
    covered = merged["D3Tok"].is_not_null().sum()

    out = merged.select([pl.col(ID).cast(pl.Utf8).alias(ID), TEXT, *GOLD_COLS])
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / f"d3tok_{split}.parquet")
    print(f"[{split}] n={n}  cov(ID)={cov_id}  cov(text)={cov_txt}  cov(final)={covered} "
          f"({100*covered/n:.2f}%)  uncovered={n-covered}")
    return covered, n


def main():
    ensure_corpus()
    corpus = load_corpus()
    print(f"corpus rows={corpus.height} cols={corpus.columns}")
    total_c = total_n = 0
    for split in ["train", "validation", "test"]:
        c, n = build_split(split, corpus)
        total_c += c
        total_n += n
    print(f"\nTOTAL coverage: {total_c}/{total_n} ({100*total_c/total_n:.2f}%)")
    if total_c < total_n:
        print("NOTE: uncovered rows need the camel-tools D3Tok fallback (Phase 2).")


if __name__ == "__main__":
    main()
