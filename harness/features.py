"""Cheap interpretable surface / readability features (Polars).

No leaky columns (Document/Source/Author/Domain/Text_Class are excluded). These
feed the classical floor and, later, the stacking meta-layer. SAMER-lexicon and
morphological features are added in a later phase (need the lexicon + camel-tools).
"""
from __future__ import annotations

import numpy as np
import polars as pl

# Arabic script block, Arabic-Indic + ASCII digits, common punctuation.
_ARABIC = r"[؀-ۿ]"
_DIGIT = r"[0-9٠-٩]"
_PUNCT = r"""[.,;:!?()\[\]{}"'\-،؛؟…«»]"""

FEATURE_NAMES = [
    "f_word_count_col", "f_n_chars", "f_n_chars_nospace", "f_n_tokens",
    "f_max_tok_len", "f_mean_tok_len", "f_n_unique_tok", "f_punct", "f_digit",
    "f_arabic", "f_ttr", "f_punct_ratio", "f_digit_ratio", "f_arabic_ratio",
    "f_avg_word_len",
]


def surface_features(df: pl.DataFrame) -> pl.DataFrame:
    s = pl.col("Sentence").cast(pl.Utf8).fill_null("")
    toks = s.str.split(" ")
    tok_lens = toks.list.eval(pl.element().str.len_chars())
    base = df.select(
        pl.col("Word_Count").cast(pl.Float64).alias("f_word_count_col"),
        s.str.len_chars().cast(pl.Float64).alias("f_n_chars"),
        s.str.replace_all(" ", "").str.len_chars().cast(pl.Float64).alias("f_n_chars_nospace"),
        toks.list.len().cast(pl.Float64).alias("f_n_tokens"),
        tok_lens.list.max().cast(pl.Float64).alias("f_max_tok_len"),
        tok_lens.list.mean().cast(pl.Float64).alias("f_mean_tok_len"),
        toks.list.unique().list.len().cast(pl.Float64).alias("f_n_unique_tok"),
        s.str.count_matches(_PUNCT).cast(pl.Float64).alias("f_punct"),
        s.str.count_matches(_DIGIT).cast(pl.Float64).alias("f_digit"),
        s.str.count_matches(_ARABIC).cast(pl.Float64).alias("f_arabic"),
    )
    out = base.with_columns(
        (pl.col("f_n_unique_tok") / pl.col("f_n_tokens").clip(lower_bound=1)).alias("f_ttr"),
        (pl.col("f_punct") / pl.col("f_n_chars").clip(lower_bound=1)).alias("f_punct_ratio"),
        (pl.col("f_digit") / pl.col("f_n_chars").clip(lower_bound=1)).alias("f_digit_ratio"),
        (pl.col("f_arabic") / pl.col("f_n_chars").clip(lower_bound=1)).alias("f_arabic_ratio"),
        (pl.col("f_n_chars_nospace") / pl.col("f_n_tokens").clip(lower_bound=1)).alias("f_avg_word_len"),
    )
    return out.select(FEATURE_NAMES).fill_nan(0.0).fill_null(0.0)


def to_matrix(df: pl.DataFrame) -> np.ndarray:
    return surface_features(df).to_numpy().astype(np.float32)
