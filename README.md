# barec-2026-readability

Code and public-split artifacts for team thylinao's entry to the
[BAREC 2026 Shared Task](https://barec.camel-lab.com/sharedtask2026) on sentence-level
Arabic readability assessment (ArabicNLP 2026, co-located with EMNLP 2026). The system is a
fourteen-member equal-weight ensemble of Arabic encoder fine-tunes, decoded from a
continuous score to the 19-level BAREC scale by 18 tuned cutpoints.

One prediction file was submitted to both sentence-level tracks. It scored 85.3 QWK and
placed 2nd on both.

**Correction, 2026-09-03.** The entry first submitted during the testing phase had four of its
fourteen seats fine-tuned on the training split concatenated with the corpus's released gold
test split. The organisers ruled that this falls outside the shared task rule that models are
trained on the training split alone, and it was withdrawn. Everything in this repository now
describes the replacement: the same fourteen seats with those four swapped for their
training-split-only counterparts from the same fine-tuning grid, an entry that was also
submitted during the testing phase and that the board scored at 85.3. The withdrawn variant
scored 85.4 and differed on 811 of 8,077 rows.

## Install

Python 3.12, which is what the cluster venv uses. The blending and analysis scripts need
only:

```bash
pip install numpy polars scikit-learn scipy
```

Training and inference additionally need `torch` and `transformers` (plus `peft` and
`pandas` for the LoRA scripts under `scripts/`). The morphological preprocessing in
`scripts/compute_word_camel.py`, `compute_lex_camel.py` and `compute_d3tok_camel.py` needs
`camel-tools`. Those three are written to run in a separate camel-tools virtualenv, as
their docstrings say.

The corpus is not in this repository. Download the shared task splits from Hugging Face
(`CAMeL-Lab/BAREC-Shared-Task-2026-sent`, not gated) and place the three parquet files at
`data/raw/barec-2026-sent/data/{train,validation,test}-00000-of-00001.parquet`, which is
where `harness/data.py` looks for them.

## Run

Harness tests:

```bash
python -m pytest tests
```

`tests/test_cv.py` needs the 2026 train parquet. `tests/test_qwk_parity.py` additionally
needs the 2025 dev gold and the organisers' 2025 example prediction fixture under `ref/`.
It is the byte-parity gate against the official `eval.py`, and it fails without them.

The blend that produced the submitted file:

```bash
python scripts/blind_blend.py --tag fleet14 --members \
  wordCE,camelbert,qarib,reg_myd3,\
avg-arabertv2-reg-d3tok,avg-arabertv02-ce-word,avg-camelbert-ce-word,\
avg-qarib-ce-raw,avg-alarge02-reg-word,avg-araelectra-reg-word,\
avg-camelbert-reg-word,avg-marbert-ce-raw,avg-arabertv2-ce-d3tok,xlmrL-reg-raw-s42
```

With the member score arrays over the blind set in place, that command rebuilds the
submitted file byte for byte:

```
sha256(prediction) = cf2c752406a9ea2bc0ad94e83f40190cab87b958512ef9c03dcf01c12768b762
```

Those blind arrays are not published (see "What is not published" below), so the command
needs them regenerated first. The analysis that runs off the committed validation and test
arrays plus the 2026 data does not: `scripts/fleet_rank.py` ranks members and proposes
blends, `scripts/noise_floor.py` reproduces the noise measurements, and
`scripts/endgame/e1_fitted_weights.py` through `e5_robust_agg.py` rerun the aggregation
experiments.

## Method

The final system averages fourteen member scores at equal weight. Each member's validation
and blind scores are z-standardised within their own split before averaging, then 18
monotone cutpoints map the averaged score to a level in 1..19. The cutpoints are the
midpoint between thresholds fitted on validation and thresholds chosen to match the blind
score distribution to the training label prior, at `f = 0.50`. Four seats are public
CAMeL-Lab checkpoints, nine are seed-averaged families of our own fine-tunes and one is a
single run. Every seat is trained on the BAREC training split alone.

The members came from a Slurm fine-tuning grid on A100 GPUs (`scripts/cluster/`). The
manifest defines 112 runs; `artifacts/fleet/runs/` holds 133 directories, 113 of which carry
a `DONE` marker together with both score arrays. Combined with public checkpoints that gave
a pool of 172 blendable candidates over 13 backbones and 6 loss families.

Most of the work went into measuring rather than modelling, because the offline evaluation
turned out to be too coarse to rank the changes we were making. `scripts/noise_floor.py`
estimates that noise three independent ways: varying the resolution of the threshold search
grid moves the same blend on the same data by sd 0.094 QWK, bootstrapping the test rows with
the cutpoints frozen gives sd 0.44, and reshuffling the fold seed returns bit-identical
numbers, which makes fold-seed stability a trap rather than a noise estimate. The same
script runs the paired bootstrap that shrinks a naive +0.52 gain to +0.24 with a 95%
interval of [+0.03, +0.45].

Two pieces of the harness exist because of that. `harness/metrics.py` has two QWK entry
points: `official_qwk` mirrors the organisers' `eval.py` exactly, and `fold_qwk` pins the
label set to 1..19 so a fold with a sparse tail cannot silently build a smaller confusion
matrix and return a number the official scorer would not. `harness/rounder.py` provides
`nested_oof_qwk`, which fits each fold's cutpoints on the other folds' out-of-fold scores,
so the reported QWK is never measured on the rows the thresholds were tuned on.

The fitted alternatives all lost. `scripts/endgame/` holds them: non-negative least squares
weights, direct-QWK coordinate ascent, an expanded member pool, per-member isotonic
calibration, and median, trimmed-mean and rank-mean aggregation. Every configuration scored
below plain equal weight on the honest nested out-of-fold number.

One preprocessing caveat matters before rerunning anything. Our locally computed
D3Tok matches the corpus's released gold D3Tok column on about 76% of sentences, because the
released column was produced with a license-gated morphological database. That mismatch
matters less than the standardisation bug it masked: a member standardised with gold-D3Tok
validation scores and applied to locally computed blind scores is being z-scored across two
different preprocessing regimes, which is what `scripts/blind_myd3_members.py` was written to
untangle.

## Repository layout

| path | contents |
|---|---|
| `harness/` | metrics, document-grouped fold map, cutpoint rounder, data loading, submission writing |
| `scripts/` | preprocessing, training, blending and analysis |
| `scripts/cluster/` | Slurm job array and the 112-row fine-tuning grid |
| `scripts/endgame/` | final-day instruments and the aggregation experiments |
| `notebooks/` | the Kaggle T4 fine-tuning recipe and its setup notes |
| `artifacts/fleet/runs/` | per-run validation and test score arrays as saved on the cluster, public splits only |
| `artifacts/scores/` | the score arrays the blending and analysis scripts read: public-checkpoint members at the top level, locally recomputed validation arrays and seed-average test arrays under `fleet/`, per-run test arrays in the layout the endgame scripts expect under `cluster_test/`, and the validation and test arrays of the locally tokenised D3Tok members under `blind/`; public splits only |
| `artifacts/folds_2026_train.json` | the committed document-to-fold map |
| `paper/` | LaTeX source of the system description paper and a verification log |
| `tests/` | harness tests, including the byte-parity gate against `eval.py` |

## Results

Final blind-test standings. Submission counts and the first-place scores were pulled from the
Codabench leaderboard API; our own row is the corrected entry (see the correction note above).

| track | ranked submissions | our rank | QWK | Acc | Acc±1 | Dist | Acc7 | Acc5 | Acc3 |
|---|---|---|---|---|---|---|---|---|---|
| Open | 4 | 2 | 85.27 | 37.75 | 70.82 | 1.136 | 60.51 | 66.93 | 74.59 |
| Strict | 11 | 2 | 85.27 | 37.75 | 70.82 | 1.136 | 60.51 | 66.93 | 74.59 |

The seven metrics are the organisers' rescoring of the corrected file, sent to us on
2026-09-04 after the phase had closed: QWK 0.852738, accuracy 0.377492, adjacent accuracy
0.708184, average absolute distance 1.136189, and 0.605051 / 0.669308 / 0.745945 at the 7, 5
and 3 collapsed levels. They round to the 85.3 / 37.7 the board displayed for the same file
during the testing phase (Open submission 872638, 2026-08-01).

The first-place entry scored 85.5 on Open and 85.4 on Strict. Full leaderboard tables and the
checks behind them are in `paper/VERIFIED_FACTS.md`.

## What is not published

Predictions over the blind test set are not included. The blind set is released only to
registered participants and may be reused in a future edition of the shared task, so a
strong prediction file over it would be a pseudo-label source for later entrants. Member
score arrays over the blind set are withheld for the same reason: every `*_blind.npy` that
the scripts write under `artifacts/scores/` is absent, as is the gold-derived D3Tok lexicon.
Every score array over the public validation and test splits is published, which is what
`artifacts/scores/` contains (the `blind/` folder there holds only the validation and test
arrays of the locally tokenised members, named as the scripts expect them). The code
regenerates the withheld arrays from the blind input for anyone who holds it, and the sha256
above lets the organisers verify the submitted file exactly.

## Citation

The system description paper is in `paper/`: "thylinao at BAREC Shared Task 2026: Ensembling
at the Noise Floor". Please also cite the shared task overview:

```bibtex
@inproceedings{elmadani-etal-2026-barec-shared-task,
    title = "{BAREC}-{ST}-2026: The Second Shared Task on {A}rabic Readability Assessment",
    author = "Elmadani, Khalid N. and Alhafni, Bashar and Taha, Hanada and Habash, Nizar",
    booktitle = "Proceedings of the Fourth Arabic Natural Language Processing Conference",
    month = oct, year = "2026", address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics"
}
```

## License

MIT, see `LICENSE`. The BAREC corpus, the SAMER resources and all third-party model
checkpoints keep their own licenses.
