# thylinao at BAREC Shared Task 2026: Ensembling at the Noise Floor

Code and public-split artifacts for team **thylinao**'s entry to the
[BAREC 2026 Shared Task](https://barec.camel-lab.com/sharedtask2026) on sentence-level
Arabic readability assessment (ArabicNLP 2026, co-located with EMNLP 2026).

**Result: 85.4 QWK, second place on both the Open and the Strict sentence-level tracks.**

| track | rank | QWK | Acc | Acc±1 | Dist | Acc7 | Acc5 | Acc3 |
|---|---|---|---|---|---|---|---|---|
| Open (4 ranked) | 2 | 85.4 | 37.8 | 71.3 | 1.1 | 60.1 | 67.1 | 74.8 |
| Strict (11 ranked) | 2 | 85.4 | 37.8 | 71.3 | 1.1 | 60.1 | 67.1 | 74.8 |

The same file was submitted to both tracks. On Strict the first-place entry also scored
85.4 and won on the earlier timestamp.

## What is interesting here

The system is an ordinary equal-weight ensemble. The paper is about how its choices were
measured, and that is what this repository is organised around.

- **`scripts/noise_floor.py`** measures the noise floor of the offline evaluation three
  independent ways: threshold-grid resolution (sd 0.094), test-set bootstrap (sd 0.44), and
  fold-seed reshuffling (which returns bit-identical results and is therefore a trap, not a
  noise estimate). It also runs the paired bootstrap that shrinks a naive +0.52 gain to
  +0.24 with a 95% interval of [+0.03, +0.45].
- **`harness/metrics.py`** reproduces the official scorer string-for-string on the provided
  development fixture, including all six secondary metrics. Two entry points: `official_qwk`
  mirrors `eval.py` exactly; `fold_qwk` pins the label set to [1,19] so a fold with a sparse
  tail cannot silently build a smaller confusion matrix.
- **`harness/rounder.py`** is the 18-cutpoint monotone coordinate-ascent rounder, plus
  `nested_oof_qwk`, the out-of-fold-of-out-of-fold protocol used as the go/no-go number.
- **`scripts/blind_blend.py`** is the blender that produced every submission, including the
  half-match cutpoint calibration described in Section 3.3 of the paper.
- **`scripts/endgame/`** holds the two instruments used on the final day: an honest
  validation nested-OOF estimator and a deployment-mirrored test oracle, plus the
  aggregation experiments (fitted weights, coordinate ascent, isotonic calibration, robust
  aggregators) that all lost to plain equal weight.

## The final system

Fourteen members, equal weight, z-standardised per split, half-match calibration at
`f = 0.50`. With the member blind score arrays in place, the committed blender rebuilds the
submitted file byte for byte:

```
sha256(prediction) = e243ab46a0edb09e1361e8bcb52159708e2f22bb4d90063c5206932db0bf2516
```

```bash
python scripts/blind_blend.py --tag fleet14tt4 --members \
  wordCE,camelbert,qarib,reg_myd3,\
avg-arabertv2-reg-d3tok-tt,avg-arabertv02-ce-word-tt,avg-camelbert-ce-word,\
avg-qarib-ce-raw,avg-alarge02-reg-word-tt,avg-araelectra-reg-word,\
avg-camelbert-reg-word,avg-marbert-ce-raw-tt,avg-arabertv2-ce-d3tok,xlmrL-reg-raw-s42
```

Four seats are public checkpoints, nine are seed-averaged families of our own fine-tunes,
one is a single run. Four seats use variants retrained on train plus the released test
split, which is legal in both tracks.

## Layout

```
harness/          metrics, fold map, rounder, data loading, submission writing
scripts/          preprocessing, training, blending, analysis
  cluster/        Slurm job array and the 112-row fine-tuning grid
  endgame/        final-day instruments and aggregation experiments
artifacts/fleet/  per-run validation and test score arrays (public splits only)
paper/            the LaTeX source and a first-hand verification log
tests/            harness tests, including the byte-parity gate
```

## Reproducing

Requires the BAREC 2026 shared task data from Hugging Face
(`CAMeL-Lab/BAREC-Shared-Task-2026-sent`, not gated) and CAMeL Tools for the morphological
views. The `Word` and `D3Tok` pipelines are in `scripts/compute_word_camel.py` and
`scripts/compute_d3tok_camel.py`. Note that our locally computed D3Tok matches the released
gold column at about 76% because the released column used a license-gated morphological
database; the paper explains why that mattered less than the standardisation bug it masked.

Fine-tuning used a Slurm cluster with A100 GPUs (`scripts/cluster/`). The 113 completed runs
produced the score arrays in `artifacts/fleet/runs/`, so the blending and analysis can be
rerun without retraining anything.

## What is deliberately not here

**Predictions on the blind test set are not published.** The blind set is released only to
registered participants, and it may be reused in a future edition of the shared task; a
strong prediction file over it would be a pseudo-label source for later participants. The
code regenerates those files from the blind input for anyone who holds it, and the sha256
above lets the organisers verify the submitted file exactly. Internal campaign notes are
also not included.

## Citation

The system description paper is in `paper/`. Please also cite the shared task overview:

```bibtex
@inproceedings{elmadani-etal-2025-barec-shared-task,
    title = "{BAREC} Shared Task 2025 on {A}rabic Readability Assessment",
    author = "Elmadani, Khalid N. and Alhafni, Bashar and Taha, Hanada and Habash, Nizar",
    booktitle = "Proceedings of the Third Arabic Natural Language Processing Conference",
    month = nov, year = "2025", address = "Suzhou, China",
    publisher = "Association for Computational Linguistics"
}
```

## License

MIT, see `LICENSE`. The BAREC corpus, the SAMER resources and all third-party model
checkpoints keep their own licenses.
