# VERIFIED_FACTS — everything checked first-hand on 2026-08-04

Only facts confirmed by a live fetch, a run, or a file read appear here. Anything that
could not be confirmed is listed under "Unconfirmable" with the fallback the paper uses.

---

## A. Organiser requirements (fetched 2026-08-04)

Source: `https://barec.camel-lab.com/sharedtask2026` and `.../paper-guidelines`, plus
`.../faqs`. All three fetched today.

- **Paper deadline 2026-08-13** (site shows "August 8, 2026" struck through, replaced by
  "August 13, 2026"). Notification 2026-08-15. Camera-ready 2026-08-22.
- **Length: up to 4 pages, unlimited references.** Over-length is desk rejected.
- **Title must be** `<Team Name> at BAREC Shared Task 2026: <Your Contribution>`.
- Template: EMNLP 2026 (ACL) style files, unmodified. Wrong paper size, margins or font
  size is rejected without review.
- Not anonymous. OpenReview submission; **link still reads "Coming Soon"** as of today.
- Prescribed skeleton: Abstract, Introduction (about 3/4 page), Background, System
  Overview, Experimental Setup, Results, Conclusion, Appendices.
- Five required citations, BibTeX supplied by the organisers, copied verbatim into
  `paper/latex/refs.bib`.
  **RE-CHECKED 2026-08-06: the organisers changed required citation #1.** It is now the
  2026 overview, `elmadani-etal-2026-barec-shared-task` ("BAREC-ST-2026: The Second Shared
  Task on Arabic Readability Assessment", ArabicNLP 2026, Budapest), where on 2026-08-04
  the page listed the 2025 overview. Same list otherwise: BAREC corpus
  (`elmadani-etal-2025-large`), annotation guidelines (`habash-etal-2025-guidelines`), and
  SAMER corpus plus SAMER lexicon which are required only for the Constrained track.
  The 2026 entry was added to `refs.bib` and is cited in Background; the 2025 overview is
  kept as related work. Verified in the compiled PDF on 2026-08-06.
  Cross-checked against `custom.bib` in the organisers' Overleaf template
  (`https://www.overleaf.com/read/xpmgjmsqtphr`, emailed 2026-08-06 by Khalid Elmadani):
  the template and the guidelines page carry byte-identical BibTeX. The template adds four
  optional related works not on the guidelines page: `altarbouch-etal-2025-barec` (BAREC
  demo), `elmadani-etal-2026-large` (LREC 2026 corpus), `liberato-etal-2024-strategies`,
  `hazim-etal-2022-arabic`. None is required and none is cited by us.
  Template section skeleton matches the paper's: Abstract, Introduction, Background
  (Shared Task Description, Related Work), System Overview, Experimental Setup, Results,
  Conclusion, Appendices.
- Binding commitment to submit (blind data was received). One author must register and
  attend. Teams serve as reviewers. Code release encouraged, URL in the paper.
- Prizes: top system per task+track $100; **Best System Description Paper $250,
  runner-up or honorable mention $150, "regardless of leaderboard ranking"**, judged on
  "clarity, reproducibility, and insight".

## B. Final leaderboard (Codabench API, pulled 2026-08-04)

`api/phases/29340/get_leaderboard/` (Open) and `29342` (Strict).

**Open, 4 ranked submissions**

| # | team | QWK | Acc | Acc±1 | Dist | Acc7 | Acc5 | Acc3 |
|---|---|---|---|---|---|---|---|---|
| 1 | zaher-m | 85.5 | 39.5 | 72.7 | 1.1 | 62.3 | 69.1 | 76.2 |
| 2 | **thylinao** | **85.4** | 37.8 | 71.3 | 1.1 | 60.1 | 67.1 | 74.8 |
| 3 | salmadi | 81.7 | 50.6 | 72.9 | 1.0 | 65.0 | 71.2 | 77.2 |
| 4 | turabusmani | 80.6 | 49.1 | 73.6 | 1.1 | 64.5 | 70.0 | 75.1 |

**Strict, 11 ranked submissions**

| # | team | QWK | Acc | Acc±1 | Dist | Acc7 | Acc5 | Acc3 |
|---|---|---|---|---|---|---|---|---|
| 1 | zaher-m | 85.4 | 38.7 | 73.7 | 1.1 | 61.8 | 69.5 | 76.8 |
| 2 | **thylinao** | **85.4** | 37.8 | 71.3 | 1.1 | 60.1 | 67.1 | 74.8 |
| 3 | rishta_19 | 84.8 | 37.9 | 73.5 | 1.1 | 61.4 | 68.4 | 75.1 |
| 4 | jhliang | 84.2 | 41.7 | 73.3 | 1.1 | 60.9 | 69.2 | 76.1 |
| 5 | bedeo | 84.1 | 40.9 | 73.6 | 1.1 | 62.3 | 69.2 | 75.7 |
| 6 | approx | 83.8 | 41.8 | 73.7 | 1.1 | 62.6 | 69.6 | 76.3 |
| 7 | phonghoang | 83.7 | 47.5 | 69.4 | 1.1 | 58.8 | 65.5 | 72.5 |
| 8 | ahmedabdou | 83.6 | 45.7 | 73.1 | 1.1 | 63.2 | 69.4 | 75.7 |
| 9 | khloud_j | 82.4 | 37.1 | 73.2 | 1.1 | 59.5 | 67.2 | 74.7 |
| 10 | serry | 82.2 | 33.0 | 70.9 | 1.2 | 56.8 | 65.9 | 73.5 |
| 11 | monmon2000 | 79.0 | 55.6 | 70.0 | 1.1 | 65.3 | 70.5 | 76.2 |

Submission ids and timestamps: Open thylinao id 875866 at 2026-08-03T10:13:41Z; Strict
thylinao id 875864 at 2026-08-03T10:12:22Z; Strict zaher-m id 875092 at
2026-08-02T21:03:17Z.

**CORRECTED 2026-08-12: the Strict tie was NOT lost on the earlier timestamp.** That
reading was an assumption, and it is wrong. It appears in `main.tex:288-289` ("taking
first on the earlier timestamp") and in the Experimental Setup mechanic ("ties break
against the later submission"), so **both need fixing in the camera-ready**.

Re-pulled from the leaderboard API. Primary scores are stored as already-rounded strings
(`"85.4"`, `"86.3"`, `precision: 2`), so there is no hidden precision doing the ordering.
Every tie group instead orders by the **secondary metric, exact accuracy, descending**,
while submission times are scrambled within the group:

| phase | tied QWK | order as displayed (owner, accuracy, submitted) |
|---|---|---|
| Strict dev 27157 | 86.3 | thylinao 38.3 (07-16), jhliang 35.9 (07-17), iMak AI Lab 35.0 (07-13) |
| Strict dev 27157 | 85.2 | ahmedabdou 45.1 (07-17), phonghoang 44.1 (07-08) |
| Strict final 29342 | 85.4 | zaher-m 38.7 (08-02), thylinao 37.8 (08-03) |

In the first group the earliest submission is listed last and the latest is listed
second, which rules out submission time in either direction. Three groups, three times
ordered by accuracy. So Strict second place was decided by exact accuracy, 38.7 against
37.8, which is the same metric `main.tex:289-291` already identifies as the winner's
system-level edge. The corrected version is a tidier story than the timestamp one, and
it does not change any decision that was made: sending only strictly better scores to a
led track is safe under either rule.

Note the accuracy inversion worth one sentence in the paper: salmadi and monmon2000 have
much higher exact accuracy (50.6 and 55.6 against our 37.8) and much lower QWK. Under a
quadratically weighted metric the shape of the prediction distribution dominates
per-sentence accuracy.

## C. Final system, reproduced byte-for-byte

Rebuilt `fleet14tt4` from its member list with the committed blender and compared against
the file actually submitted:

```
shipped sha256: e243ab46a0edb09e1361e8bcb52159708e2f22bb4d90063c5206932db0bf2516
rebuilt sha256: e243ab46a0edb09e1361e8bcb52159708e2f22bb4d90063c5206932db0bf2516
BYTE-IDENTICAL
```

Command that reproduces it:

```
python scripts/blind_blend.py --tag fleet14tt4 --members \
  wordCE,camelbert,qarib,reg_myd3,\
  avg-arabertv2-reg-d3tok-tt,avg-arabertv02-ce-word-tt,avg-camelbert-ce-word,\
  avg-qarib-ce-raw,avg-alarge02-reg-word-tt,avg-araelectra-reg-word,\
  avg-camelbert-reg-word,avg-marbert-ce-raw-tt,avg-arabertv2-ce-d3tok,xlmrL-reg-raw-s42
```

14 members, equal weight, half-match calibration at f = 0.50, no document shrinkage.
Four seats are the public-checkpoint champion (wordCE, camelbert, qarib, reg_myd3);
nine are seed-averaged families of our own fine-tunes; one is a single run
(xlmrL-reg-raw-s42). Four of the fourteen are the train+test retrained variants (`-tt`).

`fleet14tt4` differs from the previous incumbent `fleet14` on **811 of 8,077 rows
(10.0%)** — measured, not quoted.

## D. Fine-tuning fleet, counted from disk

- `scripts/cluster/fleet_manifest.tsv`: **112 rows** (the dispatched grid definition).
- `artifacts/fleet/runs/`: 133 directories; **113** carry a `DONE` marker together with
  both `*_validation.npy` and `*_test.npy`. Those 113 are the completed runs.
- Backbones in the manifest: **13** distinct (`aubmindlab/bert-base-arabertv02`,
  `bert-base-arabertv2`, `bert-large-arabertv02`, `bert-large-arabertv2`,
  `araelectra-base-discriminator`; `CAMeL-Lab/bert-base-arabic-camelbert-msa`, `-ca`;
  `UBC-NLP/MARBERTv2`, `ARBERTv2`; `ahmedabdelali/bert-base-qarib`,
  `bert-base-qarib60_1970k`; `FacebookAI/xlm-roberta-large`; `microsoft/mdeberta-v3-base`).
- Loss families in the manifest: **6** — ce 45, reg 44, softqwk 10, focal 6, corn 5, soft 2.
- Blendable member pool (a member counts only if it has a matched validation and blind
  score array): **172**, of which **26** are seed-averaged families and 146 single runs.
- No model weights are tracked in git (checked: zero `.safetensors` / `pytorch_model` files).

## E. Hugging Face identifiers, all VERIFIED via the public API

Every id below returned HTTP 200 with `gated: false`, `private: false`, and an exact
string match on the canonical id.

Datasets: `CAMeL-Lab/BAREC-Shared-Task-2026-sent`, `CAMeL-Lab/BAREC-Shared-Task-2026-doc`,
`CAMeL-Lab/BAREC-Corpus-v1.0`, `CAMeL-Lab/BAREC-Shared-Task-2025-sent`.

Models: `CAMeL-Lab/readability-arabertv2-d3tok-reg`, `-d3tok-CE`,
`CAMeL-Lab/readability-arabertv02-word-CE`, `CAMeL-Lab/readability-camelbert-word-CE`,
`AymanTarig/qarib-barec-optimized-v8`, plus all 11 backbones listed in D.

The shared task's linked collection `CAMeL-Lab/barec-shared-task-2026` contains only the
two 2026 datasets; the corpus and the readability checkpoints live in separate
collections. Cite them by repo id, not via that collection.

## F. Unconfirmable, with the fallback used

- **Calibration point f = 0.40 (recorded as board 84.5).** `HANDOFF.md:306` marks this
  reading as coming from an ambiguous screenshot. The Codabench submissions API needs
  authentication (an unauthenticated request returns the SPA shell, not JSON), so it could
  not be re-checked. **Fallback: drop the 0.40 row.** Report the four confirmed points
  (0.00 → 84.1, 0.50 → 84.5, 0.60 → 84.4, 1.00 → 84.0) and describe the peak as flat
  between 0.5 and 0.6 rather than across 0.4 to 0.6.
- **Ledger rows 1 and 4: which track.** Not recoverable without the authenticated "My
  Submissions" pages. **Fallback: mark the track as not recorded** in the appendix ledger
  rather than guessing.
- **Solo per-member test QWK.** `STATE.md:83-84` and `HANDOFF.md:319-330` disagree for the
  same members (arabertv02-word-CE 85.24 vs 84.52, and three others). The protocols behind
  the two tables were not reconstructed. **Fallback: the table is cut.** The paper does not
  depend on it, and quoting either without knowing which protocol produced it would be
  guessing.
- **Number of registered teams in 2026.** Not published. The paper reports ranked
  submissions per track (4 Open, 11 Strict), which is what the board shows.
