# VERIFIED_FACTS: everything checked first-hand on 2026-08-04

Only facts confirmed by a live fetch, a run, or a file read appear here. Anything that
could not be confirmed is listed under "Unconfirmable" with the fallback the paper uses.

---

## -1. TEST-SET CORRECTION (added 2026-09-03, supersedes sections C and B for our own rows)

On 2026-09-02 Khalid Elmadani wrote that our system "appears to use the test set for training",
which is not permitted, and asked for a retrained system, an updated paper and the final
prediction file by 2026-09-07, failing which the paper will not be accepted (verbatim in
`docs/correction/ORGANISER-RULING-2026-09-02.md`).

He is right. Four of the fourteen seats in `fleet14tt4` were fine-tuned on train plus the
released gold test split (`artifacts/fleet/runs/<name>-tt-s*/meta.json` carries
`"extra_train": "test"`). The shipped system is now `fleet14`, the same fourteen seats with
those four replaced by their training-split-only counterparts.

- Shipped file: `submissions/blind_blend/fleet14/prediction`, sha256
  `cf2c752406a9ea2bc0ad94e83f40190cab87b958512ef9c03dcf01c12768b762`
  (zip `2aec7688a5d84f12fcd1b10390039eb3e03501be3b4966dc37ffbbe78084b7e1`).
  Rebuilt from member arrays and confirmed byte-identical on 2026-09-03.
- Withdrawn file: `fleet14tt4`, sha256 `e243ab46...`, the hash that section C records and that
  the pre-correction paper printed. They differ on 811 of 8,077 rows.
- Board reading for the shipped file: 85.3 QWK, exact accuracy 37.7, Open submission id 872638,
  2026-08-01, also uploaded to Strict that day. Source is the campaign log written on the day
  (`HANDOFF.md` lines 105 to 108 and 245).
- **Official rescoring, received from Khalid Elmadani on 2026-09-04** (Gmail thread
  `1a0621ad1865e59f`; verbatim copy and checks in
  `docs/correction/ORGANISER-RESCORE-2026-09-04.md`) of the file attached to our 2026-09-03
  reply, which is byte-identical to the zip above: Accuracy 0.377492, Accuracy (±1) 0.708184,
  Average Absolute Distance 1.136189, QWK 0.852738, Accuracy 7-point 0.605051, 5-point
  0.669308, 3-point 0.745945. Every rate times 8,077 lands within 0.003 of an integer (3,049 /
  5,720 / 4,887 / 5,406 / 6,025 correct, distance sum 9,177), so the scored file has the blind
  set's row count,
  and QWK and accuracy round to the board's 85.3 / 37.7. The paper prints the line after the
  ledger table in Appendix A as 85.27 / 37.75 / 70.82 / 1.136 / 60.51 / 66.93 / 74.59, and the
  Final standing subsection points to it with "(85.27 and 37.75 in the organisers'
  rescoring, Appendix A)". The body had no room for more: the 4-page limit was met to the line.
- **Development-split tuning ruled in, same email, verbatim:** "we don't have a tuning set in
  BAREC corpus, so you can use the development set for tuning thresholds and model selection."
  The paper says so at the end of the training-data paragraph in Appendix D (`app:fleet`).
- Camera-ready due **2026-09-10** on OpenReview
  (`https://openreview.net/group?id=SIGARAB.org/ArabicNLP/2026/BAREC_Shared_Task`), re-fetched
  2026-09-05. The ArabicNLP 2026 conference site says 2026-09-17 for camera-ready; the shared
  task's earlier date governs. What the OpenReview camera-ready form asks for beyond the PDF
  (author metadata, publication agreement, source files) is UNVERIFIED: the group is
  login-gated and its public instructions field is empty.
- **Decision and reviews, read on OpenReview by the operator 2026-09-05** (verbatim in
  `docs/correction/OPENREVIEW-DECISION-2026-09-02.md`): **Accept (Conditional)**, decision
  comment = the 2 Sept email text (retrain, final file by 7 Sept). Reviewer bsMq (rating 2,
  confidence 5) asks for three things: name the two tracks in the abstract; broaden related
  work to SAMER, DARES and OSMAN; clarify the data-use procedure, since the test split should
  be evaluated once and the blind set must not shape the system. Reviewer VtE3 (rating 4):
  "Solid presentation. no issues." The camera-ready answers all three bsMq points (section
  12 of `docs/correction/CORRECTION-2026-09.md`).
- The live Codabench boards (phases 29340 and 29342, API re-read 2026-09-05) still list
  thylinao at the withdrawn 85.4 / 37.8 on both tracks with no withdrawn marker. The paper and
  README print the corrected line; REPLY-2-DRAFT.md asks the organisers whether the overview
  and final leaderboard will carry it.
- Placings do not change. Against the final boards pulled live on 2026-09-02, 85.3 is second on
  Open (winner 85.5, third 81.7) and second on Strict (winner 85.4, third 84.8). The Strict tie
  at 85.4, and therefore the accuracy tie-break, no longer applies to us.
- Strict rule verbatim, re-fetched 2026-09-02: "Models must be trained exclusively on the
  training set of the BAREC Corpus." The Open track separately permits "any publicly available
  data". We are complying on both tracks and not arguing the Open reading.
- Verified clean: the shipped decoder reads only train (label prior) and validation (cutpoints);
  no fleet14 member uses the train-plus-validation-plus-test D3Tok lexicon; the test oracle in
  `scripts/endgame/common.py` postdates fleet14's selection by two days and is never imported
  by the blender.
- fleet14 is a local optimum of the compliant pool: all 14 leave-one-out drops cost 0.102 to
  0.357, and all 148 single-member additions from the 162-member compliant pool score below it.
- Still undocumented: the training data of `AymanTarig/qarib-barec-optimized-v8`, one of the
  fourteen seats. A question about it was drafted but cut from the reply actually sent on
  2026-09-03, which asked only about development-split tuning and the metric line. It has
  NOT been raised with the organisers.

Full record: `docs/correction/CORRECTION-2026-09.md`.

---

## 0. SUBMISSION RECORD (added 2026-08-17)

**Paper submitted to OpenReview on 2026-08-17, both sentence-level tracks.** Operator
reported the submission; the receipt goes to `e1506804@u.nus.edu`, which is the address on
OpenReview profile `~Maksim_Silchenko1` and is not reachable from the connected Gmail
account, so the receipt was not independently seen from this side. Confirm in NUS mail.

Submitted artefact: `paper/latex/main.pdf`, 9 pages, body ending page 4. Built with
`tectonic -X compile main.tex`. Zero overfull hboxes, zero undefined references.

**Deadlines re-verified against the live site on 2026-08-17** (the dates in section A below
are the pre-extension ones and are stale): papers due **2026-08-22**, notification
**2026-09-02**, camera-ready **2026-09-10**. The site shows the old date struck through
followed by the new one, which is easy to misread in the opposite direction.

**Strict track rule, fetched verbatim 2026-08-17:** "Models must be trained exclusively on
the training set of the BAREC Corpus." The four seats retrained on train plus the released
test split therefore fall outside a literal reading. The paper no longer claims they are
"legal in both tracks"; it states the fact and leaves the judgement to the organisers.

DONE 2026-09-05: `artifacts/scores/` is now in the public checkout under one rule, "every
score array over a public split is published, every array over the blind set is withheld".
Copied by rsync from the private directory: the 20 top-level validation and test arrays
(train arrays, `.kaggle` copies, `phase1_oof.npy` and the backup folder left out), `fleet/`
without its 172 `*_blind.npy` (172 locally recomputed `*_val.npy`, 22 seed-average
`*_test.npy`, the manifest), `cluster_test/` in full (116 per-run test arrays, byte-identical
to the runs' arrays under `artifacts/fleet/runs/` but in the layout `endgame/common.py`
resolves), and the seven validation and test arrays from `blind/` (`reg_valmyd3`,
`dtCE_valmyd3`, `regold_*`, `regv2_*`), with the blind arrays and `d3tok_lexicon.json` left
out. 339 files, 18 MB, zero `*_blind.npy` anywhere. The fleet `*_val.npy` are NOT duplicates
of the runs' `*_validation.npy` (different bytes: they are the local recomputation the parity
gate requires), which is why both are published. Verified by running
`scripts/noise_floor.py` in the public checkout with the three parquet files staged under the
gitignored `data/`: exit 0 in 12 min 53 s, paired bootstrap +0.240 with 95% CI [+0.030,
+0.451] and bit-identical fold-seed results, the figures Section 4 prints. The endgame
scripts needed one code change in both repos, the blender's registry and
`members_with_test()` keyed on the validation array instead of the blind one; verified
result-preserving (same 141-member pool as the original code with blind arrays present, all
14 FLEET14 members resolving in the public checkout), details in
`docs/correction/CORRECTION-2026-09.md` section 10. The
leaderboard-citation item that used to sit here is done: no leaderboard citation remains in
the paper, and Section 5 already cites the final boards (winner 85.5 Open / 85.4 Strict, 4 and
11 ranked).

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

## C. Final system, reproduced byte-for-byte (SUPERSEDED, see section -1)

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
(10.0%)**, measured, not quoted.

## D. Fine-tuning fleet, counted from disk

- `scripts/cluster/fleet_manifest.tsv`: **112 rows** (the dispatched grid definition).
- `artifacts/fleet/runs/`: 133 directories; **113** carry a `DONE` marker together with
  both `*_validation.npy` and `*_test.npy`. Those 113 are the completed runs.
- Backbones in the manifest: **13** distinct (`aubmindlab/bert-base-arabertv02`,
  `bert-base-arabertv2`, `bert-large-arabertv02`, `bert-large-arabertv2`,
  `araelectra-base-discriminator`; `CAMeL-Lab/bert-base-arabic-camelbert-msa`, `-ca`;
  `UBC-NLP/MARBERTv2`, `ARBERTv2`; `ahmedabdelali/bert-base-qarib`,
  `bert-base-qarib60_1970k`; `FacebookAI/xlm-roberta-large`; `microsoft/mdeberta-v3-base`).
- Loss families in the manifest: **6**: ce 45, reg 44, softqwk 10, focal 6, corn 5, soft 2.
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
