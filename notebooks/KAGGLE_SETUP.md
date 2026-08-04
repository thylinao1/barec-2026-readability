# Phase 3 Kaggle fine-tunes - step by step

Goal: train diverse backbones the public checkpoints lack (MARBERT, AraELECTRA,
a CORAL head) to push the ensemble from 86.1 toward ~87.5. Free on Kaggle T4.
Each member only needs val + test predictions (the blender tunes on val, evals on
test), so it is ONE ~30-min training run per member - not hours.

Every member is scored through the local harness (local == board is proven), so
we know the exact board impact before uploading anything.

---

## 0. One-time: enable GPU (you already have an account)
- kaggle.com -> your avatar -> Settings -> under "Phone verification", verify.
  This unlocks the free GPU (T4 x2 / P100, ~30 GPU-hours/week). Without it,
  notebooks run CPU-only (too slow).

## 1. Upload the data once (as a Dataset)
- Desktop file to upload: `~/Desktop/barec_kaggle_data.zip` (I created it).
- kaggle.com -> "Create" (or "+") -> "New Dataset".
- Drag `barec_kaggle_data.zip` in. Kaggle auto-extracts it.
- Title it exactly: **barec-2026-open**  (this sets the mount path). Create.
- After it finishes, the files live at `/kaggle/input/barec-2026-open/`.

## 2. Train member 1 (MARBERTv2) - repeat for each member
- kaggle.com -> "Create" -> "New Notebook".
- Right panel: Settings -> Accelerator = **GPU T4 x2**; Internet = **On**.
- Right panel: "+ Add Input" -> Datasets -> add your **barec-2026-open**.
- Delete the default cell. Paste the ENTIRE contents of `kaggle_finetune.py`
  (I'll give it to you / it is in notebooks/). The CONFIG block up top is already
  set for member 1:
      BACKBONE = "UBC-NLP/MARBERTv2"
      OUT_NAME = "marbertv2-reg"
- Click **Run All**. It trains ~5 epochs (~30 min) and prints "DONE".
- Right panel "Output" (or "Data" -> /kaggle/working) -> download
  `marbertv2-reg_validation.npy` and `marbertv2-reg_test.npy`.

## 3. Give me the files
- Drop both .npy into `~/Developer/barec-2026-sentence-open/artifacts/scores/`
  and tell me. I run `scripts/blend_all.py`, which re-selects members, re-tunes
  the rounder, and reports the exact new board QWK + writes a fresh
  prediction.zip. Upload it only if it beats the current 86.1 (we'll know first).

## 4. Repeat for more diverse members (one at a time)
Just change the two CONFIG lines and re-run a new notebook:
- Member 2: `BACKBONE="aubmindlab/araelectra-base-discriminator"`, `OUT_NAME="araelectra-reg"`
- Member 3 (optional): `BACKBONE="UBC-NLP/ARBERTv2"`, `OUT_NAME="arbertv2-reg"`
(QARiB is already free from public checkpoints, so no need to train it.)

Each is a different family = the low-correlation diversity that actually raises
QWK. Expected: +1 -> ~86.5-86.8, +2 -> ~86.8-87.2, +3 -> ~87.0-87.5.

---

Note on the ceiling: readability labels carry inherent annotator disagreement, so
the practical QWK max is ~87-89 and a literal 100 is not reachable. The real win
is sitting at/above the winner's 87.5 with blind-test margin.
