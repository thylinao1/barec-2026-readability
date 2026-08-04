"""Phase 1 - classical floor. Validates the whole pipeline end to end.

Memory-light for the 8GB fanless M2: uses a STATELESS HashingVectorizer (no
learned vocabulary, no per-fold refit, leakage-free) so the sparse feature matrix
is built ONCE and folds index into it. surface features + hashed char/word
n-grams -> Ridge raw-level regression -> grouped 5-fold OOF ->
OptimizedRounder. Reports the HONEST nested-OOF QWK (rounder go/no-go), the
full-OOF single-rounder QWK, and naive-round QWK for reference. Then trains on
all of train, predicts validation (board mirror, has gold) and test (open-test
target), and writes the prediction.zip artifact.

Rounder cutpoints are tuned on TRAIN OOF only. Validation/test are never tuned on.

    PYTHONWARNINGS=ignore ~/mac-ml-setup/.venv/bin/python -u scripts/phase1_classical.py
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.cv import fold_of_rows, load_fold_map  # noqa: E402
from harness.data import GROUP, ID, TARGET, load_2026  # noqa: E402
from harness.features import to_matrix  # noqa: E402
from harness.metrics import fold_qwk  # noqa: E402
from harness.rounder import OptimizedRounder, apply_cutpoints, nested_oof_qwk  # noqa: E402
from harness.submission import write_codabench_zip  # noqa: E402

SEED = 42
SCORES = ROOT / "artifacts" / "scores"
SUBS = ROOT / "submissions" / "phase1"

# Stateless hashing vectorizers (no fit; no vocabulary held in memory).
CHAR_VEC = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             n_features=2 ** 14, alternate_sign=False, norm="l2")
WORD_VEC = HashingVectorizer(analyzer="word", ngram_range=(1, 2),
                             n_features=2 ** 13, alternate_sign=False, norm="l2")


def featurize(df):
    texts = df["Sentence"].fill_null("").to_list()
    num = sp.csr_matrix(to_matrix(df))
    X = sp.hstack([num, CHAR_VEC.transform(texts), WORD_VEC.transform(texts)], format="csr")
    return X.astype(np.float32)


def model():
    # Ridge on l2-normalized hashed TF features: fast + light + deterministic.
    # This floor validates plumbing/tracking; Phase 2 checkpoints carry the win.
    return Ridge(alpha=2.0, solver="auto")


def main():
    SCORES.mkdir(parents=True, exist_ok=True)
    SUBS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train, val, test = load_2026("train"), load_2026("validation"), load_2026("test")
    y = np.array(train[TARGET].to_list(), dtype=int)
    dmap = load_fold_map(ROOT / "artifacts" / "folds_2026_train.json")
    fold_idx = fold_of_rows(dmap, train[GROUP].to_list())

    Xtr = featurize(train)
    print(f"train feature matrix {Xtr.shape} nnz={Xtr.nnz} ({time.time()-t0:.0f}s)", flush=True)

    # ---- grouped OOF (index into the prebuilt matrix; no refit) ----
    oof = np.zeros(len(y), dtype=float)
    for f in range(5):
        tr, te = fold_idx != f, fold_idx == f
        m = model().fit(Xtr[tr], y[tr])
        oof[te] = m.predict(Xtr[te])
        q = fold_qwk(y[te], np.clip(np.rint(oof[te]), 1, 19))
        print(f"  fold {f}: naive fold_qwk={q:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        del m
        gc.collect()

    naive_qwk = fold_qwk(y, np.clip(np.rint(oof), 1, 19))
    nested_qwk, _ = nested_oof_qwk(oof, y, fold_idx, n_candidates=200)
    full_rounder = OptimizedRounder(n_candidates=300).fit(oof, y)
    full_oof_qwk = fold_qwk(y, full_rounder.predict(oof))
    print("\n=== PHASE 1 grouped-OOF QWK (train) ===", flush=True)
    print(f"  naive round(score):          {naive_qwk*100:.4f}%")
    print(f"  nested-OOF rounder (HONEST):  {nested_qwk*100:.4f}%   <- go/no-go")
    print(f"  full-OOF single rounder:      {full_oof_qwk*100:.4f}%   (optimistic)")

    # ---- final model on all train -> val/test ----
    final = model().fit(Xtr, y)
    del Xtr
    gc.collect()
    va_score = final.predict(featurize(val))
    te_score = final.predict(featurize(test))
    cut = full_rounder.cutpoints_
    va_pred, te_pred = apply_cutpoints(va_score, cut), apply_cutpoints(te_score, cut)

    va_y = np.array(val[TARGET].to_list(), dtype=int)
    te_y = np.array(test[TARGET].to_list(), dtype=int)
    print("\n=== board-mirror QWK (rounder tuned on train OOF only) ===", flush=True)
    print(f"  validation (7310): {fold_qwk(va_y, va_pred)*100:.4f}%   <- primary board mirror")
    print(f"  test (7286):       {fold_qwk(te_y, te_pred)*100:.4f}%   (bonus check, NOT tuned on)")

    np.save(SCORES / "phase1_oof.npy", oof)
    np.save(SCORES / "phase1_val.npy", va_score)
    np.save(SCORES / "phase1_test.npy", te_score)
    zip_path = write_codabench_zip(SUBS, test[ID].to_list(), te_pred)
    print(f"\nsaved OOF/val/test scores to {SCORES}", flush=True)
    print(f"wrote submission artifact: {zip_path}  (NOT uploaded - human gate)")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
