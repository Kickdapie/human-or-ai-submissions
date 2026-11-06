# Group 14 — Human or AI?

**Team members:** TODO (names + NetIDs)  
**Kaggle team name:** TODO

## Setup

Python 3.13 (3.10+ should work).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

Place `train.csv`, `unlabeled.csv`, `test.csv` in `./data/` (not committed).

## Preprocessing → TF-IDF features

```bash
python preprocess.py --train data/train.csv --test data/test.csv --unlabeled data/unlabeled.csv --out-dir features
```

Writes `X_train.npz`, `X_test.npz` (sparse float32, 300,000 columns), `y_train.npy`, `test_ids.csv`, the fitted `vectorizer.joblib`, and `meta.json`.

To reproduce the representation comparison (5-fold stratified CV, ROC-AUC):

```bash
python ablate_features.py --train data/train.csv
```

### Normalization (`features.normalize_text`)

1. Unicode NFKC, strip zero-width characters / soft hyphens / control characters — undoes invisible-character and full-width tricks used in RAID-style adversarial attacks.
2. Replace every digit with `0` — keeps "there is a number here" but not the specific value.
3. Collapse runs of spaces/tabs to one space.
4. Replace each line/paragraph break with a `¶` token — line structure is the single most useful normalization step (see below).
5. **No lowercasing, no stop-word removal, no stemming, punctuation kept** — casing, function words, and punctuation are style signals, which is exactly what separates human from machine text.

### Features (`features.build_vectorizer`)

| View | Tokens | n-grams | Cap |
|---|---|---|---|
| Word TF-IDF | words + each punctuation mark as its own token | 1–2 | 100k, `min_df=2`, `max_df=0.95` |
| Char TF-IDF | raw characters (crossing word boundaries) | 2–5 | 200k, `min_df=2` |

Both use `sublinear_tf=True` (1 + log tf) and L2 normalization, and are horizontally concatenated. The vocabulary/IDF is fit on `train.csv` + `unlabeled.csv` text (no labels are used from unlabeled data).

### Validation results (TF-IDF + logistic regression, C=4, seed 20261023)

| Representation | ROC-AUC |
|---|---|
| sklearn default `TfidfVectorizer()` | 0.9033 |
| + bigrams, sublinear tf | 0.9403 |
| + punctuation tokens, cased | 0.9627 |
| + normalization with `¶` line-break token (final word view) | 0.9886 |
| char 2–5 only | 0.9897 |
| word + char_wb 3–5, lowercased | 0.9827 |
| word + char 2–5, lowercased | 0.9905 |
| **word + char 2–5, cased (final)** | **0.9918 ± 0.0016** |

Removing the `¶` token from the otherwise-final word view drops it to 0.9638, so paragraph/line formatting is a major signal.

## Fixed validation partition (shared by all models)

`val_ids.txt` lists the 2,000 `train.csv` ids held out for validation (stratified, 50% AI, seed `20261023`). Every model trains on the other 8,000 rows and reports AUROC on these 2,000:

```python
from split import load_split
train_df, val_df = load_split(pd.read_csv("data/train.csv"))
```

Fit any vocabulary/IDF/scaler on `train_df` (+ `unlabeled.csv` text) only, never on `val_df`.

## Model: Linear SVM

```bash
python train_svm.py --train data/train.csv --unlabeled data/unlabeled.csv --test data/test.csv --submission submission_svm.csv
```

`LinearSVC` (squared hinge loss, L2 penalty) on the word 1–2 + char 2–5 TF-IDF features. `C` is chosen by 5-fold CV AUROC on the 8,000 training rows over {0.03, 0.1, 0.3, 1, 3}; C=1 and C=3 tie at 0.9916, and the tie goes to the more regularized C=1. AUROC is computed from `decision_function` scores. For Kaggle, the model is refit on all 10,000 labeled rows and scores are mapped through a sigmoid (monotonic, so AUROC is unchanged).

| Model | Text representation | Train AUROC | Val. AUROC | Train time |
|---|---|---|---|---|
| Linear SVM (C=1) | word 1–2 + char 2–5 TF-IDF, 300k features | 1.0000 | 0.9941 | 2.1 s fit (171 s incl. TF-IDF) |

Results are also saved to `results_svm.json`.

## Seeds

- Global seed: `20261023` (validation split, CV folds, LinearSVC).

## AI assistance

TODO — preprocessing code and ablation were written with Cursor (AI agent); describe what you verified.
