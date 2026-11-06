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

## Preprocessing v2: blank-line aware (`features_v2.py`)

`features.normalize_text` collapses any run of line breaks into one `¶`, so `"a\n\nb"` and `"a\nb"` both become `a ¶ b`. In `train.csv` the two kinds of break separate the classes almost perfectly:

| Class | Passages with a blank line | Single line breaks / passage | Blank lines / passage |
|---|---|---|---|
| Human | 0.4% | 5.8 | 0.0 |
| AI | 66.3% | 2.6 | 4.8 |

`test.csv` has the same blank-line rate as `train.csv` (32.1% vs. 33.4% of passages), so the signal should carry over to Kaggle.

- `normalize_text_v2` is identical to v1 except it emits one `¶` per line break (`a ¶ ¶ b`). `build_vectorizer_v2` is the v1 word 1–2 + char 2–5 TF-IDF with that normalizer swapped in. `features.py` is unchanged, so the SVM numbers above stay reproducible.
- `stylometric_features` computes 37 per-passage statistics: counts of characters, words, sentences, paragraphs, line breaks and blank lines; mean/std of sentence, paragraph and word length; type-token and hapax ratios; function-word and repeated-bigram rates; upper-case, digit and capitalized-word rates; invisible-character count and non-ASCII rate; whether the passage ends in `.!?`; and 15 punctuation marks per 1k characters.

## Model: Gradient-boosted trees

```bash
python train_gbdt.py --train data/train.csv --unlabeled data/unlabeled.csv --test data/test.csv --submission submission_gbdt.csv
```

`HistGradientBoostingClassifier` on the 37 stylometric features + a 200-d LSA (`TruncatedSVD` of the v2 TF-IDF). TF-IDF and SVD are unsupervised and fit on the 8,000 training texts + `unlabeled.csv`. Hyperparameters are chosen by 5-fold CV AUROC on the 8,000 training rows over learning rate {0.05, 0.1} × max leaf nodes {15, 31, 63} × L2 {0, 1}. The number of trees is chosen by early stopping (`max_iter=1000`, internal 10% holdout, patience 30). All 12 settings score 0.9974–0.9981; the best is learning rate 0.1, 15 leaves, no L2 (CV AUROC 0.9981, 160 trees). For Kaggle, LSA and GBDT are refit on all 10,000 labeled rows and `predict_proba` is written as the score.

| Model | Text representation | Train AUROC | Val. AUROC | Train time |
|---|---|---|---|---|
| Linear SVM (C=1) | word 1–2 + char 2–5 TF-IDF (v1), 300k features | 1.0000 | 0.9941 | 2.1 s fit (171 s incl. TF-IDF) |
| GBDT (lr 0.1, 15 leaves, 160 trees) | 37 stylometric + 200-d LSA of v2 TF-IDF | 0.99995 | **0.9984** | 1.6 s fit (316 s incl. features) |

Ablation with the same hyperparameters: stylometric features only 0.9977, LSA only 0.9947. Permutation importance on the validation set: shuffling `n_blank_lines` drops AUROC by 0.150, and no other feature exceeds 0.005. The model is therefore mostly a blank-line detector with small corrections from vocabulary and LSA features. It exploits how the passages were formatted, not what they say, and would not survive reformatting (e.g. stripping blank lines).

Results are also saved to `results_gbdt.json`.

## Seeds

- Global seed: `20261023` (validation split, CV folds, LinearSVC, TruncatedSVD, gradient boosting).

## AI assistance

TODO — preprocessing code and ablation were written with Cursor (AI agent); describe what you verified.
