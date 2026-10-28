# Alternative architecture: word + character TF-IDF

## Hypothesis

The current baseline represents chunks with word-level unigrams and bigrams. Contract
language contains lexical variants, inflections, punctuation-heavy references, and
related forms that may be poorly matched by exact word features. A second `char_wb`
TF-IDF channel (3–5 character n-grams), concatenated with word TF-IDF and trained
with the same one-vs-rest balanced logistic regression, may improve clause recall
and macro F1 without transformer training or inference cost.

This is a testable candidate, not a claim that it beats the baseline or transformer.
The two feature channels are fitted on training chunks only. The candidate uses
sublinear term frequency, Unicode accent normalization, and at most 100,000 features
per channel. All settings are explicit in the report.

## Reproducible comparison

From the repository root, with the pinned tokenizer available in `.venv/hf-cache`:

```sh
HF_HUB_OFFLINE=1 python -m pytest tests/test_word_char_tfidf.py
HF_HUB_OFFLINE=1 python -m scripts.evaluate_word_char_tfidf
```

The runner reads `train_separate_questions.json` only, uses `split_by_contract` with
a fixed seed and the common sentence-aware chunker, and trains the old word-only and
new hybrid models on identical chunks. It writes per-category precision/recall/F1,
threshold JSON files, validation scores, and a report under
`results/word_char_tfidf/`. `test.json` is never loaded. Thresholds are tuned on the
validation set for diagnostic parity with the existing threshold workflow; therefore
validation F1 is optimistic and must not be presented as an unbiased final estimate.

The primary comparison is macro per-category F1; also inspect per-category precision
and recall, especially rare categories, and compare PR-AUC in follow-up evaluation.
Compare against the saved baseline only if its split and chunk settings are identical.
The current saved CSV alone does not establish that comparability. Keep word-only
features as an ablation, then tune character n-gram range, minimum document frequency,
and regularization on a development/calibration protocol before any final holdout.

## Registry and store compatibility

A trained model artifact should bundle both fitted vectorizers (or the fitted feature
union), the classifier, ordered `CATEGORIES`, tokenizer/chunker identity, training
split seed, parameters, package versions, and validation-calibrated thresholds. The
inference interface should accept chunk text and return raw 41-category probabilities;
threshold application stays downstream so it can be changed without fitting again.
The generated NPZ preserves validation truth and raw probabilities for offline
threshold/scoring experiments, but is not a substitute for the SQLite artifact store.
Persist stable contract/chunk IDs and model version with probabilities when that store
is implemented.
