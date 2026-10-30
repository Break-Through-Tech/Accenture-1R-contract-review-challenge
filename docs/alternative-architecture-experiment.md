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
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_word_char_tfidf.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m scripts.evaluate_word_char_tfidf --output results/word_char_tfidf_nested
```

The runner reads `train_separate_questions.json` only. It creates fixed contract-level
training, calibration, and evaluation partitions; fits thresholds on calibration only;
and reports per-category precision/recall/F1 on the untouched evaluation partition.
Both models use the same contracts, chunker, labels, and tokenization. It writes ID
hashes, partition sizes, per-category results, thresholds, raw probabilities, and a
report under `results/word_char_tfidf_nested/`. `test.json` is never loaded. This
supersedes the initial exploratory report below, which tuned thresholds and reported
metrics on the same validation set and is explicitly optimistic.

The runner defaults to five predeclared seeds (`42 7 17 23 101`), producing
contract-level train/calibration/evaluation splits for each seed. Results for seed 42
are in `results/word_char_tfidf_nested/`; the other runs are in sibling directories
named `results/word_char_tfidf_nested_seedN/`. Aggregate per-seed macro-F1 and
per-category deltas are written under the seed-42 directory. Reproduce all five with:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m scripts.evaluate_word_char_tfidf
```

Current observed mean macro-F1 is 0.432 for word-only vs 0.504 for word+character
across five splits; the hybrid wins on all five (mean absolute delta +0.072). Positive
mean per-category deltas occur in 37/41 categories. These are overlapping random
splits, not five statistically independent datasets; this is robustness evidence,
not a confidence interval or a definitive significance claim. Rare categories remain
unstable, and calibration may have no positives for some labels on a given seed.

The comparison uses a freshly trained word-only model with exactly the same partition
and pipeline; it does not prove comparability with historical
`results/tfidf_baseline_metrics.csv` unless that CSV's split and chunking match. The
primary metric is unweighted macro per-category F1, accompanied by per-category
precision, recall, and support. The encoder has not yet been compared. Avoid further
tuning against these evaluation partitions; reserve `test.json` for an agreed final
evaluation and compare with the encoder on aligned data/settings before claiming
production superiority.

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
