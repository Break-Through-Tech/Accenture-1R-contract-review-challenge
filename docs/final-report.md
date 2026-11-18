# ClauseTriage: final report

Break Through Tech AI Studio and Accenture, Fall 2026. Written 2026-11-18.

> **Research prototype, not legal advice.** The risk ratings come from rules and weights set by the
> team. They have not been validated by a legal expert or compared with a reviewer's judgement.

ClauseTriage finds clauses in the 41 categories of the Contract Understanding Atticus Dataset
(CUAD), rates each Low, Medium or High risk, and ranks contracts for review. Details are in the
guides under [`docs/`](README.md).

## Summary

- **Clause detection was measured once on unseen contracts.** On the 102 CUAD test contracts, read
  once under a protocol fixed in advance, the released model scored macro-F1 0.539 (95% interval
  0.514 to 0.595) and macro-AUPR 0.608 (0.567 to 0.641), consistent with development. Rare
  categories are detected far worse than common ones.
- **The fine-tuned transformer did not beat the TF-IDF baseline.** DistilRoBERTa lost to word plus
  character TF-IDF on all five development splits, so the brief's first success criterion was not
  met by a transformer. The released TF-IDF model does beat the brief's original word-only baseline.
- **Risk rankings have no human validation.** The brief's second criterion, agreement with
  hand-ranked clauses, was not measured: the team decided not to collect hand rankings. Only a
  sensitivity analysis and a check against perfect detection exist.
- **Evidence quotes are the weakest part.** On the test contracts, 23% of detected clauses could not
  be backed by a quote, and 36% of the quotes produced overlap the annotated answer.
- **The software runs end to end on one machine,** but no real reviewer has used it and it has no
  authentication.
- **Time saved is an estimate, not a measurement.** Reading only the passages the model flags would
  defer 40% to 67% of the text for the middle half of test contracts (8.7 to 36.1 minutes for a
  median-length contract, depending on an assumed reading speed). About one annotated clause in five
  sits in the deferred text.

## 1. What was built

| Stage | What it does | Guide |
|---|---|---|
| Ingestion | Reads PDF, Word and pasted text | [Document ingestion](document-ingestion.md) |
| Chunking | Overlapping 512-token windows along sentence boundaries | [Chunking](chunking-experiments.md) |
| Detection | Word and character TF-IDF, one logistic regression and threshold per category | [Baseline](alternative-architecture-experiment.md) |
| Evidence | An exact supporting sentence, or a flag for manual review (an "Unknown"), never a guess | [Clause register](clause-register.md) |
| Risk scoring | Four weighted signals rate each clause; the worst clauses give the contract's triage score | [Risk weights v2](risk-weights-v2.md) |
| Clause register | CSV and JSON per contract | [Clause register](clause-register.md) |
| Serving | Local API and reviewer page, editable weight versions, review queue | [API](api.md), [Reviewer page](reviewer-ui.md), [Review queue](feedback-review-queue.md) |

## 2. How it was evaluated

CUAD's official split has 408 training and 102 test contracts. Development used contract-level
partitions of the training file (seed 42: 260 training, 66 calibration, 82 evaluation), with
thresholds tuned on calibration contracts. Metrics are per-category precision, recall, F1 and AUPR
on 512-token windows (positive when an annotated answer overlaps the window), macro-averaged over
41 categories.

The test file was read once, on 2026-11-18, under a protocol committed beforehand that fixed the
model, thresholds, metrics, a 10,000-replicate contract bootstrap with BCa intervals, and the
wording of the result ([Final test evaluation](final-test-evaluation.md)). The served pipeline gave
identical outputs on all 102 contracts, no test contract shares a title or text with the training
file, and there were no deviations. Disclosed before the run: on 2026-11-17 a text search printed a
match count for the test file; no content was displayed or used.

## 3. Detection results

The registered sentences, with ranges written out with "to":

> On the 102 test contracts, macro-F1 was 0.539 (95% CI 0.514 to 0.595), consistent with
> development: the five development splits ranged from 0.473 to 0.533 (mean 0.504; difference
> +0.035). The point estimate is above every development split.

> On the 102 test contracts, macro-AUPR was 0.608 (95% CI 0.567 to 0.641), consistent with
> development: the five development splits ranged from 0.531 to 0.597 (mean 0.563; difference
> +0.044). The point estimate is above every development split.

| Categories by development support | Count | Test mean F1 | Development mean F1 | Test mean AUPR | Development mean AUPR |
|---|---:|---:|---:|---:|---:|
| Common (30 or more positives) | 19 | 0.650 | 0.653 | 0.702 | 0.696 |
| Uncommon (10 to 29) | 13 | 0.500 | 0.480 | 0.559 | 0.511 |
| Rare (under 10) | 9 | 0.361 | 0.224 | 0.462 | 0.366 |

Five categories have under 10 test positives and get no per-category claim; Price Restrictions has
none and counts as F1 0. Of the other 35, 2 are worse on test than in development and 3 are better;
about two would fall outside by chance ([`summary.md`](../results/final_test_evaluation/summary.md)).

| Secondary analysis | Test (95% interval) | Development |
|---|---|---|
| Original word-only TF-IDF baseline, macro-F1 | 0.480 (0.459 to 0.524) | 0.432 (five-split mean) |
| Released model minus word-only, macro-F1 | +0.058 (0.040 to 0.077) | +0.072 (five-split mean) |
| Contract-level presence, macro-F1 | 0.678 (0.651 to 0.736) | 0.670 (seed-42 split) |
| Contract-level presence, macro-AUPR | 0.794 (0.751 to 0.823) | 0.779 (seed-42 split) |

In the registered wording, word+char beats the original baseline on test. Most false positives
are confusions with a related clause, such as Uncapped Liability against Cap On Liability, which
flips the risk ([Error analysis](error-analysis.md)).

## 4. The fine-tuned encoder did not beat TF-IDF

DistilRoBERTa was fine-tuned for four epochs on a Colab T4 on the same five splits and judged by a
rule fixed in advance ([Encoder comparison](encoder-comparison.md)). The verdict is `does_not_beat`.

| Split seed | TF-IDF macro-F1 | DistilRoBERTa macro-F1 | TF-IDF ahead by (95% interval) |
|---:|---:|---:|---|
| 42 | 0.533 | 0.451 | 0.082 (0.046 to 0.112) |
| 7 | 0.515 | 0.451 | 0.064 (0.027 to 0.093) |
| 17 | 0.494 | 0.431 | 0.063 (0.027 to 0.098) |
| 23 | 0.505 | 0.434 | 0.072 (0.033 to 0.098) |
| 101 | 0.473 | 0.392 | 0.081 (0.050 to 0.106) |
| Mean | 0.504 | 0.432 | 0.072 |

Mean macro-AUPR was 0.563 against 0.460. Mean F1 was lower by 0.021 on common categories, 0.125 on
uncommon and 0.105 on rare ones. Against the word-only baseline the encoder won 2 of 5 splits, with
no interval above zero. It was not run on the test set. Seven other neural designs did not robustly
beat TF-IDF either ([README](../README.md#results-and-key-findings)).

## 5. Risk scoring and contract triage

A clause's score combines category severity (60%), one-sidedness and magnitude read from its quote
(10% each) and contract context (20%). The team wrote the weight set `v2-team-proposed`; no legal
expert has reviewed it, and it cannot tell licensor from licensee
([Risk weights v2](risk-weights-v2.md)).

**The brief's second criterion is not measured,** because the team decided on 2026-11-17 not to
collect hand rankings. Whether the ranking puts the right contracts first is unknown. Weaker checks:

| Check | Result | Contracts |
|---|---|---|
| Ranking against the ranking under perfect detection (Spearman) | 0.777 (0.648 to 0.863); largest move 64; top 10 share 4 | 102 test |
| Same, development | 0.777; largest move 50; top 10 share 2 | 82 development |
| One signal weight moved 10 points: Spearman with base ranking | 0.975 to 0.998 | 82 development |
| High threshold down or up 0.05: ratings changed | 7 or 29 of 1,529 (High clauses 69 or 33, against 62) | 82 development |
| Medium threshold down or up 0.05: ratings changed | 42 or 36 of 1,529 | 82 development |

Perfect detection means the CUAD annotations scored with the same weights, so the first two rows
measure how far detection errors move the ranking, not whether it matches a reviewer. Raising the
High threshold by 0.05 halves the number of High clauses
([`report.json`](../results/contract_triage/report.json)).

## 6. Evidence quotes

| On detected clauses | Test (2,013 clauses) | Development (1,529 clauses) |
|---|---:|---:|
| No quote could be grounded (Unknown) | 23.0% (462) | 19.2% (293) |
| Grounded quote overlaps the annotated answer | 36.4% | 38.3% |
| Same, on correctly detected windows only | 58.3% | 60.1% |

Every quote is exact contract text, but often a nearby sentence rather than the annotated words, and
a false detection still gets a plausible quote. The two signals that read the quote are therefore
the least reliable.

## 7. Estimated reviewer time saved

This is not a measured saving. It estimates the reading a reviewer could **defer** in one contract
by reading every window where the model detects a clause (the whole window, since quotes are
unreliable). It ignores time spent checking flags and gives no credit to the unvalidated ranking.

| Input | Value | Measured or assumed |
|---|---|---|
| W: words in a median contract | 5,357.5 | Measured: [`EDA_SUMMARY.md`](../data/cuad/EDA_SUMMARY.md) (training file) |
| d: share of a contract's text outside flagged windows | median 53.6%; quartiles 40.4% to 67.3%; 58.2% of all test text | Measured, post hoc, test contracts |
| m: share of annotated labels (a window and a category) in deferred text | 18.7% (345 of 1,847) | Measured, post hoc, test contracts |
| R: reading speed | 100, 150 or 250 words per minute | **Assumption** |
| Model time | about 0.3 seconds per contract on CPU | Measured: [`report.json`](../results/final_test_evaluation/report.json) |

`d` and `m` describe the sealed run's saved outputs after the fact and are not part of the registered
evaluation. `scripts/estimate_deferred_reading.py` recomputes them and the grid from
`results/final_test_evaluation/` alone, after checking that it reproduces the run's detections.

Deferred minutes = d × W / R, for example 0.536 × 5,357.5 / 150 = 19.1. A full read takes 53.6, 35.7
or 21.4 minutes at the three speeds.

| Deferred minutes per median contract | R = 100 | R = 150 | R = 250 |
|---|---:|---:|---:|
| d = 40.4% (lower quartile) | 21.7 | 14.4 | 8.7 |
| d = 53.6% (median) | 28.7 | **19.1** | 11.5 |
| d = 67.3% (upper quartile) | 36.1 | 24.0 | 14.4 |

**Range:** 8.7 to 36.1 minutes, about 40% to 67% of the reading; the share does not depend on the
assumed speed. Test contracts are shorter at the median than the training file (25,657 characters
against 35,960.5), and minutes scale with length. The flagged windows include 873 false detections
out of 2,032.

**The cost is missed clauses.** 18.7% of annotated labels lie in deferred text; in 294 of 1,244 cases
(23.6%) where a category is present in a contract, the model does not detect it anywhere in that
contract; 7 of the 102 contracts had no flagged window. This describes a first pass, not a
replacement for reading the contract.

## 8. Limitations

- **One dataset.** The test set is one sample of 102 CUAD contracts and says nothing about other
  sources or about text extracted from PDFs.
- **Small training set.** The released model used 260 of the 408 training contracts. A refitted
  model would be new and unevaluated, and the test set has now been used.
- **Rare categories.** Thresholds rest on 4 to 9 calibration positives, F1 swings by 0.3 to 0.4
  between development splits, and six categories have fewer than 10 test positives.
- **Window-level labels.** Overlap labelling makes some errors artefacts, especially among dates and
  party names at the start of a contract.
- **Risk layer and evidence.** Weights are team judgement with no human validation, quotes are weak,
  and classification errors carry into the ratings.
- **Intervals** capture contract sampling only, not training randomness or the choice of split.
- **Exercised:** served outputs match the evaluated ones; quotes stay grounded on three sample PDFs;
  the API ran with about ten contracts; the reviewer page loaded in headless Chrome, with its other
  controls driven through Gradio's client; the review queue had one scripted run.
- **Not exercised:** detection on PDF text (a CUAD contract as a PDF keeps 7 of 60 answers
  verbatim), real OCR, authentication, time limits, load, and any real reviewer. Keep the API on
  127.0.0.1 ([API](api.md), [Reviewer page](reviewer-ui.md),
  [Document ingestion](document-ingestion.md)).

## 9. Next steps

- Collect hand rankings to measure the second success criterion, and real verdicts.
- Improve evidence extraction and rare-category thresholds
  ([Threshold robustness](threshold-robustness.md)); a changed model needs new held-out contracts.
- Measure detection quality on text extracted from PDFs.
- Before any real use: authentication, time limits, and a legal review of the weights.
