# ClauseTriage: final report

Break Through Tech AI Studio and Accenture, Fall 2026. Written 2026-11-18.

> **Research prototype, not legal advice.** The risk ratings come from rules and weights set by the
> team. They have not been validated by a legal expert or compared with a reviewer's judgement.

ClauseTriage reads a commercial contract, finds clauses in the 41 categories of the Contract
Understanding Atticus Dataset (CUAD), rates each clause it finds Low, Medium or High risk, and
ranks contracts by a triage score so a reviewer can choose which to open first. This report covers
what was built, how well it works, what it cannot do, and an estimate of the reading it could let a
reviewer defer. Each section links to the guide under [`docs/`](README.md) that holds the detail.

## Summary

- **Clause detection was measured once on unseen contracts.** On the 102 CUAD test contracts, read
  once under a protocol fixed in advance, the released model scored macro-F1 0.539 (95% interval
  0.514 to 0.595) and macro-AUPR 0.608 (0.567 to 0.641), consistent with development. Common
  categories are detected far better than rare ones.
- **The fine-tuned transformer did not beat the TF-IDF baseline.** A five-seed DistilRoBERTa run
  lost to word plus character TF-IDF on every seed, so the brief's first success criterion was not
  met by a transformer. The released model is the TF-IDF model, which does beat the brief's original
  word-only baseline on the test contracts.
- **Risk rankings have no human validation.** The brief's second criterion, agreement with
  hand-ranked clauses, was not measured: the team decided not to collect hand rankings. What exists
  is a sensitivity analysis and a check of how far detection errors move the ranking.
- **Evidence quotes are the weakest part.** On the test contracts, 23% of detected clauses could not
  be backed by a quote, and 36% of the quotes produced overlap the annotated answer.
- **The software runs end to end on one machine** (upload, scoring, clause registers, an API and a
  reviewer page), but no real reviewer has used it and it has no authentication.
- **Time saved is an estimate, not a measurement.** Reading only the passages the model flags would
  defer about 40% to 67% of a contract's text for the middle half of test contracts (8.7 to 36.1
  minutes for a median-length contract, depending on an assumed reading speed). About one annotated
  clause passage in five sits in the deferred text.

## 1. What was built

| Stage | What it does | Guide |
|---|---|---|
| Ingestion | Reads PDF (optional OCR for scanned files), Word and pasted text, with size, page and text limits. Positions and page numbers refer to the extracted text. | [Document ingestion](document-ingestion.md) |
| Chunking | Splits the text into overlapping windows of up to 512 tokens along sentence boundaries. On the test contracts, 2,627 of 2,643 annotated answers (99.4%) fall inside one window. | [Chunking](chunking-experiments.md) |
| Detection | Word and character TF-IDF features, one logistic regression per category, and a per-category threshold tuned on calibration contracts. | [Baseline experiment](alternative-architecture-experiment.md) |
| Evidence | Picks a supporting sentence for each detected clause and checks it character for character against the contract text. If none qualifies, the clause is listed without a quote and flagged for manual review (an "Unknown"); nothing is guessed. | [Clause register](clause-register.md) |
| Risk scoring | Four signals (category severity, one-sidedness, magnitude, contract context) combined with a versioned weight set into a clause score and a Low, Medium or High rating. A contract's triage score comes from its worst clauses. | [Risk weights v2](risk-weights-v2.md) |
| Clause register | One CSV and one JSON file per contract: every detected clause, its rating, the reasons, its position and its quote. | [Clause register](clause-register.md) |
| Serving | A local API and a reviewer page: upload and rank contracts, open details, download registers, save edited weights as new versions, and record accept, reject or missing verdicts in a review queue. | [API](api.md), [Reviewer page](reviewer-ui.md), [Review queue](feedback-review-queue.md) |

## 2. How it was evaluated

- **Data.** CUAD v1: 510 contracts, with the official split of 408 training and 102 test contracts.
  All development work used contract-level partitions of the training file (for seed 42: 260
  training, 66 calibration, 82 evaluation contracts) and five seeded splits for comparisons.
  Thresholds were tuned on calibration contracts only.
- **Metrics.** Per-category precision, recall, F1 and AUPR on 512-token windows (a window is
  positive for a category when an annotated answer overlaps it), and their macro averages over the
  41 categories. Accuracy is not reported, because most windows contain no given clause.
- **One sealed test run.** The test file was read once, on 2026-11-18, under a protocol committed
  before it was opened ([Final test evaluation](final-test-evaluation.md)). The protocol fixed the
  model, thresholds, metrics, a 10,000-replicate contract bootstrap with BCa intervals, and the
  sentences used to report the result. Before reading the data, the run confirmed that the released
  model reproduced its saved calibration probabilities (largest difference 3.6e-7). During the run,
  every contract was also scored through the served pipeline, with identical outputs. No test
  contract shares a title or text with the training file. There were no deviations. One disclosure,
  recorded before the run: on 2026-11-17 a repository-wide text search printed a match count for
  the test file; no content was displayed or used.

## 3. Detection results

The protocol fixed the wording of these sentences before the data was read. They are quoted with
ranges written out with "to":

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

On test, 35 categories have 10 or more positive windows and are reported. Five have too few for a
per-category claim (Joint Ip Ownership, Non-Disparagement, Unlimited/All-You-Can-Eat-License, Most
Favored Nation, Source Code Escrow). Price Restrictions has none and counts as F1 0 in macro-F1.
Under the registered per-category rule, 2 of the 35 reported categories are worse on test than in
development (Document Name, Effective Date) and 3 are better (Affiliate License-Licensee, Affiliate
License-Licensor, Non-Compete). With 41 intervals, about two will fall outside by chance. Per-category
F1 runs from 0.869 (Governing Law) down to 0.141 (Unlimited/All-You-Can-Eat-License, 5 positives);
the full table is in [`summary.md`](../results/final_test_evaluation/summary.md).

Secondary analyses, reported descriptively as the protocol requires:

| | Test (95% interval) | Development |
|---|---|---|
| Original word-only TF-IDF baseline, macro-F1 | 0.480 (0.459 to 0.524) | 0.432 (five-split mean) |
| Released model minus word-only, macro-F1 | +0.058 (0.040 to 0.077) | +0.072 (five-split mean) |
| Contract-level presence ("does this contract contain X?"), macro-F1 | 0.678 (0.651 to 0.736) | 0.670 (seed-42 split) |
| Contract-level presence, macro-AUPR | 0.794 (0.751 to 0.823) | 0.779 (seed-42 split) |

In the registered wording: word+char beats the original baseline on test. The
[error analysis](error-analysis.md) on development data found that most false positives are
confusions with a related clause (61% to 66% of them, in the two models analysed, fall on a window
labelled with a different category), such as Uncapped Liability against Cap On Liability, which
flips the direction of the risk.

## 4. The fine-tuned encoder did not beat TF-IDF

The brief's first success criterion was that a fine-tuned transformer clearly beats the TF-IDF
baseline. DistilRoBERTa was fine-tuned for four epochs on a Colab T4 on the same five splits and
windows, and judged by a rule fixed before any result existed
([Encoder comparison](encoder-comparison.md), [`summary.md`](../results/encoder_vs_tfidf/vs_word_char_tfidf/summary.md)).
The verdict is `does_not_beat`.

| Split seed | TF-IDF macro-F1 | DistilRoBERTa macro-F1 | TF-IDF ahead by (95% interval) |
|---:|---:|---:|---|
| 42 | 0.533 | 0.451 | 0.082 (0.046 to 0.112) |
| 7 | 0.515 | 0.451 | 0.064 (0.027 to 0.093) |
| 17 | 0.494 | 0.431 | 0.063 (0.027 to 0.098) |
| 23 | 0.505 | 0.434 | 0.072 (0.033 to 0.098) |
| 101 | 0.473 | 0.392 | 0.081 (0.050 to 0.106) |
| Mean | 0.504 | 0.432 | 0.072 |

Mean macro-AUPR was 0.563 for TF-IDF and 0.460 for DistilRoBERTa. The gap is mostly in categories
with few examples: mean F1 was lower by 0.021 on common categories, 0.125 on uncommon and 0.105 on
rare ones. Against the brief's original word-only baseline the encoder won 2 of 5 splits, with no
interval above zero. The encoder was not run on the test set; its checkpoints are not in the
repository.

Seven other neural designs were screened on the development splits: a retrieval reranker, a
contract-wide Transformer, a category-query model, a graph network, contrastive prototypes, set
prediction and sparse long context. None robustly beat TF-IDF. The closest, the reranker, gained
0.016 mean macro-F1 over five splits, with bootstrap intervals that all include zero. The seed-42
numbers are in the [README](../README.md#results-and-key-findings).

## 5. Risk scoring and contract triage

Each detected clause gets a score from four signals: a severity tier for its category (60% of the
weight), one-sidedness and magnitude read from its quote (10% each), and contract context (20%).
The weight set `v2-team-proposed` was written by the team from the CUAD category definitions. No
one with legal or procurement expertise has reviewed it, and it rates from a neutral perspective
that cannot tell licensor from licensee ([Risk weights v2](risk-weights-v2.md)).

**The brief's second criterion is not measured.** It asked for agreement between the model's risk
rankings and hand-ranked clauses. The team decided on 2026-11-17 not to collect hand rankings, so
whether the ranking puts the right contracts first is unknown. Two weaker checks exist:

| Check | Result | Contracts |
|---|---|---|
| Ranking from detected clauses against the ranking under perfect detection (Spearman) | 0.777 (0.648 to 0.863); largest rank move 64; 4 of the top 10 shared | 102 test |
| Same check on development | 0.777; largest rank move 50; 2 of the top 10 shared | 82 development |
| Any one signal weight moved by 10 points: Spearman with the base ranking | 0.975 to 0.998 | 82 development |
| High threshold moved down or up by 0.05: clause ratings changed | 7 or 29 of 1,529 (High clauses 69 or 33, against 62) | 82 development |
| Medium threshold moved down or up by 0.05: clause ratings changed | 42 or 36 of 1,529 | 82 development |

The perfect-detection reference uses the CUAD annotations scored with the same weights, so it
measures how far detection errors move the ranking, not whether the ranking matches a reviewer.
The ranking changes little when one weight moves. The High/Medium boundary changes few ratings
when lowered, but raising it by 0.05 halves the number of High clauses. Development numbers are
from [`report.json`](../results/contract_triage/report.json).

## 6. Evidence quotes

| On detected clauses | Test (2,013 clauses) | Development (1,529 clauses) |
|---|---:|---:|
| No quote could be grounded (Unknown) | 23.0% (462) | 19.2% (293) |
| Grounded quote overlaps the annotated answer | 36.4% | 38.3% |
| Same, on correctly detected windows only | 58.3% | 60.1% |

Every quote shown is exact contract text, but it is often a sentence near the clause rather than
the annotated words, and a false detection still gets a plausible quote. The one-sidedness and
magnitude signals read the quote, which is why the weights lean on severity and context. In a hand
run on six contracts, the review queue held 27 Unknowns, all because no sentence matched the
category's cue words ([Review queue](feedback-review-queue.md)).

## 7. What has and has not been exercised

| Component | Exercised | Not exercised |
|---|---|---|
| Detection as served | Outputs identical to the evaluated ones on all 102 test contracts | Detection quality on text extracted from PDFs |
| PDF reading | Three sample PDFs: every quote matches the text and its page. One CUAD contract as a PDF: 7 of 60 answers survive verbatim, 60 of 60 ignoring whitespace and curly quotes | Real OCR (tested with mocked modules only) |
| API | Unit tests; about ten stored contracts | Authentication (none), time limits on parsing or scoring (none), load, concurrent writers |
| Reviewer page | Loaded in headless Chrome with ten contracts (renders, tabs open, registers download); other controls driven through Gradio's client against a running server | Clicking uploads, weight edits and verdicts in a browser; any real reviewer |
| Review queue and feedback | Unit tests; one scripted run on six contracts, including a reviewer conflict and the training export | Real reviewer verdicts; training on exported feedback |

The API binds to 127.0.0.1 and must stay on a local machine ([API](api.md),
[Reviewer page](reviewer-ui.md)).

## 8. Estimated reviewer time saved

No reviewer has used the tool, so nothing here is a measured saving. This is a small model of the
reading that triage could let a reviewer **defer** within one contract: text they would read later,
or not at all. It does not count time spent checking flags or using the tool, and it gives no credit
to the contract ranking, which is unvalidated.

**Scenario.** The reviewer reads every passage in which the model detects at least one clause and
defers the rest. They read the whole passage, not only the quote, because only 36% of quotes overlap
the annotated answer and 23% of detections have none.

| Input | Value | Measured or assumed |
|---|---|---|
| W: words in a median contract | 5,357.5 | Measured: [`EDA_SUMMARY.md`](../data/cuad/EDA_SUMMARY.md) (training file) |
| d: share of a contract's text outside flagged passages | median 53.6%; quartiles 40.4% to 67.3%; 58.2% of all test text pooled | Measured, post hoc, on the test contracts |
| m: share of annotated clause labels (a window and a category) inside deferred text | 18.7% (345 of 1,847) | Measured, post hoc, on the test contracts |
| R: reading speed for contract text | 100, 150 or 250 words per minute | **Assumption**, not measured; chosen to bracket slow careful reading and quick reading |
| Model time | about 0.3 seconds per contract on the CPU of the evaluation machine (23.5 s chunking and 5.8 s prediction for 102 contracts) | Measured: [`report.json`](../results/final_test_evaluation/report.json) |

`d` and `m` were computed after the sealed run from its saved outputs
([`predictions.npz`](../results/final_test_evaluation/predictions.npz), `chunks.csv`, and the
thresholds in `per_category_metrics.csv`): a window is flagged when any category's probability
reaches its threshold, and overlapping flagged windows are merged before measuring their share of
each contract. This reproduces the run's 2,032 detections. It is a post hoc description and not
part of the registered evaluation.

**Arithmetic.** Deferred reading = d × W / R. For the central case,
0.536 × 5,357.5 / 150 = 19.1 minutes. Reading the whole median contract takes 53.6, 35.7 or 21.4
minutes at the three speeds.

| Deferred minutes per median contract | R = 100 | R = 150 | R = 250 |
|---|---:|---:|---:|
| d = 40.4% (lower quartile) | 21.7 | 14.4 | 8.7 |
| d = 53.6% (median) | 28.7 | **19.1** | 11.5 |
| d = 67.3% (upper quartile) | 36.1 | 24.0 | 14.4 |

**Range:** 8.7 to 36.1 minutes per median-length contract, centred on about 19 minutes, or roughly
40% to 67% of the reading. The share does not depend on the assumed reading speed and is the more
portable number.

**The cost is missed clauses.** 18.7% of annotated clause labels lie in the deferred text. At the
contract level, in 294 of 1,244 cases (23.6%) where a category is present in a contract, the model
does not detect that category anywhere in it
([`contract_level_metrics.csv`](../results/final_test_evaluation/contract_level_metrics.csv)). Seven of
the 102 contracts had no flagged passage at all. A reviewer who never reads the deferred text would
miss these clauses, so the estimate describes a first pass, not a replacement for reading.

Other caveats: the median word count comes from the training file, while the test contracts are
shorter at the median (25,657 characters against 35,960.5); minutes scale with length. The flagged
text includes false detections (873 of the 2,032). A clause that spans two windows counts twice.

## 9. Limitations

- **One dataset.** All numbers come from CUAD. The test set is one sample of 102 contracts from the
  same release and says nothing about contracts from other sources or about text extracted from PDFs.
- **Small training set for the released model.** It was trained on 260 of the 408 training
  contracts. A model refitted on all 408 would be new and unevaluated, and the test set has now been
  used.
- **Rare categories.** Thresholds rest on 4 to 9 calibration positives, F1 swings by 0.3 to 0.4
  between development splits, and six categories have fewer than 10 test positives.
- **Window-level labels.** A window counts as positive when any annotated answer overlaps it, which
  produces some errors that are labelling artefacts, especially among dates and party names at the
  start of a contract.
- **Risk layer.** Weights are team judgement, rate from a neutral perspective, and have no human
  validation. Classification errors between related categories carry into the ratings.
- **Evidence.** Quotes are weak, as in section 6.
- **Intervals.** The bootstrap captures contract sampling only, not training randomness or the
  choice of split.
- **Software.** A local prototype without authentication, time limits or load testing, never used
  by a reviewer. It is not legal advice.

## 10. Next steps

- Collect reviewer judgement: hand rankings on a sample of contracts to measure the second success
  criterion, and real verdicts through the review queue.
- Improve evidence extraction, the weakest stage, and the rare-category thresholds
  ([Threshold robustness](threshold-robustness.md) has candidate methods). Any change needs new
  held-out contracts, because the test set has been used.
- Measure detection quality on text extracted from PDFs.
- Before any real use: authentication, time limits on parsing and scoring, and a legal review of
  the weights.
