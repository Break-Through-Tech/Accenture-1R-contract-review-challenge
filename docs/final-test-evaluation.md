# Final held-out evaluation on `test.json`: pre-registered protocol

**Status: registered, not run.** This page fixes every choice of the one sealed evaluation of the
frozen model on the 102 CUAD test contracts before the test file is opened. The team authorised
exactly one such evaluation. Nothing on this page may change after the preflight starts, except
the Results section (section 12), which the run fills in, and dated amendments (section 11).

Script: `scripts/evaluate_final_test.py`, helpers in `src/holdout_metrics.py`, `src/holdout_report.py`
and `src/sealed_run.py`. Tests: `tests/test_evaluate_final_test.py`, `test_holdout_metrics.py`,
`test_holdout_report.py`, `test_sealed_run.py` (none of them read `test.json`).

## 1. Question and what is primary

**Primary question:** how well does the released model detect the 41 CUAD categories in contracts
it has never seen? The answer is the **chunk-level detection quality of the frozen bundle at its
own thresholds**, reported as per-category precision, recall, F1 and AUPR, and as macro-F1 and
macro-AUPR, each with a contract-bootstrap 95% interval. Macro-F1 is the headline, with macro-AUPR
next to it. Accuracy is never reported.

Everything in section 6 is **secondary**. It is descriptive, gets no verdict, and never replaces
the primary result.

## 2. Frozen configuration

| Item | Frozen value |
|---|---|
| Test data | `data/cuad/test.json`, git blob `16cb5c6aa4733bcaf46e24a1ee868b47a6731461` (unchanged since the initial commit `d0a04de`, 2026-08-20); 102 contracts expected. Its SHA-256 is computed by the preflight and must be identical at the run. |
| Model | `models/word_char_tfidf.joblib`, release `model-word-char-tfidf-v1`, SHA-256 `6fd84c28600fcac844b488c1c404aa56fed7de9daab007589c9b42a34f0bd194` (`BUNDLE_SHA256` in `src/registry/demo_model.py`). Word (1-2 gram) + char_wb (3-5 gram) TF-IDF, one-vs-rest logistic regression, C = 1.0, balanced class weights. |
| Training data of the model | The 260 training contracts of `split_by_contract(train_separate_questions.json, 0.2, seed 42)` then `(0.2, seed 42)`; contract-ID digests `44e5d6e9...` (training), `271e0681...` (calibration), `495488d5...` (evaluation). Training file SHA-256 `7de21d2b...dab4`. |
| Thresholds | The bundle's own 41 thresholds, equal to `results/word_char_tfidf_nested/word_char_tfidf_thresholds.json`, F1-optimal on the 66 calibration contracts. Unchanged. A chunk is detected when probability >= threshold (float64), as `ClauseModelBundle.predict` does. |
| Chunker | `chunk_sentence_windows`, 512 tokens, 64 overlap, tokenizer `distilbert/distilroberta-base` revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`, loaded only from `.venv/hf-cache` (`local_files_only`). |
| Labels | `load_cuad` (category = question id after the last `__`, trailing `_N` removed); a chunk is positive for a category when any answer span of that category shares at least one character with it (`spans_overlap`). Spans that straddle chunks label every chunk they touch. |
| Category order | `CATEGORIES` (41, alphabetical). |
| Risk weights | `v2-team-proposed`, content SHA-256 `8d8e8e26...0995` (`src/risk/weights/default_weights_v2.json`). |
| Pinned inputs | Line-ending-normalised SHA-256 of both threshold files, the weight file and `category_descriptions.csv` (`PINNED_FILES`); any difference stops the run before the data is read. |
| Bootstrap | 10,000 replicates; each draws 102 contracts with replacement, `numpy.random.default_rng(20261147).integers(0, n, size=n)` (the project rule 20261105 + split seed 42); every statistic uses the same draws. The registered interval is the **BCa 95% interval** (bias correction from the share of replicates below the estimate, acceleration from a leave-one-contract-out jackknife); plain 2.5th/97.5th percentile bounds are stored alongside. Where BCa is undefined (for example, every replicate on one side of the estimate) the percentile interval is used and marked. |
| Libraries | As pinned in `requirements*.txt` (numpy 2.5.1, scikit-learn 1.9.0, scipy 1.18.0, pandas 3.0.6, joblib 1.6.0, transformers 4.57.6, tokenizers 0.22.2); the run records the installed versions. |

The run also checks, before reading the data, that the bundle reproduces the saved seed-42
calibration probabilities within 1e-5 and that the bundle's thresholds and config match the table.

## 3. Metric definitions

- **Per category:** support (positive chunks), TP/FP/FN, precision, recall and F1 at the frozen
  threshold (0 when undefined, as `zero_division=0`), and AUPR (`average_precision_score`, undefined
  when the category has no positive chunk).
- **Macro-F1:** mean F1 over all 41 categories. A category with no test positives contributes 0
  (the project's fixed-41 convention).
- **Macro-AUPR:** mean AUPR over categories with at least one test positive.
- **Intervals:** contract-bootstrap BCa intervals as in section 2, with thresholds fixed. Point
  estimates are always computed on the full test set, never taken from the bootstrap. In a
  replicate, macro-AUPR averages the categories that have positives in that replicate; per-category
  AUPR intervals use the replicates where the category is defined (the count is reported).
- **Why BCa, not the percentile interval:** the development dry run (section 9) showed that the
  bootstrap distribution of macro-AUPR sits about 0.019 above the point estimate. Rare, low-AUPR
  categories drop out of 55% of replicates, and AP rises when contracts are duplicated. Macro-F1 sits
  about 0.006 below it. On the dry run, the percentile interval for macro-AUPR, [0.578, 0.658] around
  0.597, would have read "higher than development", although the five-seed mean 0.563 is plausible.
  BCa corrects for that shift. The project's earlier intervals were percentile intervals on paired
  *differences*, where the shift largely cancels; their percentile bounds are still reported here
  for comparison. This choice was made on development data, before `test.json` was opened.
- Also reported, descriptively: macro-F1 over categories with positives.

## 4. Low-support categories

- **Unscorable:** no test positives. Shown with AUPR "n/a", counted as F1 0 in macro-F1, excluded
  from macro-AUPR, listed by name. Nothing is said about how the model does on them.
- **Insufficient support:** 1-9 test positives. Metrics and intervals are shown, but no
  per-category claim is made about them.
- **Reported:** 10 or more test positives.
- **Buckets** (common 30+, uncommon 10-29, rare <10) are defined by mean development support over
  the five seeds, so membership is fixed now (19 / 13 / 9 categories, as in
  `results/error_analysis/category_summary.csv`). The bucket table compares mean test F1/AUPR
  with the development means of the same categories.

## 5. Reading the primary result (fixed now)

**Development reference** (recomputed from the saved runs `results/word_char_tfidf_nested[_seed*]`,
each seed with its own calibration thresholds, and pinned in the tests):

| | Five-seed mean | Range of the 5 seeds | Seed 42 (the released model's split) |
|---|---:|---:|---:|
| Macro-F1 | 0.5040 | 0.4727-0.5331 | 0.5331 |
| Macro-AUPR | 0.5633 | 0.5312-0.5971 | 0.5971 |
| Word-only macro-F1 | 0.4319 | 0.3963-0.4760 | 0.4760 |

The five-seed mean is the reference. Seed 42 was the best of the five splits, so quoting 0.533 as
"the development number" would overstate it. (`CLAUDE.md` quotes 0.505, which comes from the
reranker analysis, where seeds 17 and 101 used re-fitted thresholds. The nested runs' own value is
0.504.)

For each of macro-F1 and macro-AUPR, with test value T and interval [L, U], the script assigns
the reading mechanically (`compare_with_reference`) and the report uses the matching sentence:

| Condition | Reading | Sentence |
|---|---|---|
| L <= reference <= U | consistent | "On the 102 test contracts, macro-F1 was T (95% CI L-U), consistent with the development estimate of 0.504." |
| U < reference | lower | "... was T (95% CI L-U), lower than the development estimate of 0.504: the development splits overestimated performance on new contracts by about 0.504 - T." |
| L > reference | higher | "... was T (95% CI L-U), higher than the development estimate of 0.504; the development splits were conservative." |

Then add "and below every development split" (T < 0.4727) or "above every development split"
(T > 0.5331) when that applies. Explanations for a gap (for example, different contract types in
the test set) may only be offered as hypotheses, labelled as such, and are never tested on the
test set.

**Per category:** a category with "reported" support is called worse (better) on test only when
its five-seed development mean F1 is above (below) its test 95% interval. The number of such
categories is reported. With 41 intervals, about two will fall outside by chance, so a single
category is not over-interpreted.

## 6. Secondary analyses

All four use only settings already fixed in section 2. None is a verdict on the model.

| | Analysis | Exactly how | What is said |
|---|---|---|---|
| S1 | **Original word-only TF-IDF baseline** as a reference point | Refit `fit_tfidf_baseline` (`src/models/naive/tfid_baseline.py`, default settings: 10,000 word 1-2 grams, English stop words, balanced logistic regression) on the same 260 training contracts and chunks; thresholds are the saved calibration-tuned `results/word_char_tfidf_nested/word_tfidf_thresholds.json`, unchanged. Before the data is read, the refit must reproduce the saved seed-42 calibration probabilities within 1e-5. Same metrics; the word+char minus word-only difference gets a paired interval (same draws). | "Word+char beats the original baseline on test" only if the paired macro-F1 interval is above 0; "no detectable difference" if it contains 0; "the baseline is better" if it is below 0. Development reference: +0.072 mean (5 of 5 seeds). |
| S2 | **Contract-level presence** ("does this contract contain category X?") | Per contract and category: gold = any positive chunk; score = highest chunk probability; detected = score >= the same threshold (that is, any chunk detected). Per-category P/R/F1/AUPR over contracts, macro-F1 (41) and macro-AUPR, with intervals. | Descriptive, next to the development dry-run value on the 82 seed-42 contracts (section 9). |
| S3 | **Triage ranking stability** under `v2-team-proposed` | As `scripts/evaluate_triage.py`: detected clauses (with evidence) and CUAD gold answers scored as if detected perfectly, both ranked by `aggregate_contracts`; Spearman, largest rank move and top-10 overlap for the triage score, high-count and worst-clause rules; plus a bootstrap interval for the triage-score Spearman. | Measures how far detection errors move the ranking, **not** whether the ranking matches a reviewer (no human reference exists). Development: Spearman 0.777 on 82 contracts. |
| S4 | **Evidence grounding** | From `clauses_from_probabilities`: share of detected clauses whose evidence could not be grounded (Unknown rate); share of grounded quotes overlapping a gold answer of the same category, overall and on correctly detected chunks. | Descriptive, next to the development dry-run values. |

Also recorded, without interpretation: chunk and contract counts, the share of gold spans inside a
single chunk, and CPU timings for chunking and prediction on this machine.

**Integrity checks inside the run:** every contract is also scored through
`ContractInferencePipeline` (the path the API serves), and its chunks, probabilities and detected
clauses must equal the evaluated ones. A mismatch stops the run before any metric is computed.

## 7. Not done, whatever the result

No threshold, weight, chunker or model change; no refit except S1 as defined; no exclusion of
categories or contracts; no alternative macro definition promoted to the headline; no
re-running with other bootstrap seeds or replicate counts; no subgroup search; no look at test
predictions or per-contract outputs before every output file is written; no second run. The
encoder is not evaluated (its checkpoints are not in the repository). Any analysis asked for after
the results are seen goes in a separate "Post hoc" subsection, labelled as such.

## 8. Preflight: structure only

Before the run, `--preflight` reads the test file's structure and writes
`results/final_test_evaluation/preflight.json`. It computes no prediction and prints no contract
text. It records the file's SHA-256, size and git blob id; top-level, entry, paragraph, question and
answer keys; the number of contracts, paragraphs per contract and questions per contract; whether
question ids carry the `_N` suffix (the training file does, one question per answer); the
category names derived from question ids; answer counts, empty answers, duplicate answers, and how
many answers are grounded (`context[start:start + len(text)] == text`); contexts containing CRLF;
whether `load_cuad` succeeds (exception type only); and how many contracts share a title or an exact
text with each seed-42 partition of the training file.

**Pass criteria:** 102 contracts; exactly one paragraph each; unique titles; no category name
outside the 41; every answer non-empty and grounded; no title or text shared with the 260
training or 66 calibration contracts; `load_cuad` succeeds; the git blob equals the pinned one.
Overlap with the 82 development-evaluation contracts does not fail the preflight (those contracts did
not shape the model) but is reported.

If the preflight fails, the run does not start. The team decides a fix (for example, a loader
change for a different id format), records it as a dated amendment in section 11 and commits it,
and runs the preflight again. No prediction exists at that point, so this does not compromise the
evaluation. If titles or texts are shared with the training file, the primary analysis still uses
all 102 contracts (the split is fixed), and the overlap is reported as a limitation.

## 9. Development dry run (evidence that the pipeline is right)

The sealed run differs from the dry run only in `--data`, `--output` and the opt-in flag. The dry
run evaluates the 82 seed-42 development-evaluation contracts, written as CUAD JSON in split order
by `--write-development-holdout` (git-ignored, under `outputs/`). Results are in
`results/final_test_dry_run/`.

_Filled in by the dry run; see below._

## 10. Guarding the run, failures and re-runs

**Guards in the script**
- `test.json` is read only with `--sealed-test-run`, only from `data/cuad/test.json`, and the
  results go only to `results/final_test_evaluation/`. `CUADv1.json` and zip archives are always
  refused. The flag with any other file is refused. All of this is checked before any file is opened.
- The sealed run refuses to start if tracked files have uncommitted changes, if this protocol is not
  committed, if no committed, passing `preflight.json` exists, if the data's SHA-256 differs from the
  preflight's, or if `report.json` exists on disk **or anywhere in git history**.
- The number of bootstrap replicates cannot be overridden in the sealed run.
- Nothing about a metric is printed until every output file is written. `report.json` is written
  last, and its existence marks a completed run.

**What counts as "the run happened":** the moment the script appends an attempt to
`run_attempts.json`. That is after the pre-data checks (pinned files, bundle, partitions, word-only
refit, calibration reproduction) and before the test file is read for evaluation. The preflight is
not the run.

**Failure rules**
1. A failure before an attempt is logged (pre-data checks, environment) is not a run. Fix it and
   start again; record what happened in the Results section.
2. A crash after an attempt is logged but before `report.json` exists counts as **before any
   metric was produced**, because the script shows nothing about the metrics until all files are
   written. It may be re-run once the cause is fixed. The fix may not touch any setting in section 2
   or any metric definition. Commit it, then run again with `--resume-after-crash "<what failed, what
   was fixed>"`. The reason is recorded in `report.json` as a deviation. If the traceback happened to
   display a metric or prediction value, that is also recorded as a deviation, and nothing is changed
   because of it.
3. Once `report.json` exists, the result stands. There is no second run, even if something fails
   afterwards (for example, printing the summary).
4. If a bug in the evaluation code is found after the result exists, the registered result is still
   reported. A corrected figure may be added next to it, labelled as a post hoc correction with the
   reason. It never silently replaces the registered one.
5. Any departure from this protocol is listed under "Deviations" in the Results section.

## 11. Phase 2 procedure

From the worktree root, on a clean tree at the final phase-1 commit, with
`models/word_char_tfidf.joblib` present and the tokenizer cached (Git Bash):

```sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m scripts.evaluate_final_test --data data/cuad/test.json --output results/final_test_evaluation --sealed-test-run --preflight
git add results/final_test_evaluation/preflight.json && git commit -m "test(eval): record the sealed test-set preflight"
python -m scripts.evaluate_final_test --data data/cuad/test.json --output results/final_test_evaluation --sealed-test-run
git add results/final_test_evaluation && git commit -m "feat(eval): add the sealed test-set results"
```

Then fill in section 12 from `summary.md` and `report.json` using the sentences in section 5.

**Amendments:** none.

## 12. Results

_Not run. Filled in once, after the sealed run._

## 13. Disclosures and known limitations, fixed before the run

- On 2026-11-17 a repository-wide text search for a word matched `test.json` and printed a match
  count for it. No content was displayed or used. The report repeats this disclosure.
- The released model was trained on 260 contracts, not the full 408 in the training file. A model
  refitted on more data might do better, but it would be a new, unevaluated model. This evaluation
  measures the model that is served.
- Risk rankings have no human reference. S3 measures detection error only.
- Thresholds were tuned on 66 contracts. Rare categories have 4-9 calibration positives, and their
  thresholds transfer poorly (`docs/error-analysis.md`).
- The bootstrap captures contract sampling only. It does not capture training randomness or the
  choice of split seed.
- The test set is a single sample of 102 contracts from the same CUAD release. It says nothing
  about contracts from other sources or about text extracted from PDFs.
