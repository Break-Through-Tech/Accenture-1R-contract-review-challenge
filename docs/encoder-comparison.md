# Encoder comparison runbook: DistilRoBERTa vs. TF-IDF

The project's first success criterion is that a fine-tuned transformer encoder clearly beats the
TF-IDF baseline on per-category precision, recall and F1. This page covers how to produce that
comparison, what comes out, and how to read it.

**Status: run on a Colab T4 on 2026-11-17. Verdict `does_not_beat`.** DistilRoBERTa scored lower
macro-F1 than word+char TF-IDF on all five seeds (by 0.063 to 0.082). The results are in
`results/encoder_nested_seed{42,7,17,23,101}/` and `results/encoder_vs_tfidf/`, and the
[final report](final-report.md) summarizes them. The rest of this page describes the procedure; the
last section records what the run verified.

## What is compared

| | DistilRoBERTa (candidate) | TF-IDF (baseline) |
|---|---|---|
| Runs | `results/encoder_nested_seed{42,7,17,23,101}/` (to be produced) | `results/word_char_tfidf_nested/` (seed 42), `results/word_char_tfidf_nested_seed{7,17,23,101}/` |
| Data | `data/cuad/train_separate_questions.json` only; `test.json` is never read | same |
| Split per seed | 260 training / 66 calibration / 82 evaluation contracts (seed 42; the other seeds have the same sizes) | same contracts |
| Chunks | shared sentence-aware chunker, 512 tokens, 64 overlap, pinned DistilRoBERTa tokenizer `fb53ab88...` | same chunks |
| Model fitting | 4 epochs, lr 3e-5, effective batch 32, fp16, BCE with `pos_weight` capped at 50 (`configs/encoder.yaml`) | logistic regression on word (and word+char) TF-IDF |
| Checkpoint choice | best epoch by macro-AUPR on the **calibration** contracts | none needed |
| Thresholds | per category, F1-optimal on calibration contracts only | same |
| Metrics | evaluation contracts only | same |

There are two baselines. **Word+char TF-IDF** (`word_char_tfidf`) is the current, stronger control
and the one the verdict is judged against. **Word-only TF-IDF** (`word_tfidf`) is the original
baseline from the project brief. It is reported too, but beating only the weaker one does not
count.

Before any training, each seed's split is checked against the saved TF-IDF `report.json`:
training-file SHA-256, `test_data_used: false`, seed and fractions, contract counts, and the
SHA-256 of the ordered contract IDs in every partition. After chunking, the chunk counts and the
calibration and evaluation label matrices must also equal the saved `word_char_tfidf_*.npz` row
for row. Any difference stops the run (`SplitMismatchError`). A local `--dry-run` on 2026-11-17
passed this check for all five seeds (chunk counts 9,025/1,899/2,196 for seed 42; 8,666/2,001/2,453
for 7; 8,817/1,787/2,516 for 17; 8,058/2,641/2,421 for 23; 7,816/2,311/2,993 for 101).

## Running it on Colab (recommended)

Open `notebooks/encoder_colab.ipynb` in Colab with a T4 runtime and run all cells. It:

1. checks for a GPU,
2. mounts Google Drive (optional; on by default so a disconnect loses nothing),
3. clones the private repository with a GitHub token taken from the `GITHUB_TOKEN` Colab secret or
   a hidden prompt (the token is sent as a header and never stored),
4. installs `requirements.txt` with a constraint that keeps Colab's preinstalled CUDA torch,
5. downloads the pinned model files and runs the runner/comparison unit tests,
6. trains and evaluates the five seeds, then runs the comparison,
7. zips the result folders (no checkpoints) for download, with commit instructions.

**If Colab disconnects,** reconnect and run all cells again. Finished seeds are skipped. A seed cut
off mid-training resumes from its last saved epoch (checkpoints are written to Drive), and a seed
cut off after training is only re-scored.

**Runtime (estimate, not a measurement).** The five seeds have 42,382 training chunks in total, so
four epochs are about 170,000 training sequences of up to 512 tokens. At an assumed 40 to 80
sequences a second for DistilRoBERTa in fp16 on a T4, training takes about 35 to 70 minutes.
Scoring is about 76,000 forward passes (calibration chunks once per epoch and twice at the end,
evaluation chunks once), roughly 5 to 10 minutes. With chunking, installation and cloning, plan
for about 1 to 1.5 hours. The runner prints each seed's real training time, so the first seed
tells you how long the rest will take.

## Running it anywhere else

From the repository root, on a CUDA machine:

```sh
python -m scripts.evaluate_encoder_nested --dry-run   # verify all five splits, no training (CPU, a few minutes)
python -m scripts.evaluate_encoder_nested             # train + evaluate seeds 42 7 17 23 101
python -m scripts.compare_encoder_tfidf               # compare with both TF-IDF baselines
```

Runner options: `--seeds` (any subset of the five), `--results-root` (default `results/`),
`--work-root` (checkpoints; default `outputs/encoder_nested/`, which git ignores), `--force`
(re-run finished seeds), `--keep-checkpoints` (keep per-epoch checkpoints; `best/` is always kept),
`--batch-size` (scoring), `--register` (log each checkpoint trained in this invocation to the local
MLflow registry).

**Hugging Face cache and offline mode.** The encoder path (`evaluate_encoder_nested`, `train`,
`smoke_train`) no longer hard-codes `.venv/hf-cache`. Settings come from
`clausetriage/models/encoder/hub.py`, in this order: command-line flag, environment variable, default.

| Setting | Flag | Environment variable | Default |
|---|---|---|---|
| Cache folder | `--hf-cache-dir PATH` | `CLAUSETRIAGE_HF_CACHE` | Hugging Face default (honours `HF_HOME`) |
| Offline | `--offline` / `--no-offline` | `CLAUSETRIAGE_HF_OFFLINE`, else `HF_HUB_OFFLINE` | online |

The model revision is not configurable here: the runner refuses to start unless
`configs/encoder.yaml` names the same model and revision as the shared chunker. To reproduce the
old local behaviour, use `--hf-cache-dir .venv/hf-cache --offline`. That cache only holds tokenizer
files, so training also needs the weights in the default cache. The other experiment scripts still
hard-code `.venv/hf-cache`.

**CPU sanity check.** `python -m scripts.smoke_train` overfits 64 training chunks for a few minutes
on a laptop CPU and passes when macro-AUPR on those chunks reaches 0.9. It only shows that the
pipeline is wired correctly and says nothing about model quality.

## What comes out

One folder per seed, `results/encoder_nested_seed<seed>/`, in the same layout as the reranker and
long-context runs:

| File | Content |
|---|---|
| `report.json` | `status: complete`, split provenance (counts, contract-ID digests, chunk counts), data hash, chunking, training config and its SHA-256, macro metrics, library/GPU/commit info, `test_data_used: false` |
| `calibration_predictions.npz`, `evaluation_predictions.npz` | `y_true` and raw `y_prob` (sigmoid) for every chunk, so thresholds can be re-tuned without retraining |
| `calibration_chunks.csv`, `evaluation_chunks.csv` | `contract_id, chunk_index, start, end` for each row of the matching `.npz`, so error examples can be read from the contract text |
| `thresholds.json` | per-category thresholds fitted on calibration |
| `per_category_metrics.csv` | evaluation precision, recall, F1, support, AUPR, threshold and `low_support` per category |
| `training_history.json` | per-epoch loss and calibration macro-AUPR from the Trainer |

Checkpoints stay in `<work-root>/seed<seed>/train/best/` and are never committed.

The comparison writes `results/encoder_vs_tfidf/vs_word_char_tfidf/` and `.../vs_word_tfidf/`:

| File | Content |
|---|---|
| `summary.md` | verdict, per-seed table with bootstrap intervals, support-bucket table |
| `comparison.json` | everything above plus the protocol, aggregate means and between-seed standard deviations |
| `per_seed.csv` | macro-F1 and macro-AUPR for both models, the deltas, and their 95% intervals |
| `per_category_by_seed.csv` | per category and seed: support, P/R/F1/AUPR for both models, deltas |
| `per_category_summary.csv` | per category, averaged over seeds: mean support, bucket, mean/min/max F1 delta, AUPR delta, wins, rank (`status: insufficient support` and no rank for rare categories) |
| `support_bucket_summary.csv` | mean deltas and improved/worse counts for rare, uncommon and common categories |

The comparison reads only saved files. Thresholds are each run's own and are never re-tuned. It
works on any run in the same layout. For example, the retrieval reranker:

```sh
python -m scripts.compare_encoder_tfidf --candidate-template "results/retrieval_reranker_seed{seed}" \
  --candidate-name reranker --score-key reranker_score --threshold-key reranker --output outputs/reranker_vs_tfidf
```

Runs without `evaluation_chunks.csv` (such as the reranker) are re-chunked with the pinned
tokenizer to find contract boundaries. The rebuilt labels must match the saved ones.

## How to read the comparison

- **Macro-F1** uses the project's fixed 41-category convention: a category with no evaluation
  positives contributes F1 = 0 to both models. **Macro-AUPR** averages over categories that have
  at least one evaluation positive. Both match `results/retrieval_reranker_multiseed/`.
- **Paired contract bootstrap.** For each seed, the 82 evaluation contracts are resampled with
  replacement 1,000 times. Both models are scored on the same resample and the difference is
  recorded. Thresholds stay fixed. This is the helper in `scripts/analyze_reranker_multiseed.py`
  (`numpy default_rng(20261105 + seed)`), so intervals are comparable with the reranker analysis.
  It captures evaluation-sample uncertainty, not training randomness.
- **Seeds are not independent.** The five evaluation sets overlap. Five wins out of five is
  consistency across splits, not five independent confirmations.
- **Per-category deltas** are averaged over seeds and grouped by mean evaluation support, using
  the buckets in `docs/error-analysis.md`: common (30+), uncommon (10-29) and rare (<10). Rare
  categories swing by 0.3-0.4 F1 between seeds, so they are listed as "insufficient support" and
  not ranked. Their deltas are still shown, but do not draw per-category conclusions from them.
- **Checkpoint selection uses calibration contracts.** The encoder picks its best epoch on the
  same contracts its thresholds are fitted on. The evaluation contracts are still untouched, so
  the comparison is fair, but calibration-set numbers in `training_history.json` are optimistic.

## What counts as "clearly beating the baseline"

These rules were fixed before any encoder result existed and are coded in
`scripts/compare_encoder_tfidf.py::verdict`. They are judged against word+char TF-IDF.

**`clearly_beats`** requires all of:

1. all five predeclared seeds are present;
2. macro-F1 is higher on **5 of 5** seeds;
3. the paired-bootstrap 95% interval for the macro-F1 difference is above zero on at least
   **4 of 5** seeds;
4. the mean macro-AUPR difference is positive, so the F1 gain does not come from worse ranking.

**`does_not_beat`:** the mean macro-F1 difference is zero or negative, or the encoder wins
on 2 or fewer seeds.

**`not_clearly`:** anything in between. The encoder may be better on average, but not by enough
to claim the success criterion.

**`incomplete`:** fewer than five seeds were run. No claim either way.

For scale: per-seed bootstrap intervals for the reranker's macro-F1 difference were about ±0.035
wide, so condition 3 roughly needs a gain of 0.035-0.04 macro-F1 on most seeds. Word+char over
word-only TF-IDF (+0.072 mean, 5/5 seeds) is the size of gain that would clearly pass. The
reranker (+0.016 mean, 4/5 seeds, no interval above zero) is the kind of result that does not.

**What does not count, whatever the verdict:**

- beating only word-only TF-IDF;
- a gain that rests on rare categories, or a single standout category;
- calibration-set (`training_history.json`) or `trainer.evaluate()` numbers;
- a single seed, or any subset of seeds chosen after seeing results;
- numbers from runs whose split check was bypassed or whose thresholds were re-tuned on
  evaluation contracts.

Even with `clearly_beats`, check `support_bucket_summary.csv` and `per_category_summary.csv` for
common or uncommon categories that got worse. Report them as limitations next to the headline
result.

## What is verified and what is not

**Verified by the Colab run on 2026-11-17** (from the saved `results/encoder_nested_seed*/report.json`):
all five seeds trained for four epochs with fp16 on a Tesla T4 (torch `2.14.1+cu130`, commit
`0bf69ce`), passed the split check, and were scored on GPU; each `report.json` has
`status: complete`. Training took 531 to 586 seconds per seed, about 47 minutes for all five,
inside the 35 to 70 minute estimate above. The comparison was produced and the result folders were
downloaded and committed (`e2708ec`).

**Not verified:** resuming from a real interrupted Trainer checkpoint (the saved reports do not
show whether any seed was resumed), and which of the notebook's optional paths ran (Drive mount,
cloning with the `GITHUB_TOKEN` secret or the prompt); the saved results do not record them.

Before the run, these were checked locally on CPU: the smoke test (`SMOKE PASS`, macro-AUPR 0.980 on the overfit
subset), the five-seed `--dry-run` split check, and the unit tests for the runner (seed handling,
resume, split verification), the cache settings and the comparison on synthetic data. The
comparison was also run on the saved retrieval-reranker runs, using the command above, as an
end-to-end check of the script. That gives verdict `not_clearly` against word+char TF-IDF (4/5 F1
wins, no seed with an interval above zero). Its numbers reproduce
`results/retrieval_reranker_multiseed/` exactly for seeds 42, 7 and 23, including the seed-42
interval. Seeds 17 and 101 differ slightly on the baseline side, because this script uses the
saved TF-IDF thresholds while that analysis re-fitted the retriever's thresholds.
