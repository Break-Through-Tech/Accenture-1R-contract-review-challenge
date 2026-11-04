# Chunking and encoder branch consolidation

This note compares the two outstanding branches with current `main` at `3b5e3fb` and records what is safe to carry forward. The branch tips are not merge candidates as-is: both predate substantial changes now on `main`.

## PR #48: token-window prototype

PR #48 adds `experiments/token_chunking.py` and tests. Its implementation is not the current production-facing chunker: it maps token offsets directly to character slices, uses the tokenizer's default revision in its test, and does not explicitly validate ordered chunk indices or strict advancement of both starts and ends. It treats `window_tokens` as content tokens and checks the 512-token model limit only in tests.

The algorithm was preserved unchanged for the comparison in PR #49 through `src/chunking/gwen_adapter.py`, pinned to commit `94d6d3d8b332de91338625bde100259a1717728b` and checked against the shared `Chunk` implementation. The closed PR's exact source is now archived at `experiments/archive/pr48_token_chunking.py` and verified by SHA-256, so reproducibility no longer depends on keeping the stale remote branch alive. The current `src/chunking/token_chunker.py` is the compatible implementation for further work: it uses a pinned fast tokenizer, accounts for special tokens, retokenizes exact original-text slices, guarantees source coverage and advancing offsets, and handles whitespace/Unicode offsets. Real-tokenizer tests exercise these invariants. `src/chunking/evaluation.py` and `src/chunking/experiment.py` provide annotation containment, token-limit, duplication, overlap, fallback, and per-category measurements without using annotations to choose boundaries.

The saved 326-contract development comparison found 98.82% annotation containment and 6.29% duplicate-character overhead for sentence windows, versus 97.10% and 32.64% for PR #48's 480/120 setting. PR #48 was faster in the reported single-machine run (8.25s versus 37.93s). The settings differ in budget/overlap semantics, and this is a chunk/annotation comparison, not a classifier-quality result. Earlier exploration included all 408 training-file contracts, so the saved validation subset is not pristine. Do not tune on it or present it as held-out classifier evaluation.

**Disposition:** PR #48 was closed as superseded, not merged. Do not merge its prototype verbatim. Its behavior has been evaluated and its comparator is reproducibly pinned in-repository; its useful contribution is the comparison, not a second competing production chunker. Any new chunker work should extend the shared implementation and tests, and should only claim downstream model effects after an uncontaminated contract-level evaluation is available.

## `init-transformer-encoder-training`: encoder implementation

The branch has 14 commits ahead of current `main`, but most of the apparent tree difference is because it is an older repository snapshot: its tree lacks the later architecture suite, registry, risk schema, and chunking evaluation. The meaningful encoder changes are already represented and evolved on current `main`:

- frozen/validated `TrainConfig`, pinned model revision, sentence-token mode, shared `prepare_train_val` path, and explicit `max_length`/overlap settings;
- weighted BCE training, train-only positive weights, macro-AUPR checkpoint selection, and a smoke-training path;
- validation metrics, checkpoint/config saving, and inference helpers;
- contract-level splits and tests that keep training and validation contracts disjoint.

Do not merge the old branch wholesale. In particular, its removal of `model_revision` and sentence-token mode would weaken reproducibility and compatibility; its change from `warmup_ratio` to floating-point `warmup_steps` is not a safe equivalent. `requirements.txt`, `.gitignore`, and `CLAUDE.md` also conflict materially with current main. Its unique window-sweep notebook is now summarized, with sampling and tokenizer caveats, in `docs/chunking-experiments.md`; later pinned chunking comparisons supersede it as the reproducible reference.

**Disposition:** treat current `main` as the maintained encoder implementation. The old branch's remaining unique findings have been transferred as historical context; no branch-only implementation should be merged. The existing encoder results are research results; no new performance claim follows from the old branch's commits. Further classifier comparisons must use contract-level train/calibration/evaluation splits, calibrate thresholds on calibration only, and must not read `data/cuad/test.json`.

## Current-main compatibility check

The full suite on current main passed before this hardening work: **560 tests passed**. This includes real-tokenizer chunk tests and the sentence-token training path. This work branch adds inference input validation so invalid sequence lengths and nonpositive batch sizes fail before tokenization, including for empty input; makes `fit` explicitly load the same pinned fast tokenizer used by sentence-token chunking; and archives PR #48's byte-identical source so experiment reproduction no longer depends on a remote branch. The test file does not read or modify `data/cuad/test.json`.
