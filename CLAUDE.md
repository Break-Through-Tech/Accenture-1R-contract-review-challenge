# CLAUDE.md

Guidance for Claude Code in this repo. ARCHITECTURE.md is the source of truth for design; read it before adding any new module.

## Project

- **ClauseTriage**: Break Through Tech AI Studio × Accenture (Fall 2026). It detects the 41 CUAD clause categories, scores each clause Low/Medium/High risk, and rolls the scores up into a contract-level triage score.
- **Success:** per-category P/R/F1 clearly beats the TF-IDF baseline. Risk rankings get a strong Spearman correlation and bucket agreement against the advisor's hand-ranked clauses.
- **Deliverables:** a clause register for each contract, a clean documented repo, and a final report (limitations, estimated reviewer time saved).

## Hard Constraints

- **Input is CUAD JSON only** (`paragraphs[].context`, `qas[].answers[]`). No PDF parsing.
- **Fixed splits:** train on `train_separate_questions.json` (408 contracts) and evaluate on `test.json` (102). Never re-split. Take validation out of train, grouped by contract, with a fixed seed. Touch `test.json` only for final evaluation.
- **One text source, one chunker:** chunk the original `context` string unaltered so character offsets stay exact. If you normalize, keep an offset map back to the original text.

## Repo Layout

- `src/`: `metrics.py` (per-category P/R/F1, AUPR and the Trainer's macro AUPR hook), `thresholds.py` (per-class F1-optimal thresholds), `labels/` (`loader.py` CUAD loader, `categories.py` fixed category order, `projection.py` spans to chunk labels, `splits.py` grouped train/val split), `chunking/` (`chunk.py` holds the `Chunk` model, `chunker.py` holds `chunk_contract`), `models/naive/tfid_baseline.py` (TF-IDF + one-vs-rest logistic regression), `models/encoder/` (`config.py` `TrainConfig`, `dataset.py` chunk labeling, tokenizing and collator, `loss.py` `pos_weight` and `WeightedBCETrainer`, `train.py`, `predict.py`, `evaluate.py`), `evidence/` (Pydantic `Evidence` schema, on `feature/22-evidence-extraction`).
- `tests/`: pytest modules, one `test_<module>.py` per `src/` module. `pytest.ini` at the repo root sets `pythonpath = .`, so run `pytest` or `python -m pytest` from the root.
- `configs/`: YAML run configs (`encoder.yaml`). `scripts/`: development utilities run as `python -m scripts.<name>` (for example `smoke_train.py`).
- `notebooks/`: exploration and plots only. Pipeline logic belongs in `src/`, imported by the notebooks.
- `data/cuad/`: dataset JSON, `category_descriptions.csv`, `eda.py`, `EDA_SUMMARY.md`. `results/` holds metric CSVs. `outputs/` holds training outputs (checkpoints, MLflow files) and is gitignored.

## Code Style

- `snake_case` for functions, variables and modules; `PascalCase` for classes. No camelCase.
- Type-hint function signatures. Use a triple-quoted docstring (a one-line summary plus key behavior) for public functions and classes. Use a short `#` comment above each logical block.
- Keep modules small and pure. Use Pydantic models for anything that crosses a pipeline stage boundary (chunks, evidence, weights, scores).
- Document tests the same way as `src/`. Give every test function, fixture and helper a one-line docstring saying what it checks, and put a short `#` comment above each group of related tests. Name tests `test_<what is tested>_<expected behavior>`. Add hand-computed golden values for exact outputs and use `@pytest.mark.parametrize` for repeated cases.

## ML Conventions

- Treat this as multi-label classification over 41 categories. Strip the trailing `_N` suffix from CUAD question IDs to get the category name.
- Report per-category precision, recall, F1 and AUPR. Never use accuracy, because CUAD is heavily imbalanced.
- Tune thresholds per class on validation only. Store raw probabilities so retuning and rescoring never need a model rerun.

## Design Rules

- **Evidence must be grounded:** `text[start:end] == quote`. Evidence that fails validation goes to the Unknown queue and is never guessed.
- **Risk scoring is rule-based,** with weights in a versioned config owned by the advisor. Every score records its `weight_version_id`, and edits create a new version.
- **Respect scope tiers:** build core work (chunker, encoder, scoring, register export) before FastAPI/Gradio. React, Postgres and Alembic are deferred.

## Planned Stack (partly installed)

- Models (installed, in `requirements.txt`): DistilRoBERTa through the Hugging Face Trainer (BCE with `pos_weight`), trained on a Colab T4 and run on CPU for inference. MLflow on SQLite for tracking and the model registry is installed but not yet wired in.
- Storage and API (not yet in `requirements.txt`): SQLite with SQLAlchemy, then FastAPI with Gradio mounted inside it.
- Tooling (not yet set up): migrate to an installable `clausetriage/` package (`pyproject.toml`, `uv`), with pytest golden tests, ruff and GitHub Actions.

## Workflow

- Name branches `<type>/<issue#>-<short-name>` using conventional types (`feature`, `fix`, `docs`, `refactor`, `test`, `chore`) and merge to `main` by PR. Example: `fix/41-threshold-indexing`.
- Write commit messages in conventional-commit form: `<type>(<scope>): <summary>`, e.g. `fix(thresholds): index preds by category`.
- Install with `pip install -r requirements.txt`. Add every new import to `requirements.txt`.
- Don't commit `__pycache__/`, `.pyc` or `.DS_Store` files.

# Current Status of Project

Status reflects `main` (through PR #45) plus unmerged branches: `init-transformer-encoder-training` (encoder), `feature/sentence-chunking-integration` (includes `feature/12-chunking-evaluation`), `feature/pipeline-integration`, `experiment/token-aware-chunking` and `feature/22-evidence-extraction`.

**Done (on `main`)**
- EDA: 408 training contracts, 41 categories, about 50% of questions answered. Category imbalance is documented in `EDA_SUMMARY.md`.
- TF-IDF baseline with per-category metrics and per-class threshold tuning (`notebooks/baseline_test.ipynb`). The `apply_thresholds` indexing bug is fixed (PR #43) and `results/tfidf_baseline_metrics.csv` is regenerated.
- `Chunk` schema with span validation and a naive fixed-window chunker stub `chunk_contract(contract_id, text, window_chars, overlap_chars)` (`src/chunking/`, PR #44), with golden and invariant tests. `Chunk.start`/`end` are absolute, end-exclusive offsets into the original contract text, so `text[start:end] == chunk.text`.
- Label preparation (`src/labels/`, PR #45): CUAD loader with grounded answer spans, fixed alphabetical `CATEGORIES` order, any-overlap span-to-chunk projection (`LabeledChunk`) and a seeded train/val split grouped by contract, all with tests.

**Done (unmerged branches)**
- `init-transformer-encoder-training`: DistilRoBERTa training pipeline in `src/models/encoder/`, driven by `configs/encoder.yaml` (800/200 fixed windows, `pos_weight_cap` 50): `TrainConfig`, `ChunkDataset`/`ChunkCollator`, `compute_pos_weight`, `WeightedBCETrainer`, `fit`/`train` and a CLI (`python -m src.models.encoder.train`), with macro AUPR checkpoint selection (`compute_aupr`, `compute_transformer_metrics` in `src/metrics.py`). Inference and grading: `predict_proba` and checkpoint loaders (`predict.py`), and `evaluate.py` (per-class thresholds, per-category P/R/F1/AUPR, saved validation probabilities). `scripts/smoke_train.py` overfits a tiny subset on CPU (macro AUPR 1.0) and `notebooks/window_experiment.ipynb` records the window choice. 258 tests pass.
- `feature/sentence-chunking-integration`: chunking evaluation on the 326 development contracts. A sentence-aware 512/64 token chunker (`src/chunking/token_chunker.py`) keeps 98.8% of annotated clauses whole with no chunk over 512 tokens, versus 81.2% (0.6% over) for fixed 800/200, plus a `src/pipeline.py` that prepares labeled chunks.
- `feature/pipeline-integration`: an earlier fixed-window `src/pipeline.py`. `experiment/token-aware-chunking`: a token-window prototype, compared in the chunking evaluation.
- `feature/22-evidence-extraction`: `Evidence` Pydantic schema with span validation (`src/evidence/schema.py`). Its offsets are chunk-relative.

**In progress / known issues**
- The encoder is not yet trained on real data. The full run goes on a Colab T4 (the fp16 path is untested and local torch is CPU-only), then `evaluate.py`, MLflow logging, and the model registry on a separate branch.
- The encoder trains on the fixed-window stub. Retrain on the sentence-aware chunker once it is merged. The chunk settings are saved with each checkpoint (`train_config.json`) and inference must reuse them.
- Dependency conflict: the sentence branch pins `transformers==4.57.6` (plus `tokenizers` and `nupunkt`), while the encoder code was written and tested on transformers 5.19 (for example fractional `warmup_steps`). Settle one version before merging.
- `feature/pipeline-integration` and `feature/sentence-chunking-integration` both add `src/pipeline.py` (and `scripts/`), so they will conflict.
- `pos_weight_cap` 50 binds for 37 of 41 categories. Compare 50, 100 and square-root weights after the first full run.
- The baseline still uses a random 80/20 split on whole contracts. Re-run it on the grouped split and the same chunks before comparing it with the encoder.
- `evidence/extractor.py` is empty, so the heuristic sentence selection is still to be written. Converting `Evidence` offsets to contract coordinates (`chunk.start + evidence.start`) is not implemented.

**Next**
- Open the encoder PR, run the full Colab training and evaluation, and compare per-category results with the baseline on the same chunks.
- Merge the sentence-aware chunker, resolve the dependency pins and retrain.
- MLflow logging and model registry, then evidence extraction, rule-based scoring and register export, before FastAPI/Gradio.
- Settle the 4 risk signals with the advisor ASAP. The scoring schema is blocked until then.
