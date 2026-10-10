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

- `src/`: `metrics.py` (per-category P/R/F1), `thresholds.py` (per-class F1-optimal thresholds), `models/naive/tfid_baseline.py` (TF-IDF + one-vs-rest logistic regression), `evidence/` (Pydantic `Evidence` schema).
- `notebooks/`: exploration and plots only. Pipeline logic belongs in `src/`, imported by the notebooks.
- `data/cuad/`: dataset JSON, `category_descriptions.csv`, `eda.py`, `EDA_SUMMARY.md`. `results/` holds metric CSVs.

## Code Style

- `snake_case` for functions, variables and modules; `PascalCase` for classes. No camelCase.
- Type-hint function signatures. Use a triple-quoted docstring (a one-line summary plus key behavior) for public functions and classes. Use a short `#` comment above each logical block.
- Keep modules small and pure. Use Pydantic models for anything that crosses a pipeline stage boundary (chunks, evidence, weights, scores).

## ML Conventions

- Treat this as multi-label classification over 41 categories. Strip the trailing `_N` suffix from CUAD question IDs to get the category name.
- Report per-category precision, recall, F1 and AUPR. Never use accuracy, because CUAD is heavily imbalanced.
- Tune thresholds per class on validation only. Store raw probabilities so retuning and rescoring never need a model rerun.

## Design Rules

- **Evidence must be grounded:** `text[start:end] == quote`. Evidence that fails validation goes to the Unknown queue and is never guessed.
- **Risk scoring is rule-based,** with weights in a versioned config owned by the advisor. Every score records its `weight_version_id`, and edits create a new version.
- **Respect scope tiers:** build core work (chunker, encoder, scoring, register export) before FastAPI/Gradio. React, Postgres and Alembic are deferred.

## Planned Stack (not yet installed)

- Models: DistilRoBERTa through the Hugging Face Trainer (BCE with `pos_weight`), trained on a Colab T4 and run on CPU for inference. MLflow on SQLite for tracking and the model registry.
- Storage and API: SQLite with SQLAlchemy, then FastAPI with Gradio mounted inside it.
- Tooling: migrate to an installable `clausetriage/` package (`pyproject.toml`, `uv`), with pytest golden tests, ruff and GitHub Actions.

## Workflow

- Name branches `<type>/<issue#>-<short-name>` using conventional types (`feature`, `fix`, `docs`, `refactor`, `test`, `chore`) and merge to `main` by PR. Example: `fix/41-threshold-indexing`.
- Write commit messages in conventional-commit form: `<type>(<scope>): <summary>`, e.g. `fix(thresholds): index preds by category`.
- Install with `pip install -r requirements.txt`. Add every new import to `requirements.txt`.
- Don't commit `__pycache__/`, `.pyc` or `.DS_Store` files.

# Current Status of Project

Status reflects `main` plus the unmerged `feature/22-evidence-extraction` branch.

**Done**
- EDA: 408 training contracts, 41 categories, about 50% of questions answered. Category imbalance is documented in `EDA_SUMMARY.md`.
- TF-IDF baseline with per-category metrics and per-class threshold tuning (`notebooks/baseline_test.ipynb`).
- `Evidence` Pydantic schema with span validation (`src/evidence/schema.py`, on `feature/22-evidence-extraction`, PR not yet merged).

**In progress / known issues**
- `evidence/extractor.py` is empty, so the heuristic sentence selection is still to be written.
- `apply_thresholds` indexes `preds[:, 1]` instead of `preds[:, i]`, so the baseline's tuned metrics are invalid until this is fixed. `results/tfidf_baseline_metrics.csv` is stale.
- The baseline uses a random 80/20 split with whole contracts as chunks. It needs a grouped-by-contract split and the real chunker.
- `pandas` is imported but missing from `requirements.txt` on `main`.

**Next**
- Chunker module with golden offset tests, then label projection (spans to multi-hot labels for each chunk).
- DistilRoBERTa fine-tuning script logged to MLflow.
- Settle the 4 risk signals with the advisor before November. The scoring schema is blocked until then.
