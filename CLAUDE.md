# CLAUDE.md

Guidance for Claude Code in this repo. ARCHITECTURE.md is the source of truth for design; read it before adding any new module.

## Project

- **ClauseTriage**: Break Through Tech AI Studio × Accenture (Fall 2026). It detects the 41 CUAD clause categories, scores each clause Low/Medium/High risk, and rolls the scores up into a contract-level triage score.
- **Success:** per-category P/R/F1 clearly beats the TF-IDF baseline. Risk rankings get a strong Spearman correlation and bucket agreement against the advisor's hand-ranked clauses.
- **Deliverables:** a clause register for each contract, a clean documented repo, and a final report (limitations, estimated reviewer time saved).

## Hard Constraints

- **Training and evaluation data is CUAD JSON** (`paragraphs[].context`, `qas[].answers[]`). Models are trained and scored on this text only.
- **Uploaded contracts are in scope.** The advisor requires PDF upload (November 2026), which supersedes the original brief's "do not parse raw PDFs". PDF, Word and pasted text are accepted at inference time.
- **Fixed splits:** train on `train_separate_questions.json` (408 contracts) and evaluate on `test.json` (102). Never re-split. Take validation out of train, grouped by contract, with a fixed seed. Touch `test.json` only for final evaluation.
- **One text source, one chunker:** chunk the original `context` string unaltered so character offsets stay exact. If you normalize, keep an offset map back to the original text. For an uploaded file, the one text source is the text extracted from it: offsets, evidence and page numbers all refer to that string.

## Repo Layout

- `src/` pipeline stages:
  - `labels/`: CUAD loader, the 41 `CATEGORIES`, span-to-chunk label projection, grouped contract split.
  - `chunking/`: `chunk.py` holds the `Chunk` model. `token_chunker.py` holds `chunk_sentence_windows` (512 tokens, 64 overlap, pinned DistilRoBERTa tokenizer), the chunker every stage uses. `chunker.py` is the old fixed-window stub.
  - `pipeline.py`: contracts to labeled chunks for training. `inference.py`: `ContractInferencePipeline` (chunk, classify, evidence, risk score, triage score).
  - `models/naive/`: TF-IDF baselines (`word_tfidf.py` is the word+char control) and the retrieval reranker. `models/encoder/`: DistilRoBERTa training, prediction and the alternative architectures.
  - `registry/`: `ClauseModelBundle`, the MLflow registry and the release-hosted demo model. `store/`: the SQLite artifact store.
  - `evidence/`: `Evidence` schema, category cues, grounded extraction with an Unknown queue.
  - `risk/`: weight sets (`weights/*.json`), the four signals, clause scoring, contract triage and sensitivity.
  - `ui/`: the Gradio demo (`python -m src.ui.app`) and the PDF, Word and text readers.
  - `metrics.py`, `thresholds.py`, `threshold_strategies.py`, `error_analysis.py`: evaluation helpers.
- `scripts/`: runnable experiments and evaluations (`python -m scripts.<name>` from the repo root). `configs/`: one YAML per model.
- `tests/`: pytest modules, one `test_<module>.py` per module. `pytest.ini` at the repo root sets `pythonpath = .`, so run `python -m pytest` from the root.
- `docs/`: one guide per experiment, indexed in `docs/README.md`. `results/`: saved metrics, thresholds and raw probabilities for each experiment.
- `notebooks/`: exploration and plots only. Pipeline logic belongs in `src/`, imported by the notebooks.
- `data/cuad/`: dataset JSON, `category_descriptions.csv`, `eda.py`, `EDA_SUMMARY.md`. `data/demo_contracts/`: sample PDFs for the demo.

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

## Stack

- Installed (pinned in `requirements.txt`): scikit-learn, `transformers` 4.57.6 with PyTorch, MLflow on SQLite, SQLAlchemy, Pydantic, Gradio, `pypdf` and `python-docx`.
- Models: word+char TF-IDF with one-vs-rest logistic regression is the supported control. DistilRoBERTa goes through the Hugging Face Trainer (BCE with `pos_weight`), trained on a Colab T4 and run on CPU for inference.
- Not yet in the repo: FastAPI (Gradio will be mounted inside it), `pyproject.toml`, ruff, GitHub Actions, Docker, and the installable `clausetriage/` package rename.

## Workflow

- Name branches `<type>/<issue#>-<short-name>` using conventional types (`feature`, `fix`, `docs`, `refactor`, `test`, `chore`) and merge to `main` by PR. Example: `fix/41-threshold-indexing`.
- Write commit messages in conventional-commit form: `<type>(<scope>): <summary>`, e.g. `fix(thresholds): index preds by category`.
- Install with `pip install -r requirements.txt` into a Python 3.12 venv, and reinstall after pulling. Add every new import to `requirements.txt`. A venv that has drifted from the pins fails tests for unrelated reasons (`transformers` 5.x breaks `tests/test_train.py`).
- Run tests with `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m pytest`. The README setup step caches the pinned tokenizer in `.venv/hf-cache`.
- Don't commit `__pycache__/`, `.pyc`, `.DS_Store`, model weights, SQLite databases or anything under `outputs/` or `models/`.

# Current Status of Project

Status reflects `main` at `7428dd3` (2026-11-17). The docs index in `docs/README.md` links every experiment report.

**Done**
- Data and labels: CUAD loader, any-overlap label projection, grouped contract splits. Experiments use train/calibration/evaluation contract partitions of the training file; `test.json` has never been read.
- Chunking: the sentence-window chunker contains 98.8% of annotated spans on the development contracts (`docs/chunking-experiments.md`).
- Baseline: word+char TF-IDF, mean macro-F1 0.505 and macro-AUPR 0.563 over five seeded splits (0.533 / 0.597 on seed 42). Released as the bundle `model-word-char-tfidf-v1`.
- Alternatives screened against it: retrieval reranker, contract-wide Transformer, category-query, graph, contrastive, set prediction and sparse long context. None robustly beats TF-IDF; the reranker's +0.016 mean macro-F1 has bootstrap intervals that include zero.
- Error analysis and threshold-strategy study (`docs/error-analysis.md`, `docs/threshold-robustness.md`).
- Model registry: `ClauseModelBundle` plus MLflow on SQLite.
- Evidence extraction with span validation and an Unknown queue.
- Risk layer: four signals (`category_severity`, `one_sidedness`, `magnitude`, `contract_context`), versioned weight sets, clause scoring, contract triage score and sensitivity analysis (`scripts/evaluate_triage.py`).
- End-to-end inference and the Gradio demo: upload PDF, Word or text, rank contracts, show risky clauses with page numbers and quotes, download a CSV clause register.

**Gaps against ARCHITECTURE.md**
- The plain DistilRoBERTa fine-tune has never been run at full scale (no GPU time yet), so the "encoder beats baseline" criterion is unmeasured.
- There are no advisor hand-ranked clauses, so Spearman and bucket agreement against the advisor do not exist. The 0.81 Spearman in `results/contract_triage/` is against a perfect-detection reference built from CUAD annotations.
- The `v1-placeholder` risk weights name a severity for only 9 of 41 categories and are not advisor-reviewed.
- The artifact store is not called by the pipeline and lacks the `feedback` and `advisor_rankings` tables.
- The clause register exists only as a CSV built inside the UI.
- No FastAPI layer, editable weights panel, feedback capture or review queue.
- No `pyproject.toml`, ruff, CI, Docker or license.
- The final `test.json` evaluation has not been run, and there is no final report.
- Evidence heuristic is weak: about 55% of grounded quotes overlap a gold answer, and a false-positive prediction still gets a plausible quote.
- Rare categories (under 10 evaluation positives) swing 0.3 to 0.4 F1 between seeds; report them with intervals or as insufficient support.

**In progress (branches opened 2026-11-17)**
- `feature/25-risk-weights-v2`: team-proposed weights covering all 41 categories.
- `feature/21-document-ingestion`: PDF, Word and text readers as a core stage.
- `feature/29-clause-register-export`: core register with CSV and JSON.
- `feature/32-artifact-store-integration`: missing tables, persistence, rescoring from cached probabilities.
- `feature/18-advisor-agreement-eval`: advisor ranking packet and the agreement evaluation.
- `feature/17-encoder-colab-run`: five-seed encoder runner and Colab notebook.
- `chore/35-tooling-ci`: `pyproject.toml`, ruff and GitHub Actions.

**Next**
- Serving: FastAPI backend with an upload endpoint, feedback and review queue, editable weights in the UI.
- Run the advisor-agreement evaluation when rankings arrive, then freeze the model and make one sealed `test.json` evaluation.
- Final report, license, Docker, and the `src` to `clausetriage` package rename.
