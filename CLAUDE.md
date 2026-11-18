# CLAUDE.md

Guidance for Claude Code in this repo. ARCHITECTURE.md is the source of truth for design; read it before adding any new module.

## Project

- **ClauseTriage**: Break Through Tech AI Studio × Accenture (Fall 2026). It detects the 41 CUAD clause categories, scores each clause Low/Medium/High risk, and rolls the scores up into a contract-level triage score.
- **Success:** per-category P/R/F1 clearly beats the TF-IDF baseline. For risk rankings the brief asks for Spearman and bucket agreement against hand-ranked clauses; the team decided (2026-11-17) not to collect hand rankings, so the risk layer is judged by its sensitivity analysis and its ranking stability against a perfect-detection reference, and the missing human validation is reported as a limitation.
- **Decisions are the team's.** The challenge advisor has limited input on the project. Do not gate work on advisor review, sign-off or rankings, and do not describe weights or results as pending advisor review.
- **Deliverables:** a clause register for each contract, a clean documented repo, and a final report (limitations, estimated reviewer time saved).

## Hard Constraints

- **Training and evaluation data is CUAD JSON** (`paragraphs[].context`, `qas[].answers[]`). Models are trained and scored on this text only.
- **Uploaded contracts are in scope.** The advisor requires PDF upload (November 2026), which supersedes the original brief's "do not parse raw PDFs". PDF, Word and pasted text are accepted at inference time.
- **Fixed splits:** train on `train_separate_questions.json` (408 contracts) and evaluate on `test.json` (102). Never re-split. Take validation out of train, grouped by contract, with a fixed seed. Touch `test.json` only for final evaluation.
- **`data/cuad/CUADv1.json` and `data/cuad/cuad-v1.zip` contain the test contracts.** Treat them like `test.json`: never load them for training, validation, manifests or demos.
- **One text source, one chunker:** chunk the original `context` string unaltered so character offsets stay exact. If you normalize, keep an offset map back to the original text. For an uploaded file, the one text source is the text extracted from it: offsets, evidence and page numbers all refer to that string.

## Repo Layout

- `src/` pipeline stages:
  - `labels/`: CUAD loader, the 41 `CATEGORIES`, span-to-chunk label projection, grouped contract split.
  - `chunking/`: `chunk.py` holds the `Chunk` model. `token_chunker.py` holds `chunk_sentence_windows` (512 tokens, 64 overlap, pinned DistilRoBERTa tokenizer), the chunker every stage uses. `chunker.py` is the old fixed-window stub.
  - `pipeline.py`: contracts to labeled chunks for training. `inference.py`: `ContractInferencePipeline` (chunk, classify, evidence, risk score, triage score).
  - `models/naive/`: TF-IDF baselines (`word_tfidf.py` is the word+char control) and the retrieval reranker. `models/encoder/`: DistilRoBERTa training, prediction and the alternative architectures.
  - `ingest/`: `Document` model and the hardened PDF, Word and text readers for uploaded contracts.
  - `registry/`: `ClauseModelBundle`, the MLflow registry and the release-hosted demo model.
  - `store/`: the SQLite artifact store (`save_inference_result`, Unknowns and feedback tables) and `rescore_contracts`, which rescores from cached probabilities.
  - `evidence/`: `Evidence` schema, category cues, grounded extraction with an Unknown queue.
  - `risk/`: weight sets (`weights/*.json`, default `v2-team-proposed`), the four signals, clause scoring, contract triage and sensitivity.
  - `register/`: `ContractRegister` schema and the CSV and JSON clause-register export.
  - `api/`: the FastAPI serving layer (`python -m src.api`, localhost only). `service.py` holds `ClauseTriageService`, the plain-Python layer the routers and the reviewer page both call. `jobs.py` is the scoring-job status table and `routers/` holds thin handlers; add a router through `create_app(routers=[...])`.
  - `review/`: the review queue (Unknowns and reviewer conflicts), verdict resolution and the leakage-safe training export.
  - `ui/`: `reviewer.py` is the store-backed reviewer page mounted at `/ui/` (`reviewer_views.py` holds its pure helpers). `app.py` stays the in-memory quick demo (`python -m src.ui.app`) for the demo-only install.
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
- **Risk scoring is rule-based,** with weights in a versioned config owned by the team. Every score records its `weight_version_id`, and edits create a new version.
- **Clients go through the service.** The reviewer page and the routers call `ClauseTriageService` (or HTTP) and never `ArtifactStore` directly. The API has no authentication, so it binds to 127.0.0.1 only.
- **Feedback verdicts:** `accept` and `reject` need a model detection (`reject` also answers a reviewer's claimed miss); `missing` needs none. History is append-only. Reviewers who disagree are a conflict, which is reported and excluded from export, never settled by majority.
- **Feedback export is allowlist-only:** a contract's feedback is exported only if its exact-text SHA-256 is in a `TrainingManifest`'s training set (default: the training partition of `train_separate_questions.json`, seed 42, fraction 0.2). Uploads and held-out contracts are excluded.
- **Never hide every child of a `gr.Row` and show several at once.** In Gradio 6.17.3 that hangs the browser. Keep the components rendered and toggle `interactive` instead.
- **Respect scope tiers:** build core work (chunker, encoder, scoring, register export) before FastAPI/Gradio. React, Postgres and Alembic are deferred.

## Stack

- Installed (pinned in the layered `requirements*.txt` files, see `docs/development.md`): scikit-learn, `transformers` 4.57.6 with PyTorch, MLflow on SQLite, SQLAlchemy, Pydantic, Gradio, `pypdf` and `python-docx`.
- Models: word+char TF-IDF with one-vs-rest logistic regression is the supported control. DistilRoBERTa goes through the Hugging Face Trainer (BCE with `pos_weight`), trained on a Colab T4 and run on CPU for inference.
- Tooling: `pyproject.toml` holds project metadata and the ruff config (check only, no reformatting). GitHub Actions runs ruff and pytest on every PR.
- Serving: FastAPI with the Gradio reviewer page mounted inside it, with job status and all pipeline outputs in one SQLite file.
- Not yet in the repo: Docker and the installable `clausetriage/` package rename.

## Workflow

- Name branches `<type>/<issue#>-<short-name>` using conventional types (`feature`, `fix`, `docs`, `refactor`, `test`, `chore`) and merge to `main` by PR. Example: `fix/41-threshold-indexing`.
- Write commit messages in conventional-commit form: `<type>(<scope>): <summary>`, e.g. `fix(thresholds): index preds by category`.
- Install with `pip install -r requirements-dev.txt` into a Python 3.12 venv, and reinstall after pulling. Pin every new import exactly, once, in the right requirements layer. A venv that has drifted from the pins fails tests for unrelated reasons (`transformers` 5.x breaks `tests/test_train.py`).
- Run tests with `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python -m pytest` and lint with `python -m ruff check .`. Both must pass before a PR. `python .github/scripts/prefetch_hf_assets.py` caches the pinned tokenizer in `.venv/hf-cache`; run it inside each git worktree.
- Don't commit `__pycache__/`, `.pyc`, `.DS_Store`, model weights, SQLite databases or anything under `outputs/` or `models/`.

# Current Status of Project

Status as of 2026-11-17, including the Wave 2 serving branches. The docs index in `docs/README.md` links every guide.

**Done**
- Data and labels: CUAD loader, any-overlap label projection, grouped contract splits. Experiments use train/calibration/evaluation contract partitions of the training file; `test.json` has never been read.
- Chunking: the sentence-window chunker contains 98.8% of annotated spans on the development contracts (`docs/chunking-experiments.md`).
- Baseline: word+char TF-IDF, mean macro-F1 0.505 and macro-AUPR 0.563 over five seeded splits (0.533 / 0.597 on seed 42). Released as the bundle `model-word-char-tfidf-v1`.
- Alternatives screened against it: retrieval reranker, contract-wide Transformer, category-query, graph, contrastive, set prediction and sparse long context. None robustly beats TF-IDF; the reranker's +0.016 mean macro-F1 has bootstrap intervals that include zero.
- Error analysis and threshold-strategy study (`docs/error-analysis.md`, `docs/threshold-robustness.md`).
- Model registry: `ClauseModelBundle` plus MLflow on SQLite.
- Evidence extraction with span validation and an Unknown queue.
- Risk layer: four signals (`category_severity`, `one_sidedness`, `magnitude`, `contract_context`), versioned weight sets, clause scoring, contract triage score and sensitivity analysis (`scripts/evaluate_triage.py`). The default weight set is `v2-team-proposed`, which names a severity tier for all 41 categories (`docs/risk-weights-v2.md`). The team set it; it has not been validated by a legal expert.
- Document ingestion: PDF, Word and text readers with size, page and text limits (`docs/document-ingestion.md`).
- Clause register: core schema with CSV and JSON export, and a batch export for the development contracts (`docs/clause-register.md`).
- Artifact store: persists inference results, Unknowns, scores and weight versions, with a `feedback` table and rescoring from cached probabilities.
- Encoder comparison: resumable five-seed runner, Colab notebook and a predeclared verdict rule (`docs/encoder-comparison.md`). Run on a Colab T4 on 2026-11-17 with verdict `does_not_beat` (`results/encoder_vs_tfidf/`).
- End-to-end inference and the Gradio quick demo: upload PDF, Word or text, rank contracts, show risky clauses with page numbers and quotes, download a CSV clause register.
- FastAPI backend (`docs/api.md`): background scoring with persisted job status, ranked reads, contract detail, `register.csv` and `register.json`, immutable weight versions rescored from cached probabilities, preview without saving, and review and feedback endpoints.
- Human in the loop (`docs/feedback-review-queue.md`): review queue, validated verdicts with conflict handling, and a deterministic leakage-safe training export.
- Reviewer page (`docs/reviewer-ui.md`): dataset contracts and uploads in one ranking, contract detail and registers, weight editing with preview and save-as-new-version, and the review tab.
- Tooling: layered pinned requirements, ruff (check only) and GitHub Actions CI.

**Gaps against ARCHITECTURE.md**
- The encoder does not beat the baseline. On the five-seed run DistilRoBERTa lost to word+char TF-IDF on every seed (macro-F1 lower by 0.063 to 0.082), mostly on uncommon and rare categories. Word+char TF-IDF is the model to freeze.
- Risk rankings have no human validation: the team decided not to collect hand-ranked clauses. The Spearman in `results/contract_triage/` is against a perfect-detection reference built from CUAD annotations, which measures detection error only.
- The reviewer page loads, switches tabs and downloads registers in headless Chrome. Its other controls have been driven only through their handlers and `gradio_client`, and no real reviewer has used it.
- The API has no authentication, no time limit on parsing or scoring, and has not been load-tested. Run it on localhost only.
- On real contracts the review queue is dominated by `no_cue_match` Unknowns, a symptom of the weak evidence heuristic.
- No Docker or license. `pyproject.toml` still has per-file lint suppressions to clear.
- The final `test.json` evaluation has not been run, and there is no final report.
- Evidence heuristic is weak: about 55% of grounded quotes overlap a gold answer, and a false-positive prediction still gets a plausible quote.
- Rare categories (under 10 evaluation positives) swing 0.3 to 0.4 F1 between seeds; report them with intervals or as insufficient support.
- PDF extraction rarely reproduces CUAD text verbatim (7 of 60 gold spans on the sample PDF, 60 of 60 ignoring whitespace and curly quotes). Grounding against the extracted text is unaffected. The OCR path is tested with mocks only.

**Next**
- Freeze the model (word+char TF-IDF) and make one sealed `test.json` evaluation.
- Final report, license, Docker, and the `src` to `clausetriage` package rename.
