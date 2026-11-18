# Accenture Contract Review Challenge

Research project exploring automated identification of contract clauses and supporting evidence using the Contract Understanding Atticus Dataset (CUAD). The repository contains data preparation, classifier baselines, alternative model experiments, evaluation reports, a rule-based risk-scoring layer, and a local prototype for uploading, ranking and reviewing contracts. The [final report](docs/final-report.md) summarizes the results, limitations and an estimate of reviewer time saved.

> **Status:** Research/prototype work, not a production legal-review system. The risk-scoring signals and weights are set by the team and have not been validated by a legal expert; outputs must not be used as legal advice or operational triage.

## Team Members

The role summaries combine work visible in merged PRs or commits with relevant GitHub-assigned responsibilities. Assignments describe expected scope, not proof of completion. Please confirm the official AI Studio roster and wording with the team.

| Name | Role and contributions |
|---|---|
| Edward Hu | Built the project architecture and guidance docs, EDA documentation and visualizations, baseline correction, chunk schema, CUAD loader, labels, and grouped splits (PRs #38, #39, #40, #42, #43, #44, #45). Also assigned technical writing, model training/inference, API, testing, and final README work (#8, #17, #21, #27, #35, #36). |
| Sebastian Bernal-Orta | Evaluated the word/character TF-IDF baseline with contract-level, multi-seed splits, then implemented five alternatives: retrieval/reranking, GraphSAGE, contrastive prototypes, set prediction, and sparse long-context. Added their training/evaluation code, tests, reports, MLflow registry, SQLite artifact-store foundation, and risk-weight provenance hardening in PR #51. Assigned alternative architecture, model registry, and artifact store work (#14, #19, #32); the implementations are submitted in PR #51, while team risk-signal decision #25 remains open. |
| Stanley Huang | Built the TF-IDF baseline and thresholding, evidence extraction, and the rule-based risk-scoring prototype (#22, #23; baseline PR #37). Also assigned contract-level triage, team risk-signal decisions, and moving the baseline to the shared pipeline (#24, #25, #46). |
| Areebah Iqbal | Completed the initial CUAD EDA (#1). Assigned risk weights, clause-register export, and reviewer UI work (#28, #29, #30). |
| Nora Farhan | Assigned evaluation metrics, contract-level risk triage, team risk-signal decisions, and the human-review/feedback workflow (#18, #24, #25, #33). |
| Osamwengumwenro Oni-Ojo | Integrated sentence-aware chunking into the CUAD pipeline, with optimization and tests (merged PR #50). Also assigned data-preprocessing, chunking, label preparation, and pipeline-architecture tasks (#11, #12, #13, #16); token-aware prototype PR #48 is open. |
| Ruvarashe Nyabando | Evaluated token- and sentence-aware chunking (merged PR #49). Also assigned data-preprocessing/chunking, label preparation, and pipeline-architecture work (#11, #12, #13, #16). |

## Project Highlights

- **An end-to-end clause triage prototype.** Contracts uploaded as PDF, Word or text are split into passages, checked for the 41 CUAD clause categories, rated Low, Medium or High risk by team-set rules, and ranked for review. Each contract gets a clause register (CSV and JSON) whose quotes are exact contract text.
- **One sealed evaluation on unseen contracts.** The released word and character TF-IDF model scored macro-F1 0.539 (95% interval 0.514 to 0.595) and macro-AUPR 0.608 (0.567 to 0.641) on the 102 CUAD test contracts, read once under a protocol fixed in advance. Both are consistent with development.
- **A tested negative result.** A fine-tuned DistilRoBERTa encoder lost to the TF-IDF model on all five development splits, and seven other neural designs did not robustly beat it.
- **Stated limits.** The risk ratings have not been validated against a reviewer or a legal expert, evidence quotes are weak, and the reviewer time saved is an estimate, not a measurement. See the [final report](docs/final-report.md).

## Setup and Installation

The project uses Python 3.12 and the dependencies in `requirements.txt`.

```bash
git clone https://github.com/Break-Through-Tech/Accenture-1R-contract-review-challenge.git
cd Accenture-1R-contract-review-challenge
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Several model experiments require the pinned DistilRoBERTa tokenizer and encoder. Download them once to the cache used by the scripts:

```bash
.venv/bin/python - <<'PY'
from transformers import AutoModel, AutoTokenizer
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
cache = ".venv/hf-cache"
AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True, cache_dir=cache)
AutoModel.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, cache_dir=cache)
PY
```

Run the tests and reproduce the word/character TF-IDF comparison:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m pytest
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m scripts.evaluate_word_char_tfidf
```

The evaluation command uses the CUAD training split, a fixed contract-level train/calibration/evaluation protocol, and predeclared seeds. Individual architecture training commands and settings are documented in [`docs/README.md`](docs/README.md). Some neural experiments benefit from CUDA; CPU execution may be substantially slower.

### Demo UI

There are two pages. Both are prototypes. Their scores use the team-proposed `v2-team-proposed` risk weights, which have not been validated by a legal expert, and they are not legal advice.

- **Quick demo** (`python -m src.ui.app`): upload contracts (PDF, Word, text, or pasted text) to rank them for review, and see each contract's risky clauses with page numbers, highlighted quotes and the signals behind each score. Everything is scored in memory and nothing is saved. It needs only the small demo install.
- **Reviewer page** (`python -m src.api`, then http://127.0.0.1:8000/ui/): the full triage loop on stored data. Stored CUAD contracts and your uploads are ranked together, and you can open any contract's detail and download its clause register. You can edit and preview the risk weights, save them as a new version, and work through the review queue, recording verdicts. It needs the full install (it uses the SQLite artifact store). See [`docs/reviewer-ui.md`](docs/reviewer-ui.md).

To run only the quick demo, a smaller install is enough (no PyTorch, MLflow or SQLAlchemy; Python 3.12 or 3.13):

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements-demo.txt
python -m src.ui.app
```

If you already installed `requirements.txt`, just run `python -m src.ui.app`. Then open http://127.0.0.1:7860 and upload contracts, or click **Try the sample contracts** (from [`data/demo_contracts/`](data/demo_contracts/SOURCES.md)). Use **Clause register (CSV)** to download every scored clause.

For the reviewer page, after `pip install -r requirements.txt`:

```bash
python -m scripts.load_dataset_contracts --limit 10    # optional: score CUAD validation contracts into the store
python -m src.api                                      # page at http://127.0.0.1:8000/ui/, API docs at /docs
```

- **Model:** on first run, the trained word/character TF-IDF model is downloaded from the GitHub Release [`model-word-char-tfidf-v1`](https://github.com/Break-Through-Tech/Accenture-1R-contract-review-challenge/releases/tag/model-word-char-tfidf-v1) to `models/word_char_tfidf.joblib` and checked against its SHA-256; the tokenizer (a few MB) downloads too. If the download fails, or the file doesn't load with your scikit-learn version, the UI builds the same model from the CUAD training data in the repo instead (a few minutes, once). To build it yourself, run `python -m scripts.build_model_bundle`; add `--register` to also log it to a local MLflow registry, then start the UI with `--registry-version latest`.
- **Scanned PDFs** need OCR, which is optional: `pip install pypdfium2 pytesseract` and install the [Tesseract](https://github.com/tesseract-ocr/tesseract) program. PDFs with a text layer need nothing extra.
- Evidence positions and page numbers refer to the text extracted from the file. Only load model files you built or downloaded from this repository, since loading one can run code.

## Project Overview

This Break Through Tech AI Studio project, hosted by Accenture, explores multi-label contract-clause detection across CUAD's 41 categories, with evidence spans grounded in the original contract text. The research compares sparse text-classification baselines with retrieval, graph, contrastive, set-prediction, and long-context neural approaches. A separate rule-based risk-scoring prototype explores how detected clauses might be prioritized; its signal definitions and weights are not approved risk policy.

## Data Exploration

CUAD v1 contains 510 commercial contracts annotated for 41 clause categories. The prepared training file has 408 contracts; the separate test file has 102. The training-set EDA summary reports 22,450 category questions, 11,180 with answers and 11,270 without. Median contract length is 35,960.5 characters (5,357.5 words); 33 unusually long contracts were identified by the IQR rule and retained.

Experiments use contract-level partitions of the training file so chunks from a contract cannot cross partitions. The shared seed-42 split is 260 training, 66 calibration, and 82 evaluation contracts. **The supplied `data/cuad/test.json` was read once, for the sealed final evaluation on 2026-11-18 ([protocol and results](docs/final-test-evaluation.md)), and was not used for training, tuning or model selection.** See [`data/cuad/README.md`](data/cuad/README.md) for dataset provenance and [`data/cuad/EDA_SUMMARY.md`](data/cuad/EDA_SUMMARY.md) for the EDA details.

## Model Development

The word and character TF-IDF classifier is the main control and the released model. Experiments also include word-only and other sparse baselines, a fine-tuned DistilRoBERTa encoder, a contract-wide Transformer, category-query evidence attention, and five further architectures: retrieval/reranking, contract GraphSAGE, supervised contrastive prototypes, Hungarian-matched span set prediction, and local/global sparse contract context.

The repository also contains local MLflow registration, a SQLite artifact store, a local API and a reviewer page. These are a research prototype for one machine, not a deployed service. See the [research documentation index](docs/README.md) for model descriptions, commands, reports, and limitations.

## Results and Key Findings

**Final evaluation.** The released word + character TF-IDF model was evaluated once on the 102 contracts of `test.json`, under a protocol registered before the file was opened. Results are chunk-level, with 95% contract-bootstrap (BCa) intervals. Details are in the [final test evaluation](docs/final-test-evaluation.md) and the [final report](docs/final-report.md).

| Metric | Test (95% interval) | Development: five-split mean (range) |
|---|---|---|
| Macro-F1, 41 categories | 0.539 (0.514 to 0.595) | 0.504 (0.473 to 0.533) |
| Macro-AUPR | 0.608 (0.567 to 0.641) | 0.563 (0.531 to 0.597) |
| Original word-only TF-IDF, macro-F1 | 0.480 (0.459 to 0.524) | 0.432 (0.396 to 0.476) |

Both headline results are consistent with development under the registered reading rule. Common categories score far higher than rare ones (mean test F1 0.650 for the 19 common categories, 0.361 for the 9 rare ones). On the test contracts, 23% of detected clauses could not be backed by a grounded quote, and 36% of the quotes produced overlap the annotated answer. Risk rankings have no human validation: the team decided not to collect hand-ranked clauses, so agreement with a reviewer is not measured.

**Development comparison.** The seed-42 results below are chunk-level macro-F1 / macro-AUPR on the 82 development evaluation contracts. Thresholds are fit on calibration contracts only.

| Model | Macro-F1 | Macro-AUPR |
|---|---:|---:|
| Word + character TF-IDF control | **0.533** | **0.597** |
| Retrieval + reranker | 0.557 | 0.595 |
| Contract-wide Transformer | 0.496 | 0.507 |
| Category-query evidence model | 0.478 | 0.512 |
| Fine-tuned DistilRoBERTa | 0.451 | 0.476 |
| Contract graph network | 0.390 | 0.426 |
| Contrastive prototypes | 0.365 | 0.372 |
| Local/global sparse context | 0.142 | 0.131 |
| Hungarian-matched set prediction | 0.099 | 0.088 |

A five-seed follow-up found the reranker improves mean macro-F1 by 0.016 and wins four of five splits, but mean macro-AUPR is 0.006 lower and all per-seed paired contract-bootstrap intervals include zero. It is not established as a robust improvement. A TF-IDF plus reranker score-fusion prototype scored higher on both macro metrics across the same five splits, but those evaluation splits had already been inspected and the fusion remains exploratory. Do not promote it without an untouched approved holdout. A fine-tuned DistilRoBERTa encoder, trained on the same five splits, lost to the TF-IDF control on every split (macro-F1 lower by 0.063 to 0.082; see the [encoder comparison](results/encoder_vs_tfidf/vs_word_char_tfidf/summary.md)), mostly on categories with few examples. The other neural candidates did not outperform the TF-IDF control in these screening runs. These results are research comparisons, not evidence of production readiness or legal-review impact. Detailed metrics, protocol, and caveats are in each linked experiment report under [`results/`](results/) and its guide in [`docs/`](docs/).

### Chunking research

Earlier chunk-boundary and token-budget experiments are summarized in the [chunking experiments guide](docs/chunking-experiments.md), with detailed results under `results/token_chunking_*`. The measurements describe annotated-span containment and token-budget behavior, not classifier quality. The heading below is retained as a redirect target for an older PR link.

### PR #48 comparison on the saved development split

See the [chunking experiments guide](docs/chunking-experiments.md) for the archived comparison and reproduction details.

## Next Steps

- Collect reviewer judgement: hand rankings on a sample of contracts, to measure agreement between the risk rankings and a reviewer, and real verdicts through the review queue.
- Validate the risk signals and weights ([docs/risk-weights-v2.md](docs/risk-weights-v2.md)) against expert judgement before any real use; the risk scores are unvalidated.
- Improve evidence extraction and the rare-category thresholds. The test set has now been used, so any changed model needs new held-out contracts for its evaluation.
- Measure detection quality on text extracted from PDFs.
- Add authentication and time limits, and confirm deployment and data-governance requirements, before proposing use beyond a local machine.

## License

The project's code is released under the [MIT License](LICENSE), copyright 2026 The ClauseTriage contributors.

The CUAD dataset has its own license, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and attribution requirements; see [`data/cuad/README.md`](data/cuad/README.md). The MIT License does not cover the dataset, and the dataset license does not cover this project's code. The sample contracts in `data/demo_contracts/` come from the sources listed in [`SOURCES.md`](data/demo_contracts/SOURCES.md).

## References

- The Atticus Project, [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad/) and [official repository](https://github.com/TheAtticusProject/cuad).
- Hendrycks et al., [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268), NeurIPS 2021.
- Base encoder: [`distilbert/distilroberta-base`](https://huggingface.co/distilbert/distilroberta-base), pinned to the revision recorded in experiment configs and reports.

## Acknowledgements

- Accenture, which hosted this challenge, and Gopinath Kolluru, the challenge advisor.
- Break Through Tech AI Studio and Parth Dali, the team's AI Studio coach.
- Hari Raghavan, who created this repository with the challenge brief and the CUAD data files.
- The Atticus Project, for the Contract Understanding Atticus Dataset (CUAD).
