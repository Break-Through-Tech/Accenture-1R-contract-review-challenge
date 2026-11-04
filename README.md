# Accenture Contract Review Challenge

Research project exploring automated identification of contract clauses and supporting evidence using the Contract Understanding Atticus Dataset (CUAD). The repository contains data preparation, classifier baselines, alternative model experiments, evaluation reports, and an early rule-based risk-scoring prototype.

> **Status:** Research/prototype work, not a production legal-review system. The risk-scoring signals and weights are placeholders pending advisor/team approval; outputs must not be used as legal advice or operational triage.

## Team Members

The role summaries combine work visible in merged PRs or commits with relevant GitHub-assigned responsibilities. Assignments describe expected scope, not proof of completion. Please confirm the official AI Studio roster and wording with the team.

| Name | Role and contributions |
|---|---|
| Edward Hu | Built the project architecture and guidance docs, EDA documentation and visualizations, baseline correction, chunk schema, CUAD loader, labels, and grouped splits (PRs #38, #39, #40, #42, #43, #44, #45). Also assigned technical writing, model training/inference, API, testing, and final README work (#8, #17, #21, #27, #35, #36). |
| Sebastian Bernal-Orta | Evaluated the word/character TF-IDF baseline with contract-level, multi-seed splits, then implemented five alternatives: retrieval/reranking, GraphSAGE, contrastive prototypes, set prediction, and sparse long-context. Added their training/evaluation code, tests, reports, MLflow registry, SQLite artifact-store foundation, and risk-weight provenance hardening in PR #51. Assigned alternative architecture, model registry, and artifact store work (#14, #19, #32); the implementations are submitted in PR #51, while team risk-signal decision #25 remains open. |
| Stanley Huang | Built the TF-IDF baseline and thresholding, evidence extraction, and the rule-based risk-scoring prototype (#22, #23; baseline PR #37). Also assigned contract-level triage, team risk-signal decisions, and moving the baseline to the shared pipeline (#24, #25, #46). |
| Areebah Iqbal | Completed the initial CUAD EDA (#1). Assigned advisor-owned risk weights, clause-register export, and reviewer UI work (#28, #29, #30). |
| Nora Farhan | Assigned evaluation metrics, contract-level risk triage, team risk-signal decisions, and the human-review/feedback workflow (#18, #24, #25, #33). |
| Osamwengumwenro Oni-Ojo | Integrated sentence-aware chunking into the CUAD pipeline, with optimization and tests (merged PR #50). Also assigned data-preprocessing, chunking, label preparation, and pipeline-architecture tasks (#11, #12, #13, #16); token-aware prototype PR #48 is open. |
| Ruvarashe Nyabando | Evaluated token- and sentence-aware chunking (merged PR #49). Also assigned data-preprocessing/chunking, label preparation, and pipeline-architecture work (#11, #12, #13, #16). |

## Project Highlights

**Draft section. Complete this with the team and advisor before presenting the project externally.**

- Primary project contribution: **[team to finalize]**
- Final model/result to highlight: **[team to select after robustness review]**
- Stakeholder impact: **[state only after validation with Accenture and the project advisor]**

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

## Project Overview

This Break Through Tech AI Studio project, hosted by Accenture, explores multi-label contract-clause detection across CUAD's 41 categories, with evidence spans grounded in the original contract text. The research compares sparse text-classification baselines with retrieval, graph, contrastive, set-prediction, and long-context neural approaches. A separate rule-based risk-scoring prototype explores how detected clauses might be prioritized; its signal definitions and weights are not approved risk policy.

## Data Exploration

CUAD v1 contains 510 commercial contracts annotated for 41 clause categories. The prepared training file has 408 contracts; the separate test file has 102. The training-set EDA summary reports 22,450 category questions, 11,180 with answers and 11,270 without. Median contract length is 35,960.5 characters (5,357.5 words); 33 unusually long contracts were identified by the IQR rule and retained.

Experiments use contract-level partitions of the training file so chunks from a contract cannot cross partitions. The shared seed-42 split is 260 training, 66 calibration, and 82 evaluation contracts. **The supplied `data/cuad/test.json` was not read or modified.** See [`data/cuad/README.md`](data/cuad/README.md) for dataset provenance and [`data/cuad/EDA_SUMMARY.md`](data/cuad/EDA_SUMMARY.md) for the EDA details.

## Model Development

The word- and character-level TF-IDF classifier is the main control. Experiments also include word-only and other sparse baselines, a contract-wide Transformer, category-query evidence attention, and five further architectures: retrieval/reranking, contract GraphSAGE, supervised contrastive prototypes, Hungarian-matched span set prediction, and local/global sparse contract context.

The repository also contains local MLflow registration and SQLite prediction/artifact-store foundations. These are research infrastructure, not a deployed service. See the [research documentation index](docs/README.md) for model descriptions, commands, reports, and limitations.

## Results and Key Findings

Seed-42 held-out evaluation results below are chunk-level macro-F1 / macro-AUPR. Thresholds are fit on calibration contracts only.

| Model | Macro-F1 | Macro-AUPR |
|---|---:|---:|
| Word + character TF-IDF control | **0.533** | **0.597** |
| Retrieval + reranker | 0.557 | 0.595 |
| Contract-wide Transformer | 0.496 | 0.507 |
| Category-query evidence model | 0.478 | 0.512 |
| Contract graph network | 0.390 | 0.426 |
| Contrastive prototypes | 0.365 | 0.372 |
| Local/global sparse context | 0.142 | 0.131 |
| Hungarian-matched set prediction | 0.099 | 0.088 |

The reranker had higher seed-42 macro-F1 but slightly lower macro-AUPR than the control and needs multi-seed confirmation. The other neural candidates did not outperform the TF-IDF control in these screening runs. These results are research comparisons, not evidence of production readiness or legal-review impact. Detailed metrics, protocol, and caveats are in each linked experiment report under [`results/`](results/) and its guide in [`docs/`](docs/).

### Chunking research

Earlier chunk-boundary and token-budget experiments are summarized in the [chunking experiments guide](docs/chunking-experiments.md), with detailed results under `results/token_chunking_*`. The measurements describe annotated-span containment and token-budget behavior, not classifier quality. The heading below is retained as a redirect target for an older PR link.

### PR #48 comparison on the saved development split

See the [chunking experiments guide](docs/chunking-experiments.md) for the archived comparison and reproduction details.

## Next Steps

- Confirm the official team roster and finish the draft project highlights with the team/advisor.
- Continue multi-seed evaluation of promising candidates before selecting a model.
- Review the risk signals and weights with the advisor; do not use placeholder risk scores for real triage.
- Confirm deployment, data-governance, and licensing requirements before proposing production use.

## License

No license for this project's code has been selected in the repository. Confirm an appropriate license with the advisor before reuse or redistribution. The CUAD dataset has its own license and attribution requirements; see [`data/cuad/README.md`](data/cuad/README.md). The dataset license does not automatically license this project's code.

## References

- The Atticus Project, [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad/) and [official repository](https://github.com/TheAtticusProject/cuad).
- Hendrycks et al., [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268), NeurIPS 2021.
- Base encoder: [`distilbert/distilroberta-base`](https://huggingface.co/distilbert/distilroberta-base), pinned to the revision recorded in experiment configs and reports.

## Acknowledgements

**Draft:** [Add advisor, Accenture, Break Through Tech, and other acknowledgements after confirming names and wording with the team.]
