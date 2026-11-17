# Research documentation

Guides for the pipeline stages and the reproducible experiments:

| Topic | Guide | Results |
|---|---|---|
| Word + character TF-IDF baseline | [Alternative architecture experiment](alternative-architecture-experiment.md) | `results/word_char_tfidf_nested/` |
| Sparse model alternatives | [Architecture suite](architecture-suite.md) | `results/architecture_suite_seed{7,17,23,42}/` |
| Pretrained encoder comparison | [Encoder comparison](encoder-comparison.md) | `results/encoder_nested_seed{42,7,17,23,101}/`, `results/encoder_vs_tfidf/` (not yet run; needs a GPU) |
| Contract-wide Transformer | [Long-context architecture](long-context-architecture.md) | `results/long_context_transformer_seed42/` |
| Category-query evidence model | [Category-query architecture](category-query-architecture.md) | `results/category_query_seed42/` |
| MLflow registry and SQLite predictions | [Registry and artifact store](model-registry-and-artifact-store.md) | Local databases and model weights under ignored `outputs/` |
| Next architecture directions | [Experiment roadmap](architecture-experiment-roadmap.md) | Five planned/implemented directions and evaluation policy |
| Retrieval + cross-encoder reranker | [Retrieval/reranker](retrieval-reranker-architecture.md) | `results/retrieval_reranker_seed{42,7,17,23,101}/` |
| TF-IDF + reranker score fusion | [Score fusion](tfidf-reranker-fusion.md) | `results/tfidf_reranker_fusion/`, `results/reranker_oof_fusion_seed42/` (exploratory; reused evaluation contracts) |
| Error analysis (TF-IDF vs. reranker) | [Error analysis](error-analysis.md) | `results/error_analysis/` (also documents the per-category AUPR fix) |
| Threshold strategies (rare categories) | [Threshold robustness](threshold-robustness.md) | `results/threshold_robustness/` (exploratory; reused evaluation contracts) |
| Confirmation gate | [Upgrade validation protocol](upgrade-validation-gate.md) | Advisor/team data-source and success-criteria decision needed |
| Contract graph neural network | [Graph architecture](contract-graph-architecture.md) | `results/contract_graph_seed42/` |
| Supervised contrastive prototypes | [Contrastive architecture](contrastive-architecture.md) | `results/contrastive_seed42/` |
| Hungarian-matched span set prediction | [Set prediction](set-prediction-architecture.md) | `results/set_prediction_seed42/` |
| Local/global sparse long context | [Sparse long context](sparse-long-context-architecture.md) | `results/sparse_long_context_seed42/` |
| Chunking experiments | [Chunking evaluation](chunking-experiments.md) | `results/token_chunking_development/`, `results/token_chunking_comparison/` |
| Risk weights (`v2-team-proposed`) | [Risk weights v2](risk-weights-v2.md) | `src/risk/weights/default_weights_v2.json`, `results/contract_triage/` |
| Uploaded contracts (PDF, Word, text) | [Document ingestion](document-ingestion.md) | `python -m scripts.evaluate_pdf_ingestion` |
| Clause register (CSV and JSON) | [Clause register](clause-register.md) | `outputs/registers/` (git-ignored; `python -m scripts.export_registers`) |
| Advisor agreement | [Advisor agreement](advisor-agreement.md) | No advisor rankings yet; the packet is built with `python -m scripts.build_advisor_packet` |
| Setup, tests, lint and CI | [Development](development.md) | |

All reported candidate metrics use contract-level splits and leave `data/cuad/test.json`
untouched. Word + character TF-IDF remains the supported control. A score-fusion
prototype exceeded it on previously inspected splits, but is explicitly exploratory and
not ready for promotion. The neural candidates add contextual signals; reports document
the protocol and limitations for each result.
