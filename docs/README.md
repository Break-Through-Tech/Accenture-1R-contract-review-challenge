# Research documentation

Start here for the reproducible experiments added on this branch:

| Topic | Guide | Results |
|---|---|---|
| Word + character TF-IDF baseline | [Alternative architecture experiment](alternative-architecture-experiment.md) | `results/word_char_tfidf_nested/` |
| Sparse model alternatives | [Architecture suite](architecture-suite.md) | `results/architecture_suite_seed{7,17,23,42}/` |
| Pretrained encoder comparison | [Encoder comparison](encoder-comparison.md) | See each model report for split and run status |
| Contract-wide Transformer | [Long-context architecture](long-context-architecture.md) | `results/long_context_transformer_seed42/` |
| Category-query evidence model | [Category-query architecture](category-query-architecture.md) | `results/category_query_seed42/` |
| MLflow registry and SQLite predictions | [Registry and artifact store](model-registry-and-artifact-store.md) | Local databases and model weights under ignored `outputs/` |
| Next architecture directions | [Experiment roadmap](architecture-experiment-roadmap.md) | Five planned/implemented directions and evaluation policy |
| Retrieval + cross-encoder reranker | [Retrieval/reranker](retrieval-reranker-architecture.md) | `results/retrieval_reranker_seed{42,7,17,23,101}/` |
| TF-IDF + reranker score fusion | [Score fusion](tfidf-reranker-fusion.md) | `results/tfidf_reranker_fusion/`, `results/reranker_oof_fusion_seed42/` (exploratory; reused evaluation contracts) |
| Category cue logit boost | [Cue boost ablation](category-cue-boost.md) | `results/category_cue_boost/` (exploratory; mixed macro-F1 results) |
| Confirmation gate | [Upgrade validation protocol](upgrade-validation-gate.md) | Advisor/team data-source and success-criteria decision needed |
| Contract graph neural network | [Graph architecture](contract-graph-architecture.md) | `results/contract_graph_seed42/` |
| Supervised contrastive prototypes | [Contrastive architecture](contrastive-architecture.md) | `results/contrastive_seed42/` |
| Hungarian-matched span set prediction | [Set prediction](set-prediction-architecture.md) | `results/set_prediction_seed42/` |
| Local/global sparse long context | [Sparse long context](sparse-long-context-architecture.md) | `results/sparse_long_context_seed42/` |
| Chunking experiments | [Chunking evaluation](chunking-experiments.md) | `results/token_chunking_development/`, `results/token_chunking_comparison/` |

All reported candidate metrics use contract-level splits and leave `data/cuad/test.json`
untouched. Word + character TF-IDF remains the supported control. A score-fusion
prototype exceeded it on previously inspected splits, but is explicitly exploratory and
not ready for promotion. The neural candidates add contextual signals; reports document
the protocol and limitations for each result.
