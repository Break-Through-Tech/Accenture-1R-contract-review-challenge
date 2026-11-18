# Research documentation

Guides for the pipeline stages and the reproducible experiments:

| Topic | Guide | Results |
|---|---|---|
| Final report: results, limitations, estimated reviewer time saved | [Final report](final-report.md) | Draws on `results/final_test_evaluation/`, `results/encoder_vs_tfidf/` and `results/contract_triage/` |
| Word + character TF-IDF baseline | [Alternative architecture experiment](alternative-architecture-experiment.md) | `results/word_char_tfidf_nested/` |
| Sparse model alternatives | [Architecture suite](architecture-suite.md) | `results/architecture_suite_seed{7,17,23,42}/` |
| Pretrained encoder comparison | [Encoder comparison](encoder-comparison.md) | `results/encoder_nested_seed{42,7,17,23,101}/`, `results/encoder_vs_tfidf/` (run on a Colab T4 on 2026-11-17; verdict `does_not_beat`) |
| Contract-wide Transformer | [Long-context architecture](long-context-architecture.md) | `results/long_context_transformer_seed42/` |
| Category-query evidence model | [Category-query architecture](category-query-architecture.md) | `results/category_query_seed42/` |
| MLflow registry and SQLite predictions | [Registry and artifact store](model-registry-and-artifact-store.md) | Local databases and model weights under ignored `outputs/` |
| Next architecture directions | [Experiment roadmap](architecture-experiment-roadmap.md) | Five planned/implemented directions and evaluation policy |
| Retrieval + cross-encoder reranker | [Retrieval/reranker](retrieval-reranker-architecture.md) | `results/retrieval_reranker_seed{42,7,17,23,101}/` |
| TF-IDF + reranker score fusion | [Score fusion](tfidf-reranker-fusion.md) | `results/tfidf_reranker_fusion/`, `results/reranker_oof_fusion_seed42/` (exploratory; reused evaluation contracts) |
| Error analysis (TF-IDF vs. reranker) | [Error analysis](error-analysis.md) | `results/error_analysis/` (also documents the per-category AUPR fix) |
| Threshold strategies (rare categories) | [Threshold robustness](threshold-robustness.md) | `results/threshold_robustness/` (exploratory; reused evaluation contracts) |
| Confirmation gate | [Upgrade validation protocol](upgrade-validation-gate.md) | Team data-source and success-criteria decision needed |
| Final `test.json` evaluation (pre-registered, run once on 2026-11-18) | [Final test evaluation](final-test-evaluation.md) | `results/final_test_evaluation/` (macro-F1 0.539, macro-AUPR 0.608 on 102 test contracts; consistent with development); `results/final_test_dry_run/` (development dry run) |
| Contract graph neural network | [Graph architecture](contract-graph-architecture.md) | `results/contract_graph_seed42/` |
| Supervised contrastive prototypes | [Contrastive architecture](contrastive-architecture.md) | `results/contrastive_seed42/` |
| Hungarian-matched span set prediction | [Set prediction](set-prediction-architecture.md) | `results/set_prediction_seed42/` |
| Local/global sparse long context | [Sparse long context](sparse-long-context-architecture.md) | `results/sparse_long_context_seed42/` |
| Chunking experiments | [Chunking evaluation](chunking-experiments.md) | `results/token_chunking_development/`, `results/token_chunking_comparison/` |
| Risk weights (`v2-team-proposed`) | [Risk weights v2](risk-weights-v2.md) | `src/risk/weights/default_weights_v2.json`, `results/contract_triage/` |
| Uploaded contracts (PDF, Word, text) | [Document ingestion](document-ingestion.md) | `python -m scripts.evaluate_pdf_ingestion` |
| Clause register (CSV and JSON) | [Clause register](clause-register.md) | `outputs/registers/` (git-ignored; `python -m scripts.export_registers`) |
| FastAPI server (`python -m src.api`) | [API](api.md) | `outputs/api/clausetriage.sqlite` (git-ignored); local only, no authentication |
| Review queue, feedback and training export | [Feedback and the review queue](feedback-review-queue.md) | `outputs/feedback_export/` (git-ignored; `python -m scripts.export_feedback`) |
| Reviewer page at `/ui/` | [Reviewer UI](reviewer-ui.md) | Loads in headless Chrome; not yet used by a real reviewer |
| Setup, tests, lint and CI | [Development](development.md) | |

All candidate and development metrics use contract-level splits of the training file. The one
exception is the sealed final evaluation, which read `data/cuad/test.json` once, under a
pre-registered protocol. Word + character TF-IDF remains the supported control. A score-fusion
prototype exceeded it on previously inspected splits, but is explicitly exploratory and
not ready for promotion. The neural candidates add contextual signals; reports document
the protocol and limitations for each result.
