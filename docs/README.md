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

All reported candidate metrics use contract-level splits and leave `data/cuad/test.json`
untouched. The word + character TF-IDF model is the current strongest classifier on
these experiments; the neural candidates additionally expose contract context and,
for the category-query model, an evidence-attention signal. Reports document the
protocol and limitations for each result.
