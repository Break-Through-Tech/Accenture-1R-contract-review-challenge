# Model registry and artifact store

## Model/prediction contract

`ClauseModelBundle` is the stable inference boundary: a predictor accepts chunk text
and returns raw probabilities shaped `(chunks, 41)` in the exact canonical
`CATEGORIES` order. The bundle holds category order, calibrated per-category
thresholds, model name, configuration, and seed. Thresholding is downstream of raw
probabilities so rescoring does not require rerunning the classifier.

`src/registry/mlflow_registry.py` stores each serialized bundle as a numbered MLflow
registered-model version, using a SQLite tracking database and local artifact
folder. Every version logs its evaluation metrics, seed, category schema hash,
thresholds, config, and caller-supplied data/split metadata. Load a specific version
rather than implicitly using `latest` in reproducible scoring.

Example after fitting a model and calibrating thresholds:

```python
from src.labels.categories import CATEGORIES
from src.registry.bundle import ClauseModelBundle
from src.registry.mlflow_registry import register_bundle
from src.registry.predictors import VectorizedClassifier

bundle = ClauseModelBundle(
    predictor=VectorizedClassifier(fitted_vectorizer, fitted_classifier),
    categories=CATEGORIES,
    thresholds=calibrated_thresholds,
    model_name="word_char_tfidf",
    config={"c": 1.0, "word_ngrams": [1, 2], "char_wb_ngrams": [3, 5]},
    seed=42,
)
registered = register_bundle(
    bundle, experiment_name="clause-review", registry_name="clause-classifier",
    metrics={"evaluation_macro_f1": 0.53}, metadata={"split": "...", "data_sha256": "..."},
    tracking_uri="results/mlflow/tracking.db",
)
```

Do not use held-out evaluation metrics to choose thresholds or hyperparameters. Use
an agreed calibration set for thresholds and record the untouched evaluation
partition and protocol in `metadata`.

## Artifact store

`ArtifactStore` creates SQLite tables for contracts, exact source chunks, raw
per-category chunk probabilities keyed by model version, grounded evidence,
weight versions, clause scores, and contract scores. Stable chunk and evidence IDs
are supplied by callers. Use the registry-qualified `registered.model_key` (for
example, `clause-classifier:3`) rather than a bare version because separate models
can each have a version `1`. Contract text is stored so chunk/evidence offsets can
be checked against the exact source. Prediction writes are idempotent for matching
values and reject conflicting overwrites; a new model version creates separate
probability rows. This initial layer does not implement risk scoring, weight CRUD,
feedback/advisor ranking APIs, migrations, or concurrent production serving.

```python
from src.store.artifact_store import ArtifactStore

store = ArtifactStore("results/artifacts.sqlite")
store.put_contract(contract.contract_id, contract.text)
store.put_chunk(chunk_id=f"{contract.contract_id}:{chunk.chunk_index}",
                contract_id=contract.contract_id, chunk_index=chunk.chunk_index,
                start=chunk.start, end=chunk.end, text=chunk.text)
store.put_probabilities(chunk_id, registered.model_key, CATEGORIES, probabilities)
```

Install dependencies with `uv pip install --python .venv/bin/python -r requirements.txt`.
The local test suite exercises MLflow save/load and SQLite persistence in temporary
directories, including a registered-bundle-to-artifact-store prediction roundtrip.

The local registry currently contains the long-context Transformer
`clause-classifier-long-context:1`, the category-query evidence candidate
`clause-classifier-category-query:2` (version 2 exposes attention through
`predict_with_evidence`), and the word+character TF-IDF control
`clause-classifier-word-char-tfidf:1`. The neural models remain candidates, not
replacements for the stronger sparse baseline. TF-IDF version 1 was fitted on the
seed-42 training contracts; thresholds were calibrated on calibration contracts only.
Its recorded evaluation metrics are diagnostic because those contracts have prior
research exposure. Registration metadata includes dataset hash, split ID hashes, model
configuration, threshold provenance, and explicit `test_data_used: false`.

Recreate/register the TF-IDF artifact locally with:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m scripts.register_tfidf_bundle
```

This creates a new numbered version if rerun; it does not promote it to a production
alias or deployment. The existing neural reports are in
`results/long_context_transformer_seed42/` and `results/category_query_seed42/`.
The registry database, artifact database, and model weights are local and ignored by
Git. Do not commit datasets or model weights.
