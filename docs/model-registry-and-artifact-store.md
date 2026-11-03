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
are supplied by callers. Contract text is stored so chunk/evidence offsets can be
checked against the exact source. Prediction writes are idempotent for matching
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
store.put_probabilities(chunk_id, registered.model_version, CATEGORIES, probabilities)
```

Install dependencies with `uv pip install --python .venv/bin/python -r requirements.txt`.
The local test suite exercises MLflow save/load and SQLite persistence in temporary
directories, including a registered-bundle-to-artifact-store prediction roundtrip.

The long-context Transformer candidate has also been registered locally as
`clause-classifier-long-context` version `1` (experiment `clause-review-long-context`).
Its held-out metrics are recorded in
`results/long_context_transformer_seed42/report.json`; it is a candidate, not a
replacement for the stronger word+character baseline. A CPU-loaded registry bundle
was smoke-tested through inference and its 41 raw probabilities were persisted to
`outputs/artifact_store.sqlite` under that model version. Both outputs are local and
ignored by Git. Keep generated registry databases and model artifacts local; do not
commit datasets or model weights.
