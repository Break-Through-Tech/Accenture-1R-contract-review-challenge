"""MLflow registry operations using a local SQLite tracking backend."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from src.registry.bundle import ClauseModelBundle


class RegisteredClauseModel:
    """Loaded versioned bundle plus its immutable MLflow identifiers."""

    def __init__(self, bundle: ClauseModelBundle, run_id: str, model_version: str,
                 registry_name: str):
        self.bundle = bundle
        self.run_id = run_id
        self.model_version = model_version
        self.registry_name = registry_name

    @property
    def model_key(self) -> str:
        """Globally unambiguous key for probability-store rows."""
        return f"{self.registry_name}:{self.model_version}"

    def predict_proba(self, texts: list[str]):
        return self.bundle.predict_proba(texts)

    def predict(self, texts: list[str]):
        return self.bundle.predict(texts)


def set_local_tracking(uri: str | Path) -> MlflowClient:
    """Configure MLflow with a SQLite URI and return its client."""
    raw = str(uri)
    if raw.startswith("sqlite:"):
        tracking_uri = raw
    else:
        path = Path(raw).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def register_bundle(
    bundle: ClauseModelBundle,
    *,
    experiment_name: str,
    registry_name: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
    tracking_uri: str | Path,
) -> RegisteredClauseModel:
    """Log a serialized bundle and register it under an incrementing model version."""
    if not metrics:
        raise ValueError("At least one evaluation metric is required")
    if any(not isinstance(value, (int, float)) for value in metrics.values()):
        raise ValueError("metrics must be numeric")
    client = set_local_tracking(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        if str(tracking_uri).startswith("sqlite:"):
            db_path = Path(str(tracking_uri).replace("sqlite:///", "", 1))
        else:
            db_path = Path(tracking_uri).expanduser().resolve()
        artifact_root = db_path.parent / f"{experiment_name}-artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=artifact_root.as_uri(),
        )
    with tempfile.TemporaryDirectory(prefix="clause-model-") as tmp:
        artifact = Path(tmp) / "bundle.joblib"
        import joblib
        joblib.dump(bundle, artifact)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            for key, value in {
                "model_name": bundle.model_name,
                "seed": bundle.seed,
                "category_count": len(bundle.categories),
            }.items():
                client.log_param(run.info.run_id, key, value)
            for key, value in metrics.items():
                client.log_metric(run.info.run_id, str(key), float(value))
            client.set_tag(run.info.run_id, "metadata_json", json.dumps(metadata, sort_keys=True))
            client.set_tag(run.info.run_id, "category_schema_sha256", _schema_hash(bundle.categories))
            client.log_artifact(run.info.run_id, str(artifact), artifact_path="model")
            client.log_dict(run.info.run_id, {
                "model_name": bundle.model_name,
                "categories": bundle.categories,
                "thresholds": bundle.thresholds,
                "config": bundle.config,
                "seed": bundle.seed,
                "metrics": metrics,
                "metadata": metadata,
            }, "model/manifest.json")
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            client.create_registered_model(registry_name)
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
        version = client.create_model_version(registry_name, model_uri, run.info.run_id)
    loaded = load_bundle(registry_name, version.version, tracking_uri=tracking_uri)
    return RegisteredClauseModel(loaded, run.info.run_id, str(version.version), registry_name)


def load_bundle(
    registry_name: str,
    version: str | int = "latest",
    *,
    tracking_uri: str | Path,
) -> ClauseModelBundle:
    """Load a specific registered version (or latest) without relying on aliases."""
    client = set_local_tracking(tracking_uri)
    versions = client.search_model_versions(f"name='{registry_name}'")
    if not versions:
        raise FileNotFoundError(f"No registered model named {registry_name!r}")
    selected = max(versions, key=lambda item: int(item.version)) if version == "latest" else next(
        (item for item in versions if str(item.version) == str(version)), None,
    )
    if selected is None:
        raise FileNotFoundError(f"No version {version!r} for model {registry_name!r}")
    local_dir = client.download_artifacts(selected.run_id, "model")
    import joblib
    return joblib.load(Path(local_dir) / "bundle.joblib")


def _schema_hash(categories: list[str]) -> str:
    import hashlib
    return hashlib.sha256("\n".join(categories).encode("utf-8")).hexdigest()
