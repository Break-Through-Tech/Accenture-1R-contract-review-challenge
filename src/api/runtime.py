"""
Builds a ClauseTriageService from command-line settings: the artifact store, the demo model, the
pinned tokenizer, the category cue words and a weight set. Shared by the server (python -m src.api)
and the dataset loader (python -m scripts.load_dataset_contracts), so both score with the same
pipeline, chunking and model version.

How everything comes together:
add_service_arguments(): the --database, --bundle, --registry-version, --weights and --model-version options
file_model_version(): the model version key of a bundle file, from its model name and SHA-256
load_pipeline(): the ContractInferencePipeline and its model version key
build_service(): opens the store and wraps it with the pipeline

Model version keys: a bundle loaded from the local MLflow registry is "clause-classifier-tfidf:<n>"
(a version number is required, never "latest", so stored rows always name one model); a bundle
file is "<model_name>@sha256:<first 12 hex of the file's SHA-256>", so a locally rebuilt model
never shares rows with the released one. --model-version overrides either.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.api.service import ClauseTriageService
from src.inference import ContractInferencePipeline
from src.risk.schema import DEFAULT_WEIGHTS_PATH, load_weight_set
from src.store.artifact_store import ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
# Under outputs/, which git ignores; kept apart from other tools' artifact_store.sqlite files
DEFAULT_DATABASE = ROOT / "outputs" / "api" / "clausetriage.sqlite"
# The chunking every stored contract uses (the demo model's and scripts/evaluate_triage.py's)
MAX_TOKENS, OVERLAP_TOKENS = 512, 64


def add_service_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options build_service needs to a command-line parser."""
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE,
                        help="artifact store SQLite file (created if missing)")
    parser.add_argument("--bundle", type=Path, help="ClauseModelBundle .joblib to load (only load trusted files)")
    parser.add_argument("--registry-version", help="load this version number from the local MLflow registry")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH,
                        help="weight set JSON new contracts are scored with (the default weight version)")
    parser.add_argument("--model-version", help="model version key to store results under (default: derived)")


def file_model_version(model_name: str, path: Path) -> str:
    """'<model_name>@sha256:<12 hex>' for a bundle file, so different files never share a key."""
    from src.registry.demo_model import file_sha256

    return f"{model_name}@sha256:{file_sha256(path)[:12]}"


def load_pipeline(bundle: Path | None = None, registry_version: str | None = None,
                  weights: Path = DEFAULT_WEIGHTS_PATH) -> tuple[ContractInferencePipeline, str]:
    """
    Load the demo model (a bundle file, a registry version number, or the released model), the
    pinned tokenizer and the weight set, and return the pipeline and its model version key.
    ValueError if registry_version isn't a version number.
    """
    from transformers import AutoTokenizer

    from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
    from src.evidence.cues import load_category_cues
    from src.registry.demo_model import BUNDLE_PATH, REGISTRY_NAME, TOKENIZER_CACHE, load_demo_bundle

    # The model, and the key its stored rows are filed under
    if registry_version is not None:
        if not registry_version.isdigit():
            raise ValueError("--registry-version must be a version number (not 'latest'), so stored results "
                             "always name the exact model")
        model = load_demo_bundle(registry_version=registry_version)
        model_version = f"{REGISTRY_NAME}:{registry_version}"
    else:
        model = load_demo_bundle(bundle)
        model_version = file_model_version(model.model_name, bundle or BUNDLE_PATH)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
                                              cache_dir=TOKENIZER_CACHE)
    pipeline = ContractInferencePipeline(model, tokenizer, load_category_cues(), max_tokens=MAX_TOKENS,
                                         overlap_tokens=OVERLAP_TOKENS, weight_set=load_weight_set(weights))
    return pipeline, model_version


def build_service(args: argparse.Namespace) -> ClauseTriageService:
    """A service over args.database, scoring with the pipeline add_service_arguments' options describe."""
    pipeline, model_version = load_pipeline(args.bundle, args.registry_version, args.weights)
    return ClauseTriageService(ArtifactStore(args.database), pipeline, args.model_version or model_version)
