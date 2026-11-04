"""Load the archived, unchanged PR #48 prototype for reproducible comparison."""

import hashlib
import json
from pathlib import Path
from types import ModuleType
from typing import Callable

from src.chunking.chunk import Chunk
from src.labels.loader import Contract

GWEN_COMMIT = "94d6d3d8b332de91338625bde100259a1717728b"
GWEN_SOURCE_PATH = "experiments/token_chunking.py"
GWEN_SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "experiments/archive/pr48_token_chunking.py"
GWEN_SOURCE_SHA256 = "6be39944ce5ee89293973411e164384bae8865ae872a6347a2790c1c8f5b6379"


def load_gwen_prototype() -> tuple[Callable[..., list[Chunk]], str]:
    """Load the byte-pinned PR #48 source snapshot and return its callable and SHA-256.

    The archived source is unchanged from the PR commit, but runs against current main's
    shared Chunk type. This avoids requiring the closed PR branch or its Git object at
    evaluation time. The comparison tests audit source coverage and schema compatibility.
    """
    source = GWEN_SNAPSHOT_PATH.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != GWEN_SOURCE_SHA256:
        raise ValueError("Archived PR #48 prototype does not match its pinned source hash")
    module = ModuleType("gwen_pr48_pinned")
    exec(compile(source, f"archive:{GWEN_COMMIT}:{GWEN_SOURCE_PATH}", "exec"), module.__dict__)
    return module.token_chunk_contract, digest


def saved_development(
    contracts: list[Contract], saved_report: Path, dataset_path: Path,
) -> tuple[list[Contract], dict]:
    """Verify the saved dataset/split and select its development IDs in their saved order.

    Never recompute or tune the split. Validation IDs are checked for membership
    and disjointness only; their annotations are not evaluated.
    """
    metadata = json.loads(saved_report.read_text(encoding="utf-8"))["metadata"]
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != metadata["dataset_sha256"]:
        raise ValueError("Dataset does not match the saved development evaluation")
    split = metadata["split"]
    development_ids, validation_ids = split["development_ids"], split["validation_ids"]
    by_id = {contract.contract_id: contract for contract in contracts}
    if (len(by_id) != len(contracts) or len(development_ids) != 326 or len(validation_ids) != 82 or
            len(set(development_ids)) != 326 or len(set(validation_ids)) != 82 or
            set(development_ids) & set(validation_ids) or
            set(development_ids + validation_ids) != set(by_id) or
            split["seed"] != 42 or split["val_fraction"] != 0.2 or split["validation_evaluated"]):
        raise ValueError("Saved split identity, membership, or evaluation state is invalid")
    return [by_id[contract_id] for contract_id in development_ids], metadata
