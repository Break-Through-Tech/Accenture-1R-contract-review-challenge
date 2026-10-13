"""Load PR #48's exact committed prototype without merging or modifying its source."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Callable

from src.chunking.chunk import Chunk
from src.labels.loader import Contract

GWEN_COMMIT = "94d6d3d8b332de91338625bde100259a1717728b"
GWEN_SOURCE_PATH = "experiments/token_chunking.py"
ROOT = Path(__file__).resolve().parents[2]


def load_gwen_prototype() -> tuple[Callable[..., list[Chunk]], str]:
    """Load the reviewed Git blob unchanged and return its callable and source SHA-256.

    The pinned commit must have been fetched. Only the prototype module is loaded;
    its import of the shared Chunk model resolves to our identical implementation.
    No checkout, source rewrite, bytecode file, or branch merge is performed.
    """
    try:
        source = subprocess.check_output(
            ["git", "show", f"{GWEN_COMMIT}:{GWEN_SOURCE_PATH}"], cwd=ROOT, stderr=subprocess.PIPE,
        )
        original_chunk = subprocess.check_output(
            ["git", "show", f"{GWEN_COMMIT}:src/chunking/chunk.py"], cwd=ROOT, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Fetch the pinned PR #48 commit {GWEN_COMMIT} before comparing") from error
    if original_chunk != (ROOT / "src/chunking/chunk.py").read_bytes():
        raise ValueError("PR #48 and this checkout have different Chunk implementations")
    module = ModuleType("gwen_pr48_pinned")
    exec(compile(source, f"git:{GWEN_COMMIT}:{GWEN_SOURCE_PATH}", "exec"), module.__dict__)
    return module.token_chunk_contract, hashlib.sha256(source).hexdigest()


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
