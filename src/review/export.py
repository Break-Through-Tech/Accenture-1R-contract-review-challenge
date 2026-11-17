"""
Export resolved reviewer feedback as labelled chunk examples for training, without leaking held-out
contracts into training.

One example per (chunk, category) with resolved feedback: the chunk's exact text and absolute offsets,
the category, a 0/1 label (accept/missing -> 1, reject -> 0) and provenance (verdicts, reviewers, model
versions, timestamps, feedback ids). Feedback covers one category of a chunk at a time, so examples are
partial labels: to_partial_label_matrix() groups them into the (texts, labels) shape of
src.pipeline.to_model_inputs plus a mask of which labels are known. The other categories of a chunk
are unknown, not 0.

Leakage rule: an example is exported only if its contract's exact text is in a TrainingManifest's
training set and is neither listed nor identical in text to a held-out (evaluation) contract. The
default manifest is the training partition of train_separate_questions.json under the project split
(split_by_contract, val_fraction 0.2, seed 42); the validation partition is its evaluation set.
Anything else, including uploaded documents and any contract from test.json, is excluded and reported
as not_in_training_manifest. The manifest builder refuses test.json and CUADv1.json (which contains
the test contracts), and nothing here reads either file.

Order of exclusion checks per (chunk, category): offset_mismatch (stored chunk or contract text fails
validation), evaluation_contract, not_in_training_manifest, conflict. Each is reported, never guessed.

How everything comes together:
build_training_manifest():   the default manifest from the CUAD training file and the project split
export_training_examples():  resolve stored feedback, apply the leakage rule, validate offsets
write_export(), read_examples(): examples.jsonl plus report.json, byte-identical for the same input
to_partial_label_matrix():   examples -> chunk ids, texts, labels (n, 41) and a known-label mask
check_no_evaluation_overlap(): a training run's last check that no example comes from its evaluation texts
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from src.labels.categories import CATEGORIES, CATEGORY_TO_INDEX
from src.labels.loader import load_cuad
from src.labels.splits import split_by_contract
from src.review.feedback import list_resolutions
from src.review.schema import (
    ExcludedFeedback, FeedbackExport, Resolution, TrainingExample, TrainingManifest,
)
from src.store.artifact_store import ArtifactStore
from src.store.records import ChunkRecord, ContractRecord

# Files that hold the held-out test contracts; a manifest must never be built from them
FORBIDDEN_MANIFEST_SOURCES = frozenset({"test.json", "cuadv1.json"})
EXAMPLES_FILE, REPORT_FILE = "examples.jsonl", "report.json"


def text_sha256(text: str) -> str:
    """SHA-256 of a contract's exact text (UTF-8), the same digest the store keeps."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_training_manifest(train_path: str | Path, *, val_fraction: float = 0.2, seed: int = 42) -> TrainingManifest:
    """
    The training partition of a CUAD training file as a manifest, with its held-out partition as the
    evaluation set (split_by_contract with the given fraction and seed, the project's split by default).
    Raises ValueError for test.json or CUADv1.json, which hold the test contracts.
    """
    path = Path(train_path)
    if path.name.lower() in FORBIDDEN_MANIFEST_SOURCES:
        raise ValueError(f"refusing to build a training manifest from {path.name}: it holds held-out test contracts")
    training, held_out = split_by_contract(load_cuad(path), val_fraction=val_fraction, seed=seed)
    return TrainingManifest(source=path.name, source_sha256=_file_sha256(path), split_method="split_by_contract",
                            seed=seed, val_fraction=val_fraction,
                            training={c.contract_id: text_sha256(c.text) for c in training},
                            evaluation={c.contract_id: text_sha256(c.text) for c in held_out})


def _exclusion(resolution: Resolution, contract: ContractRecord, chunk: ChunkRecord,
               manifest: TrainingManifest) -> tuple[str, str] | None:
    """(reason, detail) when this resolved key must not be exported, else None."""
    # Integrity first: the stored text must still hash to its digest and hold the chunk at its offsets
    digest = text_sha256(contract.text)
    if digest != contract.text_sha256:
        return "offset_mismatch", "stored contract text no longer matches its SHA-256"
    if not 0 <= chunk.start < chunk.end <= len(contract.text) or contract.text[chunk.start:chunk.end] != chunk.text:
        return "offset_mismatch", f"chunk text does not equal contract text[{chunk.start}:{chunk.end}]"

    # Leakage: held-out by id or by content, then anything the manifest doesn't list as training text
    if contract.contract_id in manifest.evaluation or digest in set(manifest.evaluation.values()):
        return "evaluation_contract", "contract is held out for evaluation in the manifest"
    if digest not in set(manifest.training.values()):
        listed = contract.contract_id in manifest.training
        return "not_in_training_manifest", ("contract text differs from the manifest's" if listed
                                            else "contract is not in the manifest's training set")
    if resolution.status == "conflict":
        verdicts = ", ".join(f"{v.reviewer}={v.verdict}" for v in resolution.current)
        return "conflict", f"reviewers disagree: {verdicts}"
    return None


def _example(resolution: Resolution, contract: ContractRecord, chunk: ChunkRecord) -> TrainingExample:
    """A labelled example from a resolved key, with provenance from its current verdicts."""
    return TrainingExample(
        contract_id=contract.contract_id, contract_sha256=contract.text_sha256, chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index, start=chunk.start, end=chunk.end, chunk_text=chunk.text,
        category=resolution.category, category_index=CATEGORY_TO_INDEX[resolution.category],
        label=resolution.label, verdicts=sorted({v.verdict for v in resolution.current}),
        reviewers=[v.reviewer for v in resolution.current],
        model_versions=sorted({v.model_version for v in resolution.current}),
        first_reviewed_at=min(v.timestamp for v in resolution.history),
        last_reviewed_at=max(v.timestamp for v in resolution.current), votes=resolution.current)


def export_training_examples(store: ArtifactStore, manifest: TrainingManifest, *,
                             model_version: str | None = None) -> FeedbackExport:
    """
    Every (chunk, category) with feedback (under model_version, or under any version if None) as a
    labelled example, or as an exclusion with its reason. Deterministic for a given store and manifest:
    examples and exclusions are in (contract_id, chunk_index, category column) order. KeyError if
    model_version is given but was never stored (so a typo can't produce an empty export).
    """
    if model_version is not None:
        store.get_model_version(model_version)
    contracts: dict[str, ContractRecord] = {}
    examples, excluded = [], []
    for resolution in list_resolutions(store, model_version=model_version):
        if resolution.contract_id not in contracts:
            contracts[resolution.contract_id] = store.get_contract(resolution.contract_id)
        contract, chunk = contracts[resolution.contract_id], store.get_chunk(resolution.chunk_id)
        exclusion = _exclusion(resolution, contract, chunk, manifest)
        if exclusion is None:
            examples.append(_example(resolution, contract, chunk))
        else:
            excluded.append((chunk.chunk_index, ExcludedFeedback(
                contract_id=contract.contract_id, chunk_id=chunk.chunk_id, category=resolution.category,
                reason=exclusion[0], detail=exclusion[1], votes=resolution.current)))

    # One fixed order for both lists, so the same store always exports the same bytes
    examples.sort(key=lambda e: (e.contract_id, e.chunk_index, e.category_index))
    excluded.sort(key=lambda pair: (pair[1].contract_id, pair[0], CATEGORY_TO_INDEX[pair[1].category]))
    counts: dict[str, int] = {}
    for _, item in excluded:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return FeedbackExport(
        model_version=model_version, manifest_source=manifest.source, manifest_sha256=manifest.manifest_sha256,
        split_method=manifest.split_method, seed=manifest.seed, val_fraction=manifest.val_fraction,
        example_count=len(examples), positive_count=sum(e.label for e in examples),
        excluded_counts=dict(sorted(counts.items())), examples=examples, excluded=[item for _, item in excluded])


def _json_line(payload: dict) -> str:
    """Canonical JSON (sorted keys, no extra whitespace) for byte-identical output."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_export(export: FeedbackExport, output: str | Path) -> tuple[Path, Path]:
    """
    Write examples.jsonl (one TrainingExample per line) and report.json (everything else, plus the
    SHA-256 of examples.jsonl) to the output folder. Same export, same bytes. Returns both paths.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = "".join(_json_line(example.model_dump(mode="json")) + "\n" for example in export.examples)
    examples_path, report_path = output / EXAMPLES_FILE, output / REPORT_FILE
    examples_path.write_text(lines, encoding="utf-8", newline="\n")
    report = export.model_dump(mode="json", exclude={"examples"})
    report["examples_file"] = EXAMPLES_FILE
    report["examples_sha256"] = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    return examples_path, report_path


def read_examples(path: str | Path) -> list[TrainingExample]:
    """Load examples.jsonl back into TrainingExample models (validated)."""
    with open(path, encoding="utf-8") as handle:
        return [TrainingExample.model_validate_json(line) for line in handle if line.strip()]


def to_partial_label_matrix(examples: Sequence[TrainingExample]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """
    Group examples by chunk into (chunk_ids, texts, labels, mask), chunks in first-seen order.
    labels is int8 (chunks, 41) like src.pipeline.to_model_inputs; mask is bool and True only where
    feedback gave a label, so a loss must ignore the other entries (they are unknown, not negative).
    """
    row_of: dict[str, int] = {}
    texts: list[str] = []
    for example in examples:
        if example.chunk_id not in row_of:
            row_of[example.chunk_id] = len(texts)
            texts.append(example.chunk_text)
    labels = np.zeros((len(texts), len(CATEGORIES)), dtype=np.int8)
    mask = np.zeros((len(texts), len(CATEGORIES)), dtype=bool)
    for example in examples:
        labels[row_of[example.chunk_id], example.category_index] = example.label
        mask[row_of[example.chunk_id], example.category_index] = True
    return list(row_of), texts, labels, mask


def check_no_evaluation_overlap(examples: Iterable[TrainingExample], evaluation_texts: Iterable[str]) -> None:
    """
    Raise ValueError if any example comes from a contract whose exact text is among evaluation_texts.
    Meant for the training run itself, which knows its evaluation contracts (validation partition or,
    at final evaluation, the test file) and so can check what this module never reads.
    """
    held_out = {text_sha256(text) for text in evaluation_texts}
    leaked = sorted({e.contract_id for e in examples if e.contract_sha256 in held_out})
    if leaked:
        raise ValueError(f"{len(leaked)} exported contract(s) are evaluation contracts: {', '.join(leaked[:5])}")
