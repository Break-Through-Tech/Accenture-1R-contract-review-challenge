"""Compare four fixed strategies on development contracts only; validation requires a later decision."""

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from transformers import AutoTokenizer

from src.chunking.chunker import chunk_contract
from src.chunking.evaluation import evaluate_chunks, validate_chunks
from src.chunking.experiment import annotation_lengths, compare_examples, measure_chunks
from src.chunking.gwen_adapter import GWEN_COMMIT, GWEN_SOURCE_PATH, load_gwen_prototype, saved_development
from src.chunking.token_chunker import (
    TOKENIZER_NAME, TOKENIZER_REVISION, chunk_sentence_windows, chunk_token_windows,
)
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.projection import project_labels

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/cuad/train_separate_questions.json"
OUTPUT = ROOT / "results/token_chunking_development"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a small report table with its dictionary keys as the CSV header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(include_gwen: bool = False) -> None:
    """Load only official training data, evaluate development, and save reproducible artifacts."""
    contracts = load_cuad(TRAIN_PATH)
    saved_path = OUTPUT / "report.json"
    development, saved_metadata = saved_development(contracts, saved_path, TRAIN_PATH)
    output = ROOT / "results/token_chunking_comparison" if include_gwen else OUTPUT
    # Check tokenizer/library/evaluation identity before reusing the saved development split.
    versions = {name: importlib.metadata.version(name) for name in saved_metadata["versions"]}
    if (saved_metadata["tokenizer"] != TOKENIZER_NAME or saved_metadata["tokenizer_revision"] != TOKENIZER_REVISION or
            saved_metadata["versions"] != versions or saved_metadata["categories"] != CATEGORIES or
            saved_metadata["token_length_options"] != {"truncation": False, "padding": False, "add_special_tokens": True} or
            saved_metadata["max_tokens"] != 512 or saved_metadata["overlap_tokens"] != 64 or
            saved_metadata["character_baselines"] != [[800, 200], [1500, 200]]):
        raise ValueError("Saved tokenizer, versions, categories, or evaluation settings do not match")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    gwen_prototype, gwen_sha256 = load_gwen_prototype() if include_gwen else (None, None)
    all_chunks = {}
    reports = {}
    summary_rows = []
    category_rows = []
    measurement_rows = []
    methods = ["character_800_200", "character_1500_200", "token_512_64", "sentence_512_64"]
    if include_gwen:
        methods.append("gwen_480_120")
    # Rerun generation and measurement consistently for every strategy, without validation.
    for method in methods:
        chunks_by_contract = {}
        fallbacks = {}
        begin = perf_counter()
        for contract in development:
            if method.startswith("character"):
                window = 800 if method == "character_800_200" else 1500
                chunks = chunk_contract(contract.contract_id, contract.text, window, 200)
            elif method == "token_512_64":
                chunks = chunk_token_windows(contract.contract_id, contract.text, tokenizer, 512, 64)
            elif method == "gwen_480_120":
                try:
                    chunks = gwen_prototype(contract, tokenizer, window_tokens=480, overlap_tokens=120)
                except Exception as error:
                    # Preserve the source/output behavior and report an audit failure, never repair it.
                    (output / "verification_failures.json").write_text(json.dumps({
                        "source_commit": GWEN_COMMIT, "contract_id": contract.contract_id,
                        "window_tokens": 480, "overlap_tokens": 120,
                        "error": f"{type(error).__name__}: {error}", "implementation_modified": False,
                    }, indent=2) + "\n", encoding="utf-8")
                    raise ValueError(f"Unmodified PR #48 failed audit for {contract.contract_id}; see verification_failures.json") from error
            else:
                cases = []
                chunks = chunk_sentence_windows(contract.contract_id, contract.text, tokenizer, 512, 64, cases)
                fallbacks[contract.contract_id] = cases
            chunks_by_contract[contract.contract_id] = chunks
        generation_seconds = perf_counter() - begin
        all_chunks[method] = chunks_by_contract

        # Audit every method outside generation timing, using the same coverage checks.
        begin = perf_counter()
        for contract in development:
            try:
                validate_chunks(contract, chunks_by_contract[contract.contract_id])
            except ValueError as error:
                (output / "verification_failures.json").write_text(json.dumps({
                    "method": method, "source_commit": GWEN_COMMIT if method == "gwen_480_120" else None,
                    "contract_id": contract.contract_id, "error": str(error),
                    "implementation_modified": False,
                }, indent=2) + "\n", encoding="utf-8")
                raise
        coverage_audit_seconds = perf_counter() - begin

        # Labels reuse the existing any-overlap policy; no label enters chunk selection.
        begin = perf_counter()
        positive_labels = sum(sum(lc.labels) for c in development
                              for lc in project_labels(chunks_by_contract[c.contract_id], c.spans))
        projection_seconds = perf_counter() - begin
        begin = perf_counter()
        character_window = (800 if method == "character_800_200" else 1500) if method.startswith("character") else None
        span_report = evaluate_chunks(development, chunks_by_contract, character_window,
                                      200 if character_window else None)
        statistics, measurements = measure_chunks(development, chunks_by_contract, tokenizer, 512, fallbacks)
        if method in ("token_512_64", "sentence_512_64") and statistics.overflow_chunks:
            raise AssertionError("Experimental final slices exceeded the model budget")
        scoring_seconds = perf_counter() - begin
        reports[method] = {
            "spans": span_report.model_dump(mode="json"), "chunks": statistics.model_dump(mode="json"),
            "generation_seconds": generation_seconds, "projection_seconds": projection_seconds,
            "coverage_audit_seconds": coverage_audit_seconds,
            "scoring_seconds": scoring_seconds, "positive_label_assignments": positive_labels,
            "fallback_cases": {key: [case.model_dump() for case in values] for key, values in fallbacks.items() if values},
            "verification": {"exact_source_slices": True, "full_source_coverage": True,
                             "valid_offsets_indices_and_progress": True, "label_projection_compatible": True,
                             "within_512_tokens": statistics.overflow_chunks == 0},
        }
        m = span_report.metrics
        summary_rows.append({
            "method": method, "raw_annotations": m.raw_annotations, "unique_annotations": m.unique_annotations,
            "duplicate_annotations": m.duplicate_annotations, "contained_annotations": m.contained_annotations,
            "contained_pct": m.contained_pct, "cut_not_contained_annotations": m.cut_not_contained_annotations,
            "cut_not_contained_pct": m.cut_not_contained_pct, "chunk_count": statistics.chunk_count,
            "overflow_pct": statistics.overflow_pct, "duplicated_char_overhead_pct": statistics.duplicated_char_overhead_pct,
            "actual_overlap_content_tokens_mean": statistics.overlap_content_tokens.mean,
            "zero_overlap_pct": statistics.zero_overlap_pct, "fallback_sentences": statistics.fallback_sentences,
            "generation_seconds": generation_seconds, "scoring_seconds": scoring_seconds,
            "coverage_audit_seconds": coverage_audit_seconds, "projection_seconds": projection_seconds,
            "total_seconds": generation_seconds + coverage_audit_seconds + projection_seconds + scoring_seconds,
        })
        category_rows.extend({"method": method, "category": category, **metrics.model_dump()}
                             for category, metrics in span_report.per_category.items())
        measurement_rows.extend({"method": method, **measurement.model_dump()} for measurement in measurements)
        print(f"{method}: contained={m.contained_pct:.2f}%, cut/not contained={m.cut_not_contained_pct:.2f}%, "
              f"overflow={statistics.overflow_pct:.2f}%, chunks={statistics.chunk_count}, "
              f"overhead={statistics.duplicated_char_overhead_pct:.2f}%, "
              f"zero overlap={statistics.zero_overlap_pct:.2f}%, generation={generation_seconds:.2f}s", flush=True)

    # Compare both new strategies against both character references, selecting gains and losses.
    examples = []
    for baseline in ("character_800_200", "character_1500_200"):
        for candidate in methods[2:]:
            examples.extend(example.model_dump() for example in compare_examples(
                development, baseline, candidate, all_chunks[baseline], all_chunks[candidate],
            ))
    if include_gwen:
        for baseline in ("token_512_64", "sentence_512_64"):
            examples.extend(example.model_dump() for example in compare_examples(
                development, baseline, "gwen_480_120", all_chunks[baseline], all_chunks["gwen_480_120"],
            ))
    metadata = {
        "dataset": str(TRAIN_PATH.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "tokenizer": TOKENIZER_NAME, "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_init": tokenizer.init_kwargs,
        "special_token_overhead": tokenizer.num_special_tokens_to_add(pair=False),
        "token_length_options": {"truncation": False, "padding": False, "add_special_tokens": True},
        "max_tokens": 512, "overlap_tokens": 64, "character_baselines": [[800, 200], [1500, 200]],
        "sentence_detector": "nupunkt.sent_spans; nonadaptive; no layout rules",
        "split": {"seed": 42, "val_fraction": 0.2,
                  "development_ids": [c.contract_id for c in development],
                  "validation_ids": saved_metadata["split"]["validation_ids"], "validation_evaluated": False},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("transformers", "tokenizers", "huggingface-hub", "nupunkt", "numpy", "scikit-learn", "pydantic")},
        "categories": CATEGORIES,
        "counting": "Deduplicate contract_id/category/start/end; unique annotations are the percentage denominator.",
        "label_policy": "Existing project_labels: any positive character overlap.",
        "overlap_policy": "At most 64 content tokens. Sentence overlap uses the longest advancing complete trailing-sentence suffix that fits new content. Long sentences use token fallback; zero overlap at transitions.",
        "limitations": ["Earlier exploration covered all 408 training-file contracts; validation is not pristine.",
                        "Containment refers to original chunks, not truncated encoder inputs.",
                        "Realized overlap differs: this compares complete strategies, not sentence boundaries alone.",
                        "Gwen's reported 9% overflow has not been independently verified."],
    }
    metadata["saved_split_report"] = str(saved_path.relative_to(ROOT))
    metadata["timing_policy"] = "Single-process wall clock; same loaded tokenizer. Algorithm generation, coverage audit, label projection, and scoring/token measurement timed separately for all methods; loading/downloads excluded."
    metadata["saved_split_report_sha256"] = hashlib.sha256(saved_path.read_bytes()).hexdigest()
    if include_gwen:
        metadata["gwen_source"] = {"commit": GWEN_COMMIT, "path": GWEN_SOURCE_PATH, "sha256": gwen_sha256,
                                   "window_tokens": 480, "overlap_tokens": 120,
                                   "window_unit": "document content tokens, excluding special tokens",
                                   "implementation_modified": False}
        previous = json.loads(saved_path.read_text())["methods"]
        for method in methods[:4]:
            if reports[method]["spans"] != previous[method]["spans"] or reports[method]["chunks"] != previous[method]["chunks"]:
                raise AssertionError(f"Saved development results changed for {method}; do not reuse silently")
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "per_category.csv", category_rows)
    write_csv(output / "chunks.csv", measurement_rows)
    (output / "report.json").write_text(json.dumps({
        "metadata": metadata, "methods": reports,
        "standalone_annotation_input_token_lengths": {key: value.model_dump() for key, value in annotation_lengths(development, tokenizer).items()},
        "comparison_examples": examples,
    }, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Development reports saved to {output}. Validation has not been evaluated.", flush=True)


if __name__ == "__main__":
    main()
