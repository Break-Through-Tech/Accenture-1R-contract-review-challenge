"""Compare four fixed strategies on development contracts only; validation requires a later decision."""

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from transformers import AutoTokenizer

from src.chunking.chunker import chunk_contract
from src.chunking.evaluation import evaluate_chunks
from src.chunking.experiment import annotation_lengths, compare_examples, measure_chunks
from src.chunking.token_chunker import (
    TOKENIZER_NAME, TOKENIZER_REVISION, chunk_sentence_windows, chunk_token_windows,
)
from src.labels.categories import CATEGORIES
from src.labels.loader import load_cuad
from src.labels.projection import project_labels
from src.labels.splits import split_by_contract

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/cuad/train_separate_questions.json"
OUTPUT = ROOT / "results/token_chunking_development"


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a small report table with its dictionary keys as the CSV header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Load only official training data, evaluate development, and save reproducible artifacts."""
    contracts = load_cuad(TRAIN_PATH)
    development, validation = split_by_contract(contracts, val_fraction=0.2, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
        cache_dir=ROOT / ".venv/hf-cache", local_files_only=True,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_chunks = {}
    reports = {}
    summary_rows = []
    category_rows = []
    measurement_rows = []
    # Four explicit strategies, with no boundary tuning against validation annotations.
    for method in ("character_800_200", "character_1500_200", "token_512_64", "sentence_512_64"):
        chunks_by_contract = {}
        fallbacks = {}
        begin = perf_counter()
        for contract in development:
            if method.startswith("character"):
                window = 800 if method == "character_800_200" else 1500
                chunks = chunk_contract(contract.contract_id, contract.text, window, 200)
            elif method == "token_512_64":
                chunks = chunk_token_windows(contract.contract_id, contract.text, tokenizer, 512, 64)
            else:
                cases = []
                chunks = chunk_sentence_windows(contract.contract_id, contract.text, tokenizer, 512, 64, cases)
                fallbacks[contract.contract_id] = cases
            chunks_by_contract[contract.contract_id] = chunks
        generation_seconds = perf_counter() - begin
        all_chunks[method] = chunks_by_contract

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
            "spans": span_report.model_dump(), "chunks": statistics.model_dump(),
            "generation_seconds": generation_seconds, "projection_seconds": projection_seconds,
            "scoring_seconds": scoring_seconds, "positive_label_assignments": positive_labels,
            "fallback_cases": {key: [case.model_dump() for case in values] for key, values in fallbacks.items() if values},
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
        for candidate in ("token_512_64", "sentence_512_64"):
            examples.extend(example.model_dump() for example in compare_examples(
                development, baseline, candidate, all_chunks[baseline], all_chunks[candidate],
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
                  "validation_ids": [c.contract_id for c in validation], "validation_evaluated": False},
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
    write_csv(OUTPUT / "summary.csv", summary_rows)
    write_csv(OUTPUT / "per_category.csv", category_rows)
    write_csv(OUTPUT / "chunks.csv", measurement_rows)
    (OUTPUT / "report.json").write_text(json.dumps({
        "metadata": metadata, "methods": reports,
        "standalone_annotation_input_token_lengths": {key: value.model_dump() for key, value in annotation_lengths(development, tokenizer).items()},
        "comparison_examples": examples,
    }, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Development reports saved to {OUTPUT}. Validation has not been evaluated.", flush=True)


if __name__ == "__main__":
    main()
