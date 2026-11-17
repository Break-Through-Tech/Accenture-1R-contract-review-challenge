"""
Score the validation contracts of the CUAD training file with the demo model and write a risk-scored
clause register for each one (CSV and JSON), plus an index that ranks the contracts for review.
Reads only train_separate_questions.json (test.json is never touched).

Run from the repo root:
    python -m scripts.export_registers
    python -m scripts.export_registers --limit 5 --output outputs/registers_try
    python -m scripts.export_registers --weights path/to/weights.json --bundle path/to/bundle.joblib

The contracts are the 20% validation split that scripts/evaluate_triage.py ranks (split_by_contract,
seed 42), chunked the same way (chunk_sentence_windows 512 / 64). The demo model never trained on
them: build_demo_bundle() trains on the other 80%, and its thresholds were tuned on a calibration
slice of that 80%. The model comes from src/registry/demo_model.py (downloaded, or built locally
into models/ in a few minutes if the download fails).

Writes to --output (default outputs/registers/, which git ignores):
    <stem>.csv and <stem>.json: one register per contract; <stem> comes from register_file_stem()
    index.csv: one row per contract in rank order (rank 1 = open first), with its counts and file stem
    index.json: the same rows plus the model, weight version, split and chunking settings
Files from an earlier run with other contracts are not deleted; index.* lists the current run's files.
"""
import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from scripts.evaluate_triage import MAX_TOKENS, OVERLAP_TOKENS, SEED, TRAIN_PATH, VAL_FRACTION
from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.evidence.cues import load_category_cues
from src.inference import ContractInferencePipeline
from src.labels.loader import Contract, load_cuad
from src.labels.splits import split_by_contract
from src.register.export import build_registers, index_frame, register_file_stem, write_register
from src.register.schema import ContractRegister
from src.registry.demo_model import TOKENIZER_CACHE, load_demo_bundle
from src.risk.schema import DEFAULT_WEIGHTS_PATH, load_weight_set

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "registers"


def validation_contracts(train_path: Path = TRAIN_PATH, limit: int | None = None) -> list[Contract]:
    """The validation contracts evaluate_triage.py ranks (same seed and fraction), the first `limit` if given."""
    _, validation = split_by_contract(load_cuad(train_path), val_fraction=VAL_FRACTION, seed=SEED)
    return validation[:limit] if limit is not None else validation


def score_contracts(pipeline: Any, contracts: Sequence[Contract],
                    progress: Callable[[str], None] = print) -> list[dict]:
    """Run every contract through pipeline.predict_contract, reporting progress per contract."""
    results = []
    for number, contract in enumerate(contracts, start=1):
        progress(f"[{number}/{len(contracts)}] {contract.contract_id}")
        results.append(pipeline.predict_contract(contract.contract_id, contract.text))
    return results


def write_registers(registers: Sequence[ContractRegister], output: Path, metadata: dict) -> pd.DataFrame:
    """
    Write each register as <stem>.csv and <stem>.json, then index.csv and index.json (metadata plus
    the index rows). Returns the index. Raises ValueError if two contracts would share a file stem.
    """
    stems = [register_file_stem(register.header.contract_id) for register in registers]
    if len(set(stems)) != len(stems):
        raise ValueError("two contract ids map to the same file stem; refusing to overwrite a register")

    # One register per contract, then the ranking that points at them
    output.mkdir(parents=True, exist_ok=True)
    for register in registers:
        write_register(register, output)
    index = index_frame(registers)
    index.to_csv(output / "index.csv", index=False, encoding="utf-8", lineterminator="\n")
    summary = {**metadata, "contracts": len(registers),
               "clause_rows": int(index["clause_count"].sum()),
               "manual_review_rows": int(index["manual_review_count"].sum()),
               "index": index.to_dict("records")}
    (output / "index.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8", newline="\n")
    return index


def _shown(path: Path) -> str:
    """A path for the index: repo-relative when inside the repo, so the file doesn't record a home folder."""
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="folder for registers and index")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH, help="weight set JSON to score with")
    parser.add_argument("--bundle", type=Path, help="ClauseModelBundle .joblib to load (trusted files only)")
    parser.add_argument("--limit", type=int, help="only the first N validation contracts (for a quick run)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Score the validation contracts, rank them together, and write every register and the index."""
    # Load the weights, the model (downloaded or built on first use) and the pinned tokenizer
    args = parse_args(argv)
    started = time.perf_counter()
    weight_set = load_weight_set(args.weights)
    bundle = load_demo_bundle(args.bundle)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
                                              cache_dir=TOKENIZER_CACHE)
    pipeline = ContractInferencePipeline(bundle, tokenizer, load_category_cues(), max_tokens=MAX_TOKENS,
                                         overlap_tokens=OVERLAP_TOKENS, weight_set=weight_set)

    # Score, rank all contracts together, and re-check every quote against its contract's text
    contracts = validation_contracts(limit=args.limit)
    results = score_contracts(pipeline, contracts)
    registers = build_registers(results, weight_set, texts={c.contract_id: c.text for c in contracts})

    metadata = {
        "model_name": bundle.model_name, "model_seed": bundle.seed,
        "weights": _shown(args.weights), "weight_version_id": weight_set.weight_version_id,
        "weight_set_sha256": weight_set.content_sha256,
        "split": {"source": _shown(TRAIN_PATH), "method": "split_by_contract", "seed": SEED,
                  "val_fraction": VAL_FRACTION, "limit": args.limit},
        "chunking": {"method": "chunk_sentence_windows", "tokenizer": TOKENIZER_NAME,
                     "tokenizer_revision": TOKENIZER_REVISION, "max_tokens": MAX_TOKENS,
                     "overlap_tokens": OVERLAP_TOKENS},
        "test_data_used": False,
    }
    index = write_registers(registers, args.output, metadata)
    print(index.head(10).to_string(index=False))
    print(f"Wrote {len(registers)} registers ({int(index['clause_count'].sum())} clause rows, "
          f"{int(index['manual_review_count'].sum())} needing manual review) to {args.output} "
          f"in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
