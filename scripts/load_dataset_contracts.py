"""
Score the CUAD development (validation) contracts into the API's artifact store, so reviewers can
pick them in the API and the page next to their own uploads. Reads only
train_separate_questions.json (test.json is never opened).

Run from the repo root, with the server stopped (one process per database):
    python -m scripts.load_dataset_contracts                 # all 82 validation contracts
    python -m scripts.load_dataset_contracts --limit 5       # the first 5
    python -m scripts.load_dataset_contracts --database outputs/api/try.sqlite --weights path/to/weights.json

The contracts are the ones scripts/export_registers.py writes registers for: the 20% validation
split of scripts/evaluate_triage.py (split_by_contract, seed 42), which the demo model never
trained on. Each is stored under its CUAD title as contract_id, with source_type "cuad", and goes
through the same service, pipeline and model version as an upload. Contracts already stored are
skipped, so the script can be rerun.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Callable, Sequence

from scripts.evaluate_triage import MAX_TOKENS, OVERLAP_TOKENS
from scripts.export_registers import validation_contracts
from src.api.runtime import MAX_TOKENS as SERVER_MAX_TOKENS
from src.api.runtime import OVERLAP_TOKENS as SERVER_OVERLAP_TOKENS
from src.api.runtime import add_service_arguments
from src.api.service import ClauseTriageService
from src.ingest.documents import Document
from src.labels.loader import Contract

SOURCE_TYPE = "cuad"


def load_contracts(service: ClauseTriageService, contracts: Sequence[Contract],
                   progress: Callable[[str], None] = print) -> Counter:
    """
    Submit and score each contract through the service, in order. Returns how many ended in each
    job status ("complete" or "failed"; already-stored contracts count as "skipped").
    """
    outcomes: Counter = Counter()
    for number, contract in enumerate(contracts, start=1):
        document = Document(name=contract.contract_id, text=contract.text, source_type="pasted")
        job, queued = service.submit_document(document, contract_id=contract.contract_id,
                                              source_name=contract.contract_id, source_type=SOURCE_TYPE)
        if queued:
            job = service.run_job(contract.contract_id)
        status = job.status if queued else "skipped"
        outcomes[status] += 1
        reason = f": {job.error}" if job.status == "failed" else ""
        progress(f"[{number}/{len(contracts)}] {status} {contract.contract_id}{reason}")
    return outcomes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, help="only the first N validation contracts (for a quick run)")
    add_service_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the model and store, then score the validation contracts into the store."""
    from src.api.runtime import build_service

    # The server's chunking must be the one the split's evaluation used, or the contracts aren't comparable
    if (MAX_TOKENS, OVERLAP_TOKENS) != (SERVER_MAX_TOKENS, SERVER_OVERLAP_TOKENS):
        raise SystemExit("scripts/evaluate_triage.py and src/api/runtime.py chunk differently")
    args = parse_args(argv)
    started = time.perf_counter()
    service = build_service(args)
    outcomes = load_contracts(service, validation_contracts(limit=args.limit))
    print(f"Done in {time.perf_counter() - started:.0f}s: {dict(outcomes)}; model {service.model_version}, "
          f"weights {service.default_weight_set.weight_version_id}, database {args.database}")


if __name__ == "__main__":
    main()
