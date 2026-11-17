"""
Builds clause registers from pipeline output and writes them as CSV and JSON. Pure Python with no
UI dependency, so the batch script, the demo page and a future API endpoint all export the same thing.

How everything comes together:
build_register(): one contract's register from its ContractInferencePipeline.predict_contract result
                  and its ContractScore; re-checks every quote against the text when it is given
build_registers(): ranks several results together (risk.contract.aggregate_contracts) and builds
                   one register per contract, in rank order
register_frame(), index_frame(): registers as tables (one row per clause / one row per contract)
write_csv(), write_json(), read_json(), write_register(): files; JSON reads back into an equal model
register_file_stem(): a safe, deterministic file name for a contract id

Inputs from other modules:
'result' dicts from inference.ContractInferencePipeline.predict_contract (needs contract_id,
model_name, grounded, clause_scores; unknown, seed and contract_score are used when present)
'ContractScore' from risk.contract.aggregate_contracts
'page_starts': page_starts[i] is the offset where page i + 1 begins, as in the UI's Document
"""
import bisect
import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.evidence.extractor import GroundedEvidence, Unknown
from src.register.schema import ContractRegister, RegisterHeader, RegisterRow
from src.risk.contract import ContractScore, aggregate_contracts
from src.risk.schema import SIGNAL_NAMES, ClauseScore, WeightSet

# Column order of a register CSV: contract summary, clause, signals, location, evidence, provenance
CSV_COLUMNS = [
    "contract_id", "contract_rank", "triage_score", "category", "rating", "composite_score", *SIGNAL_NAMES,
    "probability", "chunk_index", "page", "start", "end", "quote", "has_evidence", "needs_manual_review",
    "review_reason", "model_name", "weight_version_id", "weight_set_sha256",
]
INDEX_COLUMNS = [
    "rank", "contract_id", "triage_score", "high_count", "medium_count", "low_count", "unknown_count",
    "clause_count", "manual_review_count", "file_stem",
]
# Whole-number columns that can be blank; pandas would otherwise write 2 as "2.0" next to a blank
_NULLABLE_INT_COLUMNS = ["contract_rank", "chunk_index", "page", "start", "end"]
_UNSAFE_FILE_CHARS = re.compile(r"[^A-Za-z0-9]+")
MAX_STEM_LENGTH = 60


class GroundingError(ValueError):
    """A row's quote isn't at text[start:end] of the text supplied: the text doesn't match the scored one."""


def check_page_starts(page_starts: Sequence[int]) -> tuple[int, ...]:
    """Return page_starts as a tuple; raises ValueError unless it starts at 0 and strictly increases."""
    starts = tuple(page_starts)
    if starts and starts[0] != 0:
        raise ValueError(f"page_starts must begin at offset 0, got {starts[0]}")
    if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
        raise ValueError("page_starts must be strictly increasing")
    return starts


def page_of(offset: int, page_starts: Sequence[int]) -> int | None:
    """1-based page holding text[offset], or None when there are no page starts (plain text)."""
    return bisect.bisect_right(page_starts, offset) if page_starts else None


def _as_model(model: type, item: Any) -> Any:
    """Accept either a model instance or its dict form (e.g. a result read back from JSON)."""
    return item if isinstance(item, model) else model.model_validate(item)


def _row(clause: ClauseScore, grounded: GroundedEvidence | None, reason: str | None, model_name: str,
         text: str | None, page_starts: tuple[int, ...]) -> RegisterRow:
    """One register row; a clause flagged has_evidence needs its grounded record, else it is a review row."""
    fields = dict(
        contract_id=clause.contract_id, category=clause.category, rating=clause.rating,
        composite_score=clause.composite_score, signals={name: clause.signals[name] for name in SIGNAL_NAMES},
        probability=clause.probability, chunk_index=clause.chunk_index, has_evidence=clause.has_evidence,
        model_name=model_name, weight_version_id=clause.weight_version_id,
        weight_set_sha256=clause.weight_set_sha256,
    )
    if not clause.has_evidence:
        return RegisterRow(**fields, needs_manual_review=True, review_reason=reason or "no_evidence")

    # Grounded: the quote must exist, and (when the text is given) sit exactly at [start:end]
    if grounded is None:
        raise ValueError(f"{clause.category!r} in chunk {clause.chunk_index} of {clause.contract_id!r} is "
                         "flagged has_evidence but the result has no grounded evidence for it")
    evidence = grounded.evidence
    if text is not None and not evidence.validate_against(text):
        raise GroundingError(f"quote for {clause.category!r} in {clause.contract_id!r} is not at "
                             f"text[{evidence.start}:{evidence.end}]")
    return RegisterRow(**fields, needs_manual_review=False, quote=evidence.quote, start=evidence.start,
                       end=evidence.end, page=page_of(evidence.start, page_starts))


def build_register(
    result: Mapping[str, Any],
    contract_score: ContractScore | None = None,
    *,
    text: str | None = None,
    page_starts: Sequence[int] = (),
    ranked_out_of: int | None = None,
) -> ContractRegister:
    """
    Build one contract's register, rows riskiest first (ties by category, then chunk).

    result: a predict_contract result for this contract
    contract_score: its triage score and rank; defaults to result["contract_score"] (rank 1 on its own)
    text: the contract text that was scored; when given, every quote is re-checked against it and a
          mismatch raises GroundingError (the text is not the one the evidence came from)
    page_starts: page start offsets for page numbers; empty means the text has no pages
    ranked_out_of: how many contracts contract_score was ranked among, if known
    Raises ValueError if the score belongs to another contract or weight version.
    """
    contract_id = result["contract_id"]
    score = _as_model(ContractScore, contract_score if contract_score is not None else result["contract_score"])
    starts = check_page_starts(page_starts)
    clauses = [_as_model(ClauseScore, clause) for clause in result["clause_scores"]]

    # Every clause must come from this contract and the weight version that produced the score
    version = (score.weight_version_id, score.weight_set_sha256)
    if score.contract_id != contract_id:
        raise ValueError(f"contract_score is for {score.contract_id!r}, not {contract_id!r}")
    for clause in clauses:
        if clause.contract_id != contract_id or (clause.weight_version_id, clause.weight_set_sha256) != version:
            raise ValueError(f"clause {clause.category!r} doesn't belong to {contract_id!r} under weight "
                             f"version {score.weight_version_id!r}")

    # Look up each clause's evidence (or why it has none) by (chunk_index, category)
    grounded = {(g.chunk_index, g.category): g
                for g in (_as_model(GroundedEvidence, item) for item in result["grounded"])}
    reasons = {(u.chunk_index, u.category): u.reason
               for u in (_as_model(Unknown, item) for item in result.get("unknown", []))}
    rows = [_row(clause, grounded.get((clause.chunk_index, clause.category)),
                 reasons.get((clause.chunk_index, clause.category)), result["model_name"], text, starts)
            for clause in clauses]
    rows.sort(key=lambda row: (-row.composite_score, row.category, row.chunk_index))

    header = RegisterHeader(
        contract_id=contract_id, triage_score=score.triage_score, rank=score.rank, ranked_out_of=ranked_out_of,
        high_count=score.high_count, medium_count=score.medium_count, low_count=score.low_count,
        unknown_count=score.unknown_count, clause_count=len(rows),
        manual_review_count=sum(row.needs_manual_review for row in rows), model_name=result["model_name"],
        model_seed=result.get("seed"), weight_version_id=score.weight_version_id,
        weight_set_sha256=score.weight_set_sha256,
    )
    return ContractRegister(header=header, rows=rows)


def build_registers(
    results: Sequence[Mapping[str, Any]],
    weight_set: WeightSet,
    *,
    texts: Mapping[str, str] | None = None,
    page_starts: Mapping[str, Sequence[int]] | None = None,
) -> list[ContractRegister]:
    """
    Rank several contracts together and build one register each, rank 1 (open first) first.
    Every result's clause_scores must come from weight_set. texts and page_starts are keyed by
    contract_id and optional per contract. Raises ValueError on a repeated contract_id.
    """
    by_id = {result["contract_id"]: result for result in results}
    if len(by_id) != len(results):
        raise ValueError("contract_id values must be unique")
    ranked = aggregate_contracts([s for result in results for s in result["clause_scores"]], weight_set, by_id)
    texts, page_starts = texts or {}, page_starts or {}
    return [build_register(by_id[score.contract_id], score, text=texts.get(score.contract_id),
                           page_starts=page_starts.get(score.contract_id, ()), ranked_out_of=len(ranked))
            for score in ranked]


def register_frame(registers: Iterable[ContractRegister]) -> pd.DataFrame:
    """Every row of the given registers as one table with CSV_COLUMNS, in the order given."""
    records = [{
        "contract_id": row.contract_id, "contract_rank": register.header.rank,
        "triage_score": register.header.triage_score, **row.model_dump(exclude={"contract_id", "signals"}),
        **row.signals,
    } for register in registers for row in register.rows]
    frame = pd.DataFrame(records, columns=CSV_COLUMNS)
    return frame.astype({column: "Int64" for column in _NULLABLE_INT_COLUMNS})


def index_frame(registers: Iterable[ContractRegister]) -> pd.DataFrame:
    """One row per contract with its rank, triage score, counts and register file stem, in the order given."""
    return pd.DataFrame([{
        **register.header.model_dump(include=set(INDEX_COLUMNS)),
        "file_stem": register_file_stem(register.header.contract_id),
    } for register in registers], columns=INDEX_COLUMNS)


def register_file_stem(contract_id: str) -> str:
    """
    A file-name-safe, deterministic stem for a contract id: its letters and digits (accents folded to
    ASCII, other runs of characters turned into '-', cut to MAX_STEM_LENGTH) plus the first 8 hex
    characters of the id's SHA-256, so ids that differ only in punctuation or case never collide.
    """
    ascii_id = unicodedata.normalize("NFKD", contract_id).encode("ascii", "ignore").decode("ascii")
    slug = _UNSAFE_FILE_CHARS.sub("-", ascii_id).strip("-")[:MAX_STEM_LENGTH].rstrip("-") or "contract"
    return f"{slug}-{hashlib.sha256(contract_id.encode('utf-8')).hexdigest()[:8]}"


def write_csv(register: ContractRegister, path: str | Path) -> Path:
    """Write the register's rows as UTF-8 CSV with '\\n' line endings; blank cells mean 'none'."""
    path = Path(path)
    register_frame([register]).to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def to_json(register: ContractRegister) -> str:
    """The register as indented JSON (header and rows); ContractRegister.model_validate_json reads it back."""
    return register.model_dump_json(indent=2) + "\n"


def write_json(register: ContractRegister, path: str | Path) -> Path:
    """Write to_json(register) to path as UTF-8."""
    path = Path(path)
    path.write_text(to_json(register), encoding="utf-8", newline="\n")
    return path


def read_json(path: str | Path) -> ContractRegister:
    """Read and validate a register JSON file; raises pydantic.ValidationError if it is invalid."""
    return ContractRegister.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_register(register: ContractRegister, directory: str | Path) -> tuple[Path, Path]:
    """Write <stem>.csv and <stem>.json into directory (created if missing); returns (csv, json) paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = register_file_stem(register.header.contract_id)
    return write_csv(register, directory / f"{stem}.csv"), write_json(register, directory / f"{stem}.json")
