"""Measure how much PDF extraction costs in grounding, on a CUAD contract that was rendered to PDF.

Reads the PDF with the core reader (clausetriage.ingest.documents.read_document), loads the same contract's
original `context` string and gold answer spans from the training file, and counts how many gold
spans can still be found in the extracted text: verbatim, after collapsing whitespace runs to one
space, ignoring whitespace entirely, and ignoring whitespace and curly-vs-straight quotes. The last
two are diagnostics that show where the losses come from; only verbatim matches could be grounded
as evidence without changing the text.

The default PDF is data/demo_contracts/cuad_validation_fuelcell_development_agreement.pdf, built
from a contract in the validation split of train_separate_questions.json (seed 42). test.json is
never read: the script refuses it.

Usage, from the repo root:
    python -m scripts.evaluate_pdf_ingestion
    python -m scripts.evaluate_pdf_ingestion --pdf other.pdf --contract-id "<CUAD title>" --output report.json
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from clausetriage.ingest.documents import read_document
from clausetriage.labels.loader import load_cuad
from clausetriage.labels.splits import split_by_contract

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "data" / "demo_contracts" / "cuad_validation_fuelcell_development_agreement.pdf"
DEFAULT_CUAD = ROOT / "data" / "cuad" / "train_separate_questions.json"
DEFAULT_CONTRACT_ID = "FuelcellEnergyInc_20191106_8-K_EX-10.1_11868007_EX-10.1_Development Agreement"
LEVELS = ("verbatim", "whitespace_normalized", "ignoring_whitespace", "ignoring_whitespace_and_quote_style")
# Curly single and double quotes folded to their straight ASCII forms
_STRAIGHT_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


class MissingSpan(BaseModel):
    """A gold span not found in the extracted text after whitespace normalization."""
    category: str
    preview: str


class FidelityReport(BaseModel):
    """Extraction fidelity of one PDF against its CUAD source text and gold spans."""
    contract_id: str
    pdf: str
    in_validation_split: bool
    original_chars: int
    extracted_chars: int
    pages: int
    text_identical: bool
    whitespace_normalized_identical: bool
    gold_spans: int
    distinct_gold_spans: int
    found: dict[str, int]
    missing_after_whitespace_normalization: list[MissingSpan]


def normalize_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space and trim the ends."""
    return " ".join(text.split())


def remove_whitespace(text: str) -> str:
    """Drop all whitespace."""
    return "".join(text.split())


def straighten_quotes(text: str) -> str:
    """Replace curly single and double quotes with straight ones."""
    return text.translate(_STRAIGHT_QUOTES)


def _transforms() -> dict[str, Callable[[str], str]]:
    """The text transform applied to both sides at each matching level."""
    return {"verbatim": lambda text: text, "whitespace_normalized": normalize_whitespace,
            "ignoring_whitespace": remove_whitespace,
            "ignoring_whitespace_and_quote_style": lambda text: remove_whitespace(straighten_quotes(text))}


def count_recovered(extracted: str, spans: list[str]) -> dict[str, int]:
    """For each level in LEVELS, how many spans occur in extracted once both are transformed the same way."""
    counts = {}
    for level, transform in _transforms().items():
        haystack = transform(extracted)
        counts[level] = sum(transform(span) in haystack for span in spans)
    return counts


def evaluate(pdf: Path, cuad: Path, contract_id: str) -> FidelityReport:
    """Compare the PDF's extracted text with the CUAD contract's context and count recoverable gold spans."""
    if cuad.name == "test.json":
        raise ValueError("Refusing to read test.json; use train_separate_questions.json")

    # Find the contract and whether it is in the seed-42 validation split of the training file
    contracts = load_cuad(cuad)
    matches = [contract for contract in contracts if contract.contract_id == contract_id]
    if not matches:
        raise ValueError(f"Contract {contract_id!r} is not in {cuad.name}")
    contract = matches[0]
    _, validation = split_by_contract(contracts)

    # Extract with the core reader and compare
    document = read_document(pdf)
    spans = [span.text for span in contract.spans]
    found = count_recovered(document.text, spans)
    normalized_text = normalize_whitespace(document.text)
    missing = [MissingSpan(category=span.category, preview=normalize_whitespace(span.text)[:100])
               for span in contract.spans if normalize_whitespace(span.text) not in normalized_text]
    return FidelityReport(
        contract_id=contract_id, pdf=pdf.name,
        in_validation_split=any(item.contract_id == contract_id for item in validation),
        original_chars=len(contract.text), extracted_chars=len(document.text), pages=len(document.page_starts),
        text_identical=document.text == contract.text,
        whitespace_normalized_identical=normalized_text == normalize_whitespace(contract.text),
        gold_spans=len(spans), distinct_gold_spans=len({(span.start, span.end) for span in contract.spans}),
        found=found, missing_after_whitespace_normalization=missing,
    )


def main(argv: list[str] | None = None) -> FidelityReport:
    """Run the comparison, print the report as JSON, and optionally write it to --output."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--cuad", type=Path, default=DEFAULT_CUAD, help="CUAD training JSON (never test.json)")
    parser.add_argument("--contract-id", default=DEFAULT_CONTRACT_ID)
    parser.add_argument("--output", type=Path, help="Also write the JSON report here")
    args = parser.parse_args(argv)

    report = evaluate(args.pdf, args.cuad, args.contract_id)
    payload = json.dumps(report.model_dump(), indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
