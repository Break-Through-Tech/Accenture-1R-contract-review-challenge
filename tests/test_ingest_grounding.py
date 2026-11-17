"""
End-to-end grounding invariant for uploaded PDFs: read the file, run the real sentence-window chunker
and evidence extractor through ContractInferencePipeline, and check that every quote sits at
text[start:end] and that page_of() names the page the quote is really on.
"""
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader
from transformers import AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.evidence.cues import load_category_cues
from src.inference import ContractInferencePipeline
from src.ingest.documents import PAGE_SEPARATOR, Document, read_document
from src.labels.categories import CATEGORIES
from src.registry.bundle import ClauseModelBundle
from tests.pdf_fixtures import write_pdf

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo_contracts"
DEMO_PDFS = sorted(DEMO_DIR.glob("*.pdf"))
# Three pages, the middle one blank, so page numbers after it must not shift
FIXTURE_PAGES = [
    "MASTER SERVICES AGREEMENT\nThis Agreement is signed by the parties named below.\n"
    "Either party may terminate this Agreement without cause by giving thirty days notice.",
    None,
    "Payment is due within thirty days.\nThis Agreement is governed by the laws of the State of Delaware.",
]
GOVERNING_LAW = "This Agreement is governed by the laws of the State of Delaware."
TERMINATION = "Either party may terminate this Agreement without cause by giving thirty days notice."


class EveryCategory:
    """Predicts every category for every chunk, so evidence is extracted everywhere it can be."""
    def predict_proba(self, texts):
        """Probability 1 for all categories."""
        return np.ones((len(texts), len(CATEGORIES)), dtype=np.float32)


@pytest.fixture(scope="module")
def tokenizer():
    """The pinned DistilRoBERTa tokenizer from the local Hugging Face cache (no network)."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
                                         local_files_only=True)


def _pipeline(tokenizer, max_tokens: int, overlap_tokens: int) -> ContractInferencePipeline:
    """The real inference pipeline with an every-category predictor and the real category cues."""
    bundle = ClauseModelBundle(predictor=EveryCategory(), categories=CATEGORIES,
                               thresholds={category: 0.5 for category in CATEGORIES},
                               model_name="every-category", config={}, seed=0)
    return ContractInferencePipeline(bundle, tokenizer, load_category_cues(), max_tokens=max_tokens,
                                     overlap_tokens=overlap_tokens)


def _assert_grounded_on_true_pages(path: Path, document: Document, result: dict) -> None:
    """Every chunk and quote slices the text exactly, and each quote is found on its page as pypdf reads it alone."""
    # Rebuild page offsets independently from pypdf's per-page text, not from document.page_starts
    page_texts = [page.extract_text() or "" for page in PdfReader(path).pages]
    starts = np.cumsum([0] + [len(page) + len(PAGE_SEPARATOR) for page in page_texts[:-1]]).tolist()
    assert all(document.text[c["start"]:c["end"]] == c["text"] for c in result["chunks"])
    assert result["grounded"], "expected some grounded evidence"
    assert not [u for u in result["unknown"] if u.reason == "span_mismatch"]
    for item in result["grounded"]:
        evidence = item.evidence
        assert document.text[evidence.start:evidence.end] == evidence.quote
        page = document.page_of(evidence.start)
        assert page == document.page_of(evidence.end - 1), "a quote must not cross a page break"
        local = evidence.start - starts[page - 1]
        assert page_texts[page - 1][local:local + len(evidence.quote)] == evidence.quote


# ----- Hand-made PDF with a blank page -----
def test_pdf_evidence_is_grounded_and_paged_with_small_chunks(tmp_path, tokenizer):
    """With 48-token chunks over a PDF with a blank page, every quote is exact and on its true page."""
    write_pdf(tmp_path / "msa.pdf", FIXTURE_PAGES)
    document = read_document(tmp_path / "msa.pdf")
    result = _pipeline(tokenizer, max_tokens=48, overlap_tokens=8).predict_contract(document.name, document.text)

    _assert_grounded_on_true_pages(tmp_path / "msa.pdf", document, result)
    pages = {item.evidence.quote: document.page_of(item.evidence.start) for item in result["grounded"]}
    assert pages[TERMINATION] == 1 and pages[GOVERNING_LAW] == 3
    assert len(result["chunks"]) > 1 and document.blank_pages() == (2,)


# ----- Real sample PDFs -----
@pytest.mark.parametrize("path", DEMO_PDFS, ids=[path.stem for path in DEMO_PDFS])
def test_demo_pdf_evidence_is_grounded_and_paged(path, tokenizer):
    """On each sample contract, with the pipeline's default 512/64 windows, every quote is exact and on its page."""
    document = read_document(path)
    result = _pipeline(tokenizer, max_tokens=512, overlap_tokens=64).predict_contract(document.name, document.text)
    _assert_grounded_on_true_pages(path, document, result)


def test_demo_pdfs_are_present():
    """The three sample PDFs described in data/demo_contracts/SOURCES.md are in the repo."""
    assert len(DEMO_PDFS) == 3
