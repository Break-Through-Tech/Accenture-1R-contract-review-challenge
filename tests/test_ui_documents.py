import sys

import pytest

from src.ui.documents import PAGE_SEPARATOR, Document, read_document
from tests.pdf_fixtures import write_pdf


# ----- PDFs -----
def test_read_document_pdf_records_page_starts(tmp_path):
    """Two PDF pages are joined by PAGE_SEPARATOR and each offset maps back to its page."""
    write_pdf(tmp_path / "contract.pdf", ["Governed by Delaware law.", "Either party may terminate."])
    document = read_document(tmp_path / "contract.pdf")

    assert document.name == "contract" and not document.ocr
    second = document.text.index("Either party")
    assert document.page_starts == (0, second)
    assert document.text[:second].endswith(PAGE_SEPARATOR)
    assert document.page_of(0) == 1 and document.page_of(second - 1) == 1 and document.page_of(second) == 2


def test_read_document_scan_without_ocr_explains_what_to_install(tmp_path, monkeypatch):
    """A PDF with no text layer, and no OCR libraries, raises a ValueError naming the install."""
    monkeypatch.setitem(sys.modules, "pytesseract", None)  # makes `import pytesseract` fail
    write_pdf(tmp_path / "scan.pdf", [None])
    with pytest.raises(ValueError, match="install OCR support"):
        read_document(tmp_path / "scan.pdf")


def test_read_document_scan_uses_ocr_when_available(tmp_path, monkeypatch):
    """A scan's pages come from ocr_pages and the Document is marked as OCR'd."""
    import src.ui.documents as documents

    monkeypatch.setattr(documents, "ocr_pages", lambda path: ["Scanned page one.", "Scanned page two."])
    write_pdf(tmp_path / "scan.pdf", [None, None])
    document = read_document(tmp_path / "scan.pdf")
    assert document.ocr and document.text == "Scanned page one.\n\nScanned page two."
    assert document.page_of(document.text.index("two")) == 2


# ----- Other formats -----
def test_read_document_txt_and_docx_have_no_pages(tmp_path):
    """Text and Word files are read whole, with no page numbers."""
    from docx import Document as WordDocument

    (tmp_path / "a.txt").write_text("Governed by Delaware law.", encoding="utf-8")
    word = WordDocument()
    word.add_paragraph("Governed by Delaware law.")
    word.add_paragraph("Either party may terminate.")
    word.save(tmp_path / "b.docx")

    assert read_document(tmp_path / "a.txt") == Document("a", "Governed by Delaware law.")
    docx = read_document(tmp_path / "b.docx")
    assert docx.text == "Governed by Delaware law.\nEither party may terminate." and docx.page_of(5) is None


@pytest.mark.parametrize("name,content,match", [("a.rtf", "text", "Unsupported"), ("empty.txt", "  ", "no text")])
def test_read_document_rejects_unsupported_or_empty(tmp_path, name, content, match):
    """Unknown file types and empty files raise ValueError."""
    (tmp_path / name).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_document(tmp_path / name)
