"""
Turns an uploaded file (or pasted text) into the plain text the pipeline reads, remembering where
each PDF page starts so evidence can be reported as "page 7".

How everything comes together:
Document: name, text, page start offsets, and whether OCR produced the text
read_document(): dispatches on the file type (.pdf, .txt, .docx)
pdf_document(): pypdf's text layer; a scan without one goes to OCR if it's installed
ocr_pages(): optional OCR (pypdfium2 renders, pytesseract reads); needs the Tesseract program

Evidence offsets point into Document.text, not into the original file.
"""
import bisect
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

PAGE_SEPARATOR = "\n\n"
SUPPORTED_SUFFIXES = (".pdf", ".txt", ".docx")


@dataclass(frozen=True)
class Document:
    """A contract's text; page_starts[i] is the offset where page i + 1 begins (empty if it has no pages)."""
    name: str
    text: str
    page_starts: tuple[int, ...] = ()
    ocr: bool = False

    def page_of(self, offset: int) -> int | None:
        """1-based page holding text[offset], or None for documents without pages."""
        return bisect.bisect_right(self.page_starts, offset) if self.page_starts else None


def _from_pages(name: str, pages: list[str], ocr: bool = False) -> Document:
    """Join page texts with PAGE_SEPARATOR, recording each page's start offset."""
    starts, offset = [], 0
    for page in pages:
        starts.append(offset)
        offset += len(page) + len(PAGE_SEPARATOR)
    return Document(name, PAGE_SEPARATOR.join(pages), tuple(starts), ocr)


def ocr_pages(path: str | Path) -> list[str]:
    """OCR every page; raises ValueError explaining what to install if OCR isn't available."""
    try:
        import pypdfium2
        import pytesseract
    except ImportError as error:
        raise ValueError("This PDF has no text layer (probably a scan). To read scans, install OCR support: "
                         "pip install pypdfium2 pytesseract, plus the Tesseract program.") from error
    try:
        pdf = pypdfium2.PdfDocument(str(path))
        return [pytesseract.image_to_string(page.render(scale=300 / 72).to_pil()) for page in pdf]
    except pytesseract.TesseractNotFoundError as error:
        raise ValueError("This PDF is a scan and the Tesseract OCR program isn't installed.") from error


def pdf_document(path: str | Path) -> Document:
    """Read a PDF's text layer, falling back to OCR for scans; raises ValueError if no text can be found."""
    name = Path(path).stem
    pages = [page.extract_text() or "" for page in PdfReader(path).pages]
    if any(page.strip() for page in pages):
        return _from_pages(name, pages)
    pages = ocr_pages(path)
    if not any(page.strip() for page in pages):
        raise ValueError("No text found in this PDF, even with OCR.")
    return _from_pages(name, pages, ocr=True)


def read_document(path: str | Path) -> Document:
    """Read a .pdf, .txt or .docx file into a Document; raises ValueError for anything else or an empty file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_document(path)
    if suffix == ".txt":
        document = Document(path.stem, path.read_text(encoding="utf-8", errors="replace"))
    elif suffix == ".docx":
        from docx import Document as WordDocument

        document = Document(path.stem, "\n".join(p.text for p in WordDocument(str(path)).paragraphs))
    else:
        raise ValueError(f"Unsupported file type {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}")
    if not document.text.strip():
        raise ValueError("The file has no text.")
    return document
