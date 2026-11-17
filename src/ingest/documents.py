"""
Document ingestion: turns an uploaded contract (PDF, Word or plain text) or pasted text into the one
text string every later stage indexes into.

One text source: chunk offsets, evidence offsets and page numbers all point into Document.text, and
evidence is accepted only when text[start:end] == quote. So extraction is deterministic for a given
file and library version, Document is frozen, and its sha256 pins the exact text: a Document whose
text no longer matches its hash cannot be built.

How everything comes together:
Document: frozen Pydantic model (name, text, page_starts, ocr, source_type, sha256) with page_of()
read_document(): reads a file from disk (size-checked first) and hands the bytes to document_from_bytes()
document_from_bytes(): checks the bytes match the extension, then runs the PDF, Word or text reader
pasted_document(): wraps pasted text without changing it
ocr_pages(): optional OCR for scans (pypdfium2 renders, pytesseract reads); needs the Tesseract program

Every file is untrusted input: size, page and text limits apply before and during parsing, and any
parser failure becomes a ValueError whose message a reviewer can act on (the parser's own exception is
kept as __cause__ for debugging). See docs/document-ingestion.md.
"""
import bisect
import hashlib
import io
import itertools
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from pypdf import PdfReader
from typing_extensions import Self

PAGE_SEPARATOR = "\n\n"
CELL_SEPARATOR = " | "
SUPPORTED_SUFFIXES = (".pdf", ".txt", ".docx")
SourceType = Literal["pdf", "docx", "txt", "pasted"]

# Limits for untrusted input (MB means 1024 * 1024 bytes)
MB = 1024 * 1024
MAX_FILE_BYTES = 25 * MB
MAX_PAGES = 1_000
MAX_TEXT_CHARS = 2_000_000
MAX_DOCX_UNCOMPRESSED_BYTES = 100 * MB
MAX_DOCX_ENTRIES = 5_000

# Messages shown to reviewers
EMPTY_FILE = "The file is empty (0 bytes). Check that it was saved completely, then upload it again."
NO_TEXT = "The file has no text."
DAMAGED_PDF = ("This PDF is damaged or incomplete and can't be read. Open it in a PDF viewer, save or print it "
               "to a new PDF, and upload that.")
LOCKED_PDF = ("This PDF is password-protected. Open it with the password, save an unprotected copy (or print "
              "it to a new PDF), and upload that.")
DAMAGED_DOCX = ("This Word file is damaged or isn't really a .docx document. Open it in Word, save it again "
                "as .docx, and upload that.")
LOCKED_DOCX = ("This Word file is password-protected, or is an old .doc file renamed to .docx. Open it in Word, "
               "remove the password or save it as .docx, and upload that.")

# What a file's first bytes say it is, for "wrong extension" messages
_LOOKS_LIKE = {"pdf": "a PDF", "zip": "a ZIP-based Office file (such as .docx or .xlsx)",
               "ole": "an old-format or password-protected Office file", "rtf": "an RTF file"}
_POSITIONAL = ("name", "text", "page_starts", "ocr")
# WordprocessingML namespace, as lxml writes it in element tags
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def text_sha256(text: str) -> str:
    """Hex SHA-256 of text encoded as UTF-8 (lone surrogates passed through, so any str hashes)."""
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


class Document(BaseModel):
    """
    One contract's text, the single source every offset in the pipeline points into.

    page_starts[i] is the offset where page i + 1 begins (empty for formats without pages); a blank
    trailing page may start at len(text). sha256 is filled from text when omitted and must match it.
    For backward compatibility the first four fields may be passed positionally:
    Document(name, text, page_starts, ocr).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    text: str
    page_starts: tuple[int, ...] = ()
    ocr: bool = False
    source_type: SourceType = "pasted"
    sha256: str

    def __init__(self, *args: Any, **data: Any) -> None:
        """Accept (name, text, page_starts, ocr) positionally, as the old dataclass did, or as keywords."""
        if len(args) > len(_POSITIONAL):
            raise TypeError(f"Document takes at most {len(_POSITIONAL)} positional arguments")
        for field, value in zip(_POSITIONAL, args):
            if field in data:
                raise TypeError(f"Document got multiple values for {field!r}")
            data[field] = value
        super().__init__(**data)

    # Fill the hash from the text when the caller didn't give one
    @model_validator(mode="before")
    @classmethod
    def _fill_sha256(cls, data: Any) -> Any:
        """Compute sha256 from text if it is missing."""
        if isinstance(data, dict) and "sha256" not in data and isinstance(data.get("text"), str):
            return {**data, "sha256": text_sha256(data["text"])}
        return data

    # The hash must match the text, and page starts must index into it in order
    @model_validator(mode="after")
    def _check_hash_and_pages(self) -> Self:
        """Reject a hash that doesn't match the text, or page_starts not sorted from 0 within the text."""
        if self.sha256 != text_sha256(self.text):
            raise ValueError("sha256 does not match text: the text was changed after extraction")
        starts = self.page_starts
        if starts and starts[0] != 0:
            raise ValueError("page_starts must begin at 0")
        if any(later <= earlier for earlier, later in itertools.pairwise(starts)):
            raise ValueError("page_starts must be strictly increasing")
        if starts and starts[-1] > len(self.text):
            raise ValueError("page_starts must fall inside the text")
        return self

    def page_of(self, offset: int) -> int | None:
        """1-based page holding text[offset], or None for documents without pages; ValueError outside the text."""
        if not 0 <= offset < len(self.text):
            raise ValueError(f"Offset {offset} is outside the text (length {len(self.text)})")
        return bisect.bisect_right(self.page_starts, offset) if self.page_starts else None

    def blank_pages(self) -> tuple[int, ...]:
        """1-based numbers of pages with no text (whitespace only), such as scanned or intentionally blank pages."""
        ends = (*self.page_starts[1:], len(self.text))
        return tuple(number for number, (start, end) in enumerate(zip(self.page_starts, ends), start=1)
                     if not self.text[start:end].strip())


# ----- Limits and file-type checks -----
def _check_file_size(size: int) -> None:
    """Reject empty files and files over MAX_FILE_BYTES."""
    if size == 0:
        raise ValueError(EMPTY_FILE)
    if size > MAX_FILE_BYTES:
        raise ValueError(f"This file is {size / MB:.1f} MB, over the {MAX_FILE_BYTES // MB} MB limit. "
                         "Split it into smaller files or remove large images, then upload again.")


def _check_text_length(length: int) -> None:
    """Reject documents whose text is longer than MAX_TEXT_CHARS."""
    if length > MAX_TEXT_CHARS:
        raise ValueError(f"This document has more than {MAX_TEXT_CHARS:,} characters of text, over the limit. "
                         "Split it into smaller files and upload them separately.")


def _check_suffix(suffix: str) -> None:
    """Reject extensions other than SUPPORTED_SUFFIXES, with a hint for old Word files."""
    if suffix in SUPPORTED_SUFFIXES:
        return
    if suffix == ".doc":
        raise ValueError("Old Word .doc files aren't supported. Open it in Word, save it as .docx or PDF, "
                         "and upload that.")
    kind = f"type {suffix!r}" if suffix else "without an extension"
    raise ValueError(f"Can't read a file {kind}; upload a {', '.join(SUPPORTED_SUFFIXES)} file.")


def _sniff(data: bytes) -> str | None:
    """What the first bytes say the file is: 'pdf', 'zip', 'ole', 'rtf', or None for anything else."""
    head = data[:1024]
    if b"%PDF-" in head:
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "ole"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    return None


def _check_content(suffix: str, data: bytes) -> None:
    """Reject files whose contents don't match their extension, saying what they look like instead."""
    kind = _sniff(data)
    looks = f"; it looks like {_LOOKS_LIKE[kind]}" if kind else ""
    if suffix == ".pdf" and kind != "pdf":
        raise ValueError(f"This file is named .pdf but isn't a PDF{looks}. Check the file, or rename it with "
                         "the right extension.")
    if suffix == ".docx" and kind == "ole":
        raise ValueError(LOCKED_DOCX)
    if suffix == ".docx" and kind != "zip":
        raise ValueError(f"This file is named .docx but isn't a Word document{looks}. Check the file, or rename "
                         "it with the right extension.")
    utf16 = data.startswith((b"\xff\xfe", b"\xfe\xff"))
    if suffix == ".txt" and (kind is not None or (b"\x00" in data[:8192] and not utf16)):
        raise ValueError(f"This file is named .txt but isn't plain text{looks}. Check the file, or rename it "
                         "with the right extension.")


# ----- Entry points -----
def read_document(path: str | Path) -> Document:
    """Read a .pdf, .docx or .txt file into a Document; raises ValueError a reviewer can act on otherwise."""
    path = Path(path)
    _check_suffix(path.suffix.lower())

    # Check the size before reading anything into memory
    unreadable = f"Can't open {path.name!r}: the file is missing or can't be read."
    try:
        if not path.is_file():
            raise ValueError(unreadable)
        _check_file_size(path.stat().st_size)
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(unreadable) from error
    return document_from_bytes(data, path.name)


def document_from_bytes(data: bytes, filename: str) -> Document:
    """
    Read uploaded bytes into a Document, using filename's extension for the type and its stem as the name.
    Checks size and that the contents match the extension before any parser sees the bytes.
    """
    path = Path(filename)
    suffix = path.suffix.lower()
    _check_suffix(suffix)
    _check_file_size(len(data))
    _check_content(suffix, data)
    name = path.stem.strip() or "Untitled"
    readers = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt}
    return readers[suffix](name, data)


def pasted_document(text: str, name: str = "Pasted text") -> Document:
    """Wrap pasted text, unchanged, as a Document without pages; ValueError if empty or too long."""
    if not isinstance(text, str):
        raise TypeError("Pasted text must be a string")
    if not text.strip():
        raise ValueError("The pasted text is empty.")
    _check_text_length(len(text))
    return Document(name=name, text=text, source_type="pasted")


def _from_pages(name: str, pages: list[str], ocr: bool = False) -> Document:
    """Join PDF page texts with PAGE_SEPARATOR, recording each page's start offset (blank pages included)."""
    starts, offset = [], 0
    for page in pages:
        starts.append(offset)
        offset += len(page) + len(PAGE_SEPARATOR)
    text = PAGE_SEPARATOR.join(pages)
    _check_text_length(len(text))
    return Document(name=name, text=text, page_starts=tuple(starts), ocr=ocr, source_type="pdf")


# ----- PDF -----
def _open_pdf(data: bytes) -> PdfReader:
    """Open PDF bytes, unlocking owner-password-only encryption; ValueError if damaged, locked or too long."""
    try:
        reader = PdfReader(io.BytesIO(data))
        encrypted = reader.is_encrypted
    except Exception as error:  # untrusted bytes: any parser failure means the file can't be read
        raise ValueError(DAMAGED_PDF) from error

    # Many PDFs are encrypted only to restrict editing; they open with an empty user password
    if encrypted:
        try:
            unlocked = bool(reader.decrypt(""))
        except Exception:  # a failed unlock is reported as locked below
            unlocked = False
        if not unlocked:
            raise ValueError(LOCKED_PDF)

    # Count pages (the page tree is parsed lazily, so this can fail too)
    try:
        count = len(reader.pages)
    except Exception as error:
        raise ValueError(DAMAGED_PDF) from error
    if count == 0:
        raise ValueError("This PDF has no pages.")
    if count > MAX_PAGES:
        raise ValueError(f"This PDF has {count:,} pages, over the {MAX_PAGES:,}-page limit. "
                         "Split it into smaller files and upload them separately.")
    return reader


def _pdf_page_texts(reader: PdfReader) -> list[str]:
    """Each page's text layer ('' for a page without one), stopping once the text limit is passed."""
    pages, total = [], 0
    for index in range(len(reader.pages)):
        try:
            text = reader.pages[index].extract_text() or ""
        except Exception as error:
            raise ValueError(f"Page {index + 1} of this PDF is damaged and can't be read. Open the PDF in a "
                             "viewer, print it to a new PDF, and upload that.") from error
        total += len(text) + len(PAGE_SEPARATOR)
        _check_text_length(total - len(PAGE_SEPARATOR))
        pages.append(text)
    return pages


def ocr_pages(source: bytes | str | Path) -> list[str]:
    """OCR every page of a PDF (bytes or path); ValueError explaining what to install if OCR isn't available."""
    try:
        import pypdfium2
        import pytesseract
    except ImportError as error:
        raise ValueError("This PDF has no text layer (probably a scan). To read scans, install OCR support: "
                         "pip install pypdfium2 pytesseract, plus the Tesseract program.") from error
    try:
        pdf = pypdfium2.PdfDocument(str(source) if isinstance(source, Path) else source)
        if len(pdf) > MAX_PAGES:
            raise ValueError(f"This scan has more than {MAX_PAGES:,} pages, over the limit.")
        return [pytesseract.image_to_string(page.render(scale=300 / 72).to_pil()) for page in pdf]
    except pytesseract.TesseractNotFoundError as error:
        raise ValueError("This PDF is a scan and the Tesseract OCR program isn't installed.") from error
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("OCR couldn't read this scanned PDF; the file may be damaged.") from error


def _read_pdf(name: str, data: bytes) -> Document:
    """Read a PDF's text layer, falling back to OCR only when no page has any text."""
    pages = _pdf_page_texts(_open_pdf(data))
    if any(page.strip() for page in pages):
        return _from_pages(name, pages)
    pages = ocr_pages(data)
    if not any(page.strip() for page in pages):
        raise ValueError("No text found in this PDF, even with OCR.")
    return _from_pages(name, pages, ocr=True)


# ----- Word -----
def _check_docx_archive(data: bytes) -> None:
    """Reject a .docx whose ZIP is damaged or would expand past the limits (a 'zip bomb')."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except Exception as error:
        raise ValueError(DAMAGED_DOCX) from error
    if len(entries) > MAX_DOCX_ENTRIES or sum(entry.file_size for entry in entries) > MAX_DOCX_UNCOMPRESSED_BYTES:
        raise ValueError(f"This Word file expands to more than {MAX_DOCX_UNCOMPRESSED_BYTES // MB} MB, over the "
                         "limit. Remove large images or split it, then upload again.")


def _paragraph_text(paragraph: Any, parent: Any) -> str:
    """
    A <w:p>'s text with tracked changes accepted: runs inside insertions, hyperlinks and content
    controls count, deleted or moved-away runs don't, and text-box runs (their own <w:p>) are left out.
    """
    from docx.text.run import Run

    parts = []
    for run in paragraph.iter(f"{_W}r"):
        # The nearest of these ancestors decides: this paragraph keeps it, a deletion or a nested <w:p> doesn't
        owner = next(run.iterancestors(f"{_W}p", f"{_W}del", f"{_W}moveFrom"), None)
        if owner is paragraph:
            parts.append(Run(run, parent).text)
    return "".join(parts)


def _block_lines(container: Any, parent: Any) -> list[str]:
    """Paragraphs and table rows of a body or cell element in order, looking inside content controls."""
    from docx.table import Table

    lines = []
    for child in container.iterchildren():
        if child.tag == f"{_W}p":
            lines.append(_paragraph_text(child, parent))
        elif child.tag == f"{_W}tbl":
            lines.extend(_table_lines(Table(child, parent)))
        elif child.tag in (f"{_W}sdt", f"{_W}sdtContent", f"{_W}customXml"):
            lines.extend(_block_lines(child, parent))
    return lines


def _cell_text(cell: Any) -> str:
    """A table cell's paragraphs (and any nested table) on one line, whitespace runs collapsed."""
    return " ".join(" ".join(_block_lines(cell._tc, cell)).split())


def _table_lines(table: Any) -> list[str]:
    """One line per table row, cells joined by CELL_SEPARATOR; a merged cell is read once; empty rows skipped."""
    seen, lines = set(), []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            # Merged cells repeat the same <w:tc> element across the grid; keep its first appearance
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            cells.append(_cell_text(cell))
        if any(cells):
            lines.append(CELL_SEPARATOR.join(cells))
    return lines


def _read_docx(name: str, data: bytes) -> Document:
    """
    Read a .docx body: paragraphs and tables in order, tracked changes accepted. Headers, footers,
    footnotes, comments and text boxes are not read.
    """
    from docx import Document as open_word

    _check_docx_archive(data)
    try:
        word = open_word(io.BytesIO(data))
        text = "\n".join(_block_lines(word.element.body, word._body))
    except Exception as error:  # untrusted bytes: any parser failure means the file can't be read
        raise ValueError(DAMAGED_DOCX) from error
    _check_text_length(len(text))
    if not text.strip():
        raise ValueError(NO_TEXT)
    return Document(name=name, text=text, source_type="docx")


# ----- Plain text -----
def _decode_text(data: bytes) -> str:
    """Decode UTF-8 (BOM dropped), UTF-16 with a BOM, or else Windows-1252; line endings become '\\n'."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_txt(name: str, data: bytes) -> Document:
    """Read a plain-text file as one page-less Document."""
    text = _decode_text(data)
    _check_text_length(len(text))
    if not text.strip():
        raise ValueError(NO_TEXT)
    return Document(name=name, text=text, source_type="txt")
