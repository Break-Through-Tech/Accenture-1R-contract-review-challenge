import io
import sys
import types
import zipfile

import pytest
from pydantic import ValidationError

from src.ingest import documents
from src.ingest.documents import (DAMAGED_DOCX, DAMAGED_PDF, EMPTY_FILE, LOCKED_DOCX, LOCKED_PDF, PAGE_SEPARATOR,
                                  Document, document_from_bytes, ocr_pages, pasted_document, read_document,
                                  text_sha256)
from tests.pdf_fixtures import pdf_bytes, write_encrypted_pdf, write_pdf

# Hand-computed: SHA-256 of b"abc"
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
# Two pages "abc" and "de": "abc" + "\n\n" puts page 2 at offset 5
TWO_PAGES = "abc\n\nde"


def _docx_bytes(build) -> bytes:
    """Bytes of a .docx made by calling build(word_document) on a new python-docx Document."""
    from docx import Document as WordDocument

    word = WordDocument()
    build(word)
    buffer = io.BytesIO()
    word.save(buffer)
    return buffer.getvalue()


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    """Bytes of a ZIP archive holding the given name -> content files."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


# ----- Document model -----
def test_document_fills_sha256_from_text():
    """Omitting sha256 fills it with the hex SHA-256 of the UTF-8 text."""
    document = Document(name="a", text="abc")
    assert document.sha256 == ABC_SHA256 == text_sha256("abc")
    assert document.source_type == "pasted" and document.page_starts == () and not document.ocr


def test_document_positional_arguments_match_keywords():
    """The old dataclass order (name, text, page_starts, ocr) still works and builds the same model."""
    assert Document("a", TWO_PAGES, (0, 5), True) == Document(name="a", text=TWO_PAGES, page_starts=(0, 5), ocr=True)


@pytest.mark.parametrize("args,kwargs", [(("a", "b", (), False, "x"), {}), (("a",), {"name": "b", "text": "c"})])
def test_document_bad_positional_arguments_raise_type_error(args, kwargs):
    """More than four positional arguments, or a field given twice, is a TypeError."""
    with pytest.raises(TypeError):
        Document(*args, **kwargs)


def test_document_is_frozen():
    """Fields can't be reassigned after construction."""
    document = Document(name="a", text="abc")
    with pytest.raises(ValidationError):
        document.text = "changed"


def test_document_rejects_hash_that_does_not_match_text():
    """A sha256 for different text means the text was altered after extraction."""
    with pytest.raises(ValueError, match="sha256 does not match"):
        Document(name="a", text="abd", sha256=ABC_SHA256)


def test_document_round_trips_through_model_dump():
    """Serialising and validating again gives an equal Document (the stored hash is re-checked)."""
    document = Document(name="a", text=TWO_PAGES, page_starts=(0, 5), source_type="pdf")
    assert Document.model_validate(document.model_dump()) == document


@pytest.mark.parametrize("page_starts,match", [
    ((1,), "begin at 0"), ((0, 0), "strictly increasing"), ((0, 5, 3), "strictly increasing"),
    ((0, 8), "inside the text"),
])
def test_document_rejects_invalid_page_starts(page_starts, match):
    """page_starts must start at 0, increase strictly and stay within len(text)."""
    with pytest.raises(ValueError, match=match):
        Document(name="a", text=TWO_PAGES, page_starts=page_starts)


def test_document_allows_blank_trailing_page_at_text_end():
    """A blank last page starts at len(text): 'abc' + separator + '' gives starts (0, 5) on 5 characters."""
    document = Document(name="a", text="abc\n\n", page_starts=(0, 5))
    assert document.blank_pages() == (2,)


@pytest.mark.parametrize("offset,page", [(0, 1), (2, 1), (4, 1), (5, 2), (6, 2)])
def test_page_of_maps_offsets_to_pages(offset, page):
    """In 'abc\\n\\nde' with pages at (0, 5), offsets 0-4 are page 1 and 5-6 are page 2."""
    assert Document(name="a", text=TWO_PAGES, page_starts=(0, 5)).page_of(offset) == page


@pytest.mark.parametrize("offset", [-1, 7])
def test_page_of_rejects_offsets_outside_text(offset):
    """Offsets before the start or at/after len(text) are ValueErrors, not a guessed page."""
    with pytest.raises(ValueError, match="outside the text"):
        Document(name="a", text=TWO_PAGES, page_starts=(0, 5)).page_of(offset)


def test_page_of_is_none_without_pages():
    """Formats without pages report no page number."""
    assert Document(name="a", text="abc").page_of(1) is None


def test_blank_pages_lists_whitespace_only_pages():
    """Pages 'a', ' ', 'b' joined by the separator: only page 2 is blank."""
    document = Document(name="a", text="a\n\n \n\nb", page_starts=(0, 3, 6))
    assert document.blank_pages() == (2,)


# ----- PDFs -----
def test_read_document_pdf_records_page_starts(tmp_path):
    """Two PDF pages are joined by PAGE_SEPARATOR and each offset maps back to its page."""
    write_pdf(tmp_path / "contract.pdf", ["Governed by Delaware law.", "Either party may terminate."])
    document = read_document(tmp_path / "contract.pdf")

    assert document.name == "contract" and not document.ocr and document.source_type == "pdf"
    assert document.text == "Governed by Delaware law.\n\nEither party may terminate."
    assert document.page_starts == (0, 27)
    assert document.text[:27].endswith(PAGE_SEPARATOR)
    assert document.page_of(0) == 1 and document.page_of(26) == 1 and document.page_of(27) == 2


def test_read_document_pdf_is_deterministic(tmp_path):
    """Reading the same PDF twice gives identical text and hash."""
    write_pdf(tmp_path / "contract.pdf", ["Governed by Delaware law.\nSecond line.", "Page two."])
    first, second = read_document(tmp_path / "contract.pdf"), read_document(tmp_path / "contract.pdf")
    assert first == second and first.sha256 == text_sha256(first.text)


def test_read_document_pdf_keeps_blank_pages_in_numbering(tmp_path):
    """A page without text keeps its page number, so later pages aren't shifted; it is listed as blank."""
    write_pdf(tmp_path / "contract.pdf", ["Page one.", None, "Page three."])
    document = read_document(tmp_path / "contract.pdf")
    # "Page one." (9) + "\n\n" -> page 2 at 11; "" + "\n\n" -> page 3 at 13
    assert document.text == "Page one.\n\n\n\nPage three." and document.page_starts == (0, 11, 13)
    assert document.page_of(document.text.index("three")) == 3 and document.blank_pages() == (2,)


def test_read_document_scan_without_ocr_explains_what_to_install(tmp_path, monkeypatch):
    """A PDF with no text layer, and no OCR libraries, raises a ValueError naming the install."""
    monkeypatch.setitem(sys.modules, "pytesseract", None)  # makes `import pytesseract` fail
    write_pdf(tmp_path / "scan.pdf", [None])
    with pytest.raises(ValueError, match="install OCR support"):
        read_document(tmp_path / "scan.pdf")


def test_read_document_scan_uses_ocr_when_available(tmp_path, monkeypatch):
    """A scan's pages come from ocr_pages and the Document is marked as OCR'd."""
    monkeypatch.setattr(documents, "ocr_pages", lambda source: ["Scanned page one.", "Scanned page two."])
    write_pdf(tmp_path / "scan.pdf", [None, None])
    document = read_document(tmp_path / "scan.pdf")
    assert document.ocr and document.text == "Scanned page one.\n\nScanned page two."
    assert document.page_of(document.text.index("two")) == 2


def test_read_document_scan_with_blank_ocr_raises(tmp_path, monkeypatch):
    """If OCR finds nothing either, the reviewer is told no text was found."""
    monkeypatch.setattr(documents, "ocr_pages", lambda source: ["", "  "])
    write_pdf(tmp_path / "scan.pdf", [None, None])
    with pytest.raises(ValueError, match="even with OCR"):
        read_document(tmp_path / "scan.pdf")


def test_read_document_owner_password_only_pdf_is_read(tmp_path):
    """A PDF encrypted only to restrict editing (empty user password) opens like any viewer would."""
    write_encrypted_pdf(tmp_path / "locked.pdf", ["Governed by Delaware law."], user_password="",
                        owner_password="owner")
    assert read_document(tmp_path / "locked.pdf").text == "Governed by Delaware law."


def test_read_document_password_protected_pdf_raises(tmp_path):
    """A PDF that needs a password to open gives the password-protected message."""
    write_encrypted_pdf(tmp_path / "locked.pdf", ["Secret."], user_password="secret", owner_password="owner")
    with pytest.raises(ValueError) as error:
        read_document(tmp_path / "locked.pdf")
    assert str(error.value) == LOCKED_PDF


@pytest.mark.parametrize("cut", [40, 200, -60])
def test_read_document_truncated_pdf_raises_damaged(tmp_path, cut):
    """A PDF cut short (first 40 or 200 bytes, or missing its last 60) is reported as damaged."""
    (tmp_path / "cut.pdf").write_bytes(pdf_bytes(["Governed by Delaware law.", "Page two."])[:cut])
    with pytest.raises(ValueError) as error:
        read_document(tmp_path / "cut.pdf")
    assert str(error.value) == DAMAGED_PDF and error.value.__cause__ is not None


def test_read_document_garbage_after_pdf_header_raises_damaged(tmp_path):
    """A file with a PDF header but no PDF structure is reported as damaged."""
    (tmp_path / "junk.pdf").write_bytes(b"%PDF-1.4\nthis is not a PDF body")
    with pytest.raises(ValueError, match="damaged or incomplete"):
        read_document(tmp_path / "junk.pdf")


def test_read_document_unreadable_page_names_the_page(tmp_path, monkeypatch):
    """If pypdf fails on one page, the error says which page, not a pypdf stack trace."""
    from pypdf import PageObject

    calls = []

    def flaky(self, *args, **kwargs):
        """Fail on the second page read."""
        calls.append(1)
        if len(calls) == 2:
            raise KeyError("/Contents")
        return "Page text."

    monkeypatch.setattr(PageObject, "extract_text", flaky)
    write_pdf(tmp_path / "contract.pdf", ["One.", "Two."])
    with pytest.raises(ValueError, match="Page 2 of this PDF is damaged"):
        read_document(tmp_path / "contract.pdf")


def test_read_document_pdf_over_page_limit_raises(tmp_path, monkeypatch):
    """A PDF with more pages than MAX_PAGES is refused before any text is extracted."""
    monkeypatch.setattr(documents, "MAX_PAGES", 2)
    write_pdf(tmp_path / "long.pdf", ["One.", "Two.", "Three."])
    with pytest.raises(ValueError, match="3 pages, over the 2-page limit"):
        read_document(tmp_path / "long.pdf")


def test_read_document_pdf_over_text_limit_raises(tmp_path, monkeypatch):
    """Extracted text longer than MAX_TEXT_CHARS is refused."""
    monkeypatch.setattr(documents, "MAX_TEXT_CHARS", 10)
    write_pdf(tmp_path / "long.pdf", ["Governed by Delaware law."])
    with pytest.raises(ValueError, match="characters of text, over the limit"):
        read_document(tmp_path / "long.pdf")


# ----- Optional OCR (pypdfium2 and pytesseract are mocked; neither is installed) -----
def _fake_ocr_modules(monkeypatch, pages: int, missing_tesseract: bool = False) -> None:
    """Install fake pypdfium2 / pytesseract modules: each rendered page OCRs to 'OCR page N'."""
    class FakePage:
        """A page whose render().to_pil() returns its own number."""
        def __init__(self, number):
            self.number = number

        def render(self, scale):
            """Return self as the 'bitmap'."""
            return self

        def to_pil(self):
            """Return the page number as the 'image'."""
            return self.number

    class TesseractNotFoundError(EnvironmentError):
        """Stand-in for pytesseract's error when the Tesseract program is missing."""

    def image_to_string(image):
        """OCR the fake image, or fail as if Tesseract were missing."""
        if missing_tesseract:
            raise TesseractNotFoundError()
        return f"OCR page {image}"

    pdfium = types.SimpleNamespace(PdfDocument=lambda source: [FakePage(n) for n in range(1, pages + 1)])
    tesseract = types.SimpleNamespace(image_to_string=image_to_string, TesseractNotFoundError=TesseractNotFoundError)
    monkeypatch.setitem(sys.modules, "pypdfium2", pdfium)
    monkeypatch.setitem(sys.modules, "pytesseract", tesseract)


def test_ocr_pages_reads_every_page_with_fake_ocr(monkeypatch):
    """With OCR libraries present, each rendered page is OCR'd in order."""
    _fake_ocr_modules(monkeypatch, pages=2)
    assert ocr_pages(b"%PDF-") == ["OCR page 1", "OCR page 2"]


def test_read_document_scan_runs_real_ocr_path_with_fakes(tmp_path, monkeypatch):
    """A text-less PDF goes through ocr_pages and gets page starts from the OCR text."""
    _fake_ocr_modules(monkeypatch, pages=2)
    write_pdf(tmp_path / "scan.pdf", [None, None])
    document = read_document(tmp_path / "scan.pdf")
    # "OCR page 1" (10) + "\n\n" -> page 2 at 12
    assert document.ocr and document.text == "OCR page 1\n\nOCR page 2" and document.page_starts == (0, 12)


def test_ocr_pages_without_tesseract_program_raises(monkeypatch):
    """Libraries installed but the Tesseract program missing gives a clear ValueError."""
    _fake_ocr_modules(monkeypatch, pages=1, missing_tesseract=True)
    with pytest.raises(ValueError, match="Tesseract OCR program isn't installed"):
        ocr_pages(b"%PDF-")


def test_ocr_pages_over_page_limit_raises(monkeypatch):
    """A scan longer than MAX_PAGES isn't rendered."""
    _fake_ocr_modules(monkeypatch, pages=3)
    monkeypatch.setattr(documents, "MAX_PAGES", 2)
    with pytest.raises(ValueError, match="more than 2 pages"):
        ocr_pages(b"%PDF-")


# ----- Word -----
def test_read_document_docx_paragraphs_have_no_pages(tmp_path):
    """Word paragraphs are joined by newlines, with no page numbers."""
    def build(word):
        """Two paragraphs."""
        word.add_paragraph("Governed by Delaware law.")
        word.add_paragraph("Either party may terminate.")

    (tmp_path / "b.docx").write_bytes(_docx_bytes(build))
    document = read_document(tmp_path / "b.docx")
    assert document.text == "Governed by Delaware law.\nEither party may terminate."
    assert document.source_type == "docx" and document.page_of(5) is None


def test_read_document_docx_reads_tables_in_document_order(tmp_path):
    """A table between paragraphs becomes one line per row, cells joined by ' | ', in place."""
    def build(word):
        """Paragraph, 2x2 table of key terms, paragraph."""
        word.add_paragraph("Master Services Agreement")
        table = word.add_table(rows=2, cols=2)
        for (row, column), value in {(0, 0): "Term", (0, 1): "5 years", (1, 0): "Governing law",
                                     (1, 1): "Delaware"}.items():
            table.cell(row, column).text = value
        word.add_paragraph("Signed by both parties.")

    (tmp_path / "t.docx").write_bytes(_docx_bytes(build))
    assert read_document(tmp_path / "t.docx").text == (
        "Master Services Agreement\nTerm | 5 years\nGoverning law | Delaware\nSigned by both parties.")


def test_read_document_docx_reads_merged_cells_once(tmp_path):
    """A horizontally merged cell appears once in its row; a vertically merged one only in its first row."""
    def build(word):
        """One table with a horizontal merge, one with a vertical merge."""
        wide = word.add_table(rows=2, cols=3)
        wide.cell(0, 0).merge(wide.cell(0, 1)).text = "Payment terms"
        wide.cell(0, 2).text = "Net 30"
        for column, value in enumerate("abc"):
            wide.cell(1, column).text = value
        tall = word.add_table(rows=2, cols=2)
        tall.cell(0, 0).merge(tall.cell(1, 0)).text = "Party"
        tall.cell(0, 1).text, tall.cell(1, 1).text = "Acme", "Beta"

    (tmp_path / "m.docx").write_bytes(_docx_bytes(build))
    assert read_document(tmp_path / "m.docx").text == "Payment terms | Net 30\na | b | c\nParty | Acme\nBeta"


def test_read_document_docx_accepts_tracked_changes_and_reads_content_controls(tmp_path):
    """Insertions are kept, deletions dropped, content controls read; text boxes and headers are skipped."""
    from docx.oxml import parse_xml

    w = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    v = 'xmlns:v="urn:schemas-microsoft-com:vml"'

    def build(word):
        """A redlined fee, an inline and a block content control, a text box and a header."""
        fee = word.add_paragraph("Fee is ")
        fee._p.append(parse_xml(f'<w:ins {w} w:id="1" w:author="a"><w:r><w:t>$200</w:t></w:r></w:ins>'))
        fee._p.append(parse_xml(f'<w:del {w} w:id="2" w:author="a"><w:r><w:delText>$100</w:delText></w:r></w:del>'))
        party = word.add_paragraph("Party: ")
        party._p.append(parse_xml(f"<w:sdt {w}><w:sdtContent><w:r><w:t>Acme Corp</w:t></w:r></w:sdtContent></w:sdt>"))
        party._p.append(parse_xml(f"<w:r {w} {v}><w:pict><v:shape><v:textbox><w:txbxContent><w:p><w:r>"
                                  "<w:t>TEXT BOX</w:t></w:r></w:p></w:txbxContent></v:textbox></v:shape></w:pict></w:r>"))
        party._p.addnext(parse_xml(f"<w:sdt {w}><w:sdtContent><w:p><w:r><w:t>Governing law: Delaware</w:t></w:r>"
                                   "</w:p></w:sdtContent></w:sdt>"))
        word.sections[0].header.paragraphs[0].text = "CONFIDENTIAL"

    (tmp_path / "r.docx").write_bytes(_docx_bytes(build))
    assert read_document(tmp_path / "r.docx").text == "Fee is $200\nParty: Acme Corp\nGoverning law: Delaware"


def test_read_document_docx_flattens_nested_tables_into_their_cell(tmp_path):
    """A table inside a cell is read into that cell's text, keeping one line per outer row."""
    def build(word):
        """A 1x2 table whose second cell holds 'Fees' and a nested 1x2 table."""
        table = word.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Schedule"
        cell = table.cell(0, 1)
        cell.paragraphs[0].text = "Fees"
        nested = cell.add_table(rows=1, cols=2)
        nested.cell(0, 0).text, nested.cell(0, 1).text = "Fee", "$100"

    (tmp_path / "n.docx").write_bytes(_docx_bytes(build))
    assert read_document(tmp_path / "n.docx").text == "Schedule | Fees Fee | $100"


def test_read_document_docx_without_text_raises(tmp_path):
    """A Word file with no body text is refused."""
    (tmp_path / "empty.docx").write_bytes(_docx_bytes(lambda word: word.add_paragraph("   ")))
    with pytest.raises(ValueError, match="no text"):
        read_document(tmp_path / "empty.docx")


@pytest.mark.parametrize("data,message", [
    (_zip_bytes({"hello.txt": b"not a Word document"}), DAMAGED_DOCX),
    (b"PK\x03\x04" + b"\x00" * 100, DAMAGED_DOCX),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100, LOCKED_DOCX),
])
def test_read_document_bad_docx_raises_clear_message(tmp_path, data, message):
    """A ZIP that isn't Word, a corrupt ZIP, and an encrypted (OLE) Word file each get a message to act on."""
    (tmp_path / "bad.docx").write_bytes(data)
    with pytest.raises(ValueError) as error:
        read_document(tmp_path / "bad.docx")
    assert str(error.value) == message


def test_read_document_docx_zip_bomb_raises(tmp_path, monkeypatch):
    """A .docx that would expand past MAX_DOCX_UNCOMPRESSED_BYTES is refused before parsing."""
    monkeypatch.setattr(documents, "MAX_DOCX_UNCOMPRESSED_BYTES", 100)
    (tmp_path / "big.docx").write_bytes(_zip_bytes({"word/document.xml": b"<w/>" * 100}))
    with pytest.raises(ValueError, match="expands to more than"):
        read_document(tmp_path / "big.docx")


# ----- Plain text -----
@pytest.mark.parametrize("data,text", [
    (b"Governed by Delaware law.", "Governed by Delaware law."),
    (b"\xef\xbb\xbfWith BOM.", "With BOM."),
    (b"Line one.\r\nLine two.\rLine three.", "Line one.\nLine two.\nLine three."),
    ("UTF-16 café.".encode("utf-16"), "UTF-16 café."),
    (b"Windows caf\xe9.", "Windows café."),
])
def test_read_document_txt_decodes_deterministically(tmp_path, data, text):
    """UTF-8 (BOM dropped), UTF-16 with BOM and Windows-1252 decode; line endings become '\\n'."""
    (tmp_path / "a.txt").write_bytes(data)
    document = read_document(tmp_path / "a.txt")
    assert document == Document(name="a", text=text, source_type="txt")


# ----- Untrusted files: empty, wrong type, too large -----
@pytest.mark.parametrize("name", ["empty.pdf", "empty.docx", "empty.txt"])
def test_read_document_zero_byte_file_raises(tmp_path, name):
    """A 0-byte file of any supported type gets the empty-file message."""
    (tmp_path / name).write_bytes(b"")
    with pytest.raises(ValueError) as error:
        read_document(tmp_path / name)
    assert str(error.value) == EMPTY_FILE


def test_read_document_whitespace_txt_raises(tmp_path):
    """A text file with only whitespace has no text."""
    (tmp_path / "blank.txt").write_bytes(b" \n\t ")
    with pytest.raises(ValueError, match="no text"):
        read_document(tmp_path / "blank.txt")


@pytest.mark.parametrize("name,data,match", [
    ("contract.docx", pdf_bytes(["Text."]), "named .docx but isn't a Word document; it looks like a PDF"),
    ("contract.pdf", _zip_bytes({"a": b"b"}), "named .pdf but isn't a PDF; it looks like a ZIP"),
    ("contract.pdf", b"Plain text, not a PDF.", r"named .pdf but isn't a PDF\. Check"),
    ("contract.txt", pdf_bytes(["Text."]), "named .txt but isn't plain text; it looks like a PDF"),
    ("contract.txt", b"MZ\x90\x00\x03\x00binary", "named .txt but isn't plain text"),
    ("contract.txt", b"{\\rtf1 Hello}", "it looks like an RTF file"),
])
def test_read_document_wrong_extension_raises(tmp_path, name, data, match):
    """Contents that don't match the extension are refused, saying what the file looks like."""
    (tmp_path / name).write_bytes(data)
    with pytest.raises(ValueError, match=match):
        read_document(tmp_path / name)


@pytest.mark.parametrize("name,match", [
    ("a.rtf", "Can't read a file type '.rtf'"), ("a.doc", "Old Word .doc files"),
    ("README", "without an extension"),
])
def test_read_document_rejects_unsupported_types(tmp_path, name, match):
    """Unsupported extensions are refused before the file is read."""
    (tmp_path / name).write_bytes(b"text")
    with pytest.raises(ValueError, match=match):
        read_document(tmp_path / name)


def test_read_document_over_size_limit_raises(tmp_path, monkeypatch):
    """A file bigger than MAX_FILE_BYTES is refused before it is read."""
    monkeypatch.setattr(documents, "MAX_FILE_BYTES", documents.MB)
    (tmp_path / "big.txt").write_bytes(b"x" * (documents.MB + 1))
    with pytest.raises(ValueError, match=r"This file is 1\.0 MB, over the 1 MB limit"):
        read_document(tmp_path / "big.txt")


@pytest.mark.parametrize("make", [lambda root: root / "missing.txt", lambda root: root / "folder.txt"])
def test_read_document_missing_or_unreadable_path_raises(tmp_path, make):
    """A path that doesn't exist, or is a folder, gives a ValueError instead of an OSError."""
    (tmp_path / "folder.txt").mkdir()
    with pytest.raises(ValueError, match="missing or can't be read"):
        read_document(make(tmp_path))


@pytest.mark.parametrize("filename,name", [("../../etc/contract.txt", "contract"), ("  .txt", "Untitled")])
def test_document_from_bytes_names_document_by_file_stem(filename, name):
    """An upload name's directories are dropped; a blank stem becomes 'Untitled'."""
    document = document_from_bytes(b"Governed by Delaware law.", filename)
    assert document.name == name and document.text == "Governed by Delaware law."


# ----- Pasted text -----
def test_pasted_document_keeps_text_unchanged():
    """Pasted text is the source as-is: no newline or whitespace changes, no pages."""
    document = pasted_document("  Clause one.\r\nClause two.  ")
    assert document.text == "  Clause one.\r\nClause two.  " and document.name == "Pasted text"
    assert document.source_type == "pasted" and document.page_starts == ()


@pytest.mark.parametrize("text,error", [("", ValueError), (" \n ", ValueError), (None, TypeError)])
def test_pasted_document_rejects_empty_or_non_string(text, error):
    """Empty, whitespace-only or non-string pasted text is refused."""
    with pytest.raises(error):
        pasted_document(text)


def test_pasted_document_over_text_limit_raises(monkeypatch):
    """Pasted text longer than MAX_TEXT_CHARS is refused."""
    monkeypatch.setattr(documents, "MAX_TEXT_CHARS", 3)
    with pytest.raises(ValueError, match="over the limit"):
        pasted_document("abcd")
