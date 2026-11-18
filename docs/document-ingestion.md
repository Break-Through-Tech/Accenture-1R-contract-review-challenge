# Document ingestion: PDF, Word and text uploads

The advisor has made PDF upload a required feature, so reading uploaded files is now a core
pipeline stage (`clausetriage/ingest/documents.py`) rather than part of the demo page. This page describes
what the stage accepts, what its offsets mean, how it treats untrusted files, and how much PDF
extraction costs in grounding.

**Short answer on fidelity:** on the one CUAD contract we have as a PDF, 7 of 60 gold answer spans
survive extraction verbatim and 36 of 60 survive after whitespace is normalized. The rest are
lost to line-wrap hyphens (15 spans) and to apostrophes that come back as curly quotes (9 spans).
None of this breaks grounding, because evidence is always quoted from the extracted text itself.
It does mean the extracted text is not the CUAD text, so gold spans can't be projected onto it
verbatim. See [Extraction fidelity](#extraction-fidelity).

## One text source

Chunk offsets, evidence offsets and page numbers all index into one string: `Document.text`.
Evidence is accepted only when `text[start:end] == quote`. For an uploaded file that string is the
extracted text, so:

- Extraction is deterministic. The same bytes read with the same `pypdf` / `python-docx` versions
  give the same text.
- Nothing changes the text after extraction. `Document` is a frozen Pydantic model, and its
  `sha256` field is the SHA-256 of the text, checked whenever a `Document` is built or validated.
  Store `sha256` next to any saved offsets. If a later re-extraction (for example after a `pypdf`
  upgrade) gives a different hash, the stored offsets no longer apply.
- Offsets refer to `Document.text`. They do not refer to bytes in the PDF, positions on a printed
  page, or the CUAD `context` string.

## Public interface

```python
from clausetriage.ingest.documents import (Document, read_document, document_from_bytes, pasted_document,
                                  ocr_pages, text_sha256, PAGE_SEPARATOR, SUPPORTED_SUFFIXES)
```

| Name | What it is |
|---|---|
| `Document` | Frozen Pydantic model. Fields: `name: str`, `text: str`, `page_starts: tuple[int, ...] = ()`, `ocr: bool = False`, `source_type: Literal["pdf", "docx", "txt", "pasted"] = "pasted"`, `sha256: str` (filled from `text` when omitted, rejected if it doesn't match). Methods: `page_of(offset) -> int \| None`, `blank_pages() -> tuple[int, ...]`. |
| `read_document(path) -> Document` | Reads a `.pdf`, `.docx` or `.txt` file from disk. Checks the size before reading. |
| `document_from_bytes(data, filename) -> Document` | The same reader for bytes already in memory (an API upload). The type comes from the filename's extension and the name from its stem. |
| `pasted_document(text, name="Pasted text") -> Document` | Wraps pasted text unchanged. |
| `ocr_pages(source) -> list[str]` | Optional OCR of a PDF (bytes or path). |
| `text_sha256(text) -> str` | The hash `Document.sha256` holds. |

Validation rules for `page_starts`: it must begin at 0 and be strictly increasing, and its last
start must be `<= len(text)`. A blank last page can start exactly at `len(text)`. `page_of` raises
`ValueError` for an offset outside `[0, len(text))` rather than guessing a page.

For backward compatibility `Document` still accepts the old dataclass's positional order,
`Document(name, text, page_starts, ocr)`. `clausetriage/ui/report.py` and its tests build it that way through
the shim `clausetriage/ui/documents.py`, which now only re-exports this module. New code should use keywords
and import from `clausetriage.ingest.documents`.

## Supported formats

| Input | `source_type` | How the text is built | Pages |
|---|---|---|---|
| `.pdf` with a text layer | `pdf` | `pypdf` `extract_text()` per page, pages joined with `PAGE_SEPARATOR` (`"\n\n"`) | Yes |
| `.pdf` with no text on any page (a scan) | `pdf`, `ocr=True` | Optional OCR, see below | Yes |
| `.docx` | `docx` | Body paragraphs and tables in document order, one line each (details below) | No |
| `.txt` | `txt` | UTF-8 (BOM dropped). UTF-16 if the file starts with a BOM. Otherwise Windows-1252. `\r\n` and `\r` become `\n`. | No |
| Pasted text | `pasted` | Exactly as pasted, unchanged | No |

`.doc`, `.rtf` and other types are refused with a message saying what to convert them to.

**Word files.** The reader walks the body XML itself, because python-docx's own iterators skip some
content:

- **Tables are read, not dropped.** Contracts often keep key terms (fees, term, parties) in tables.
  Each row becomes one line with cells joined by `" | "`. A merged cell is read once: a horizontal
  merge appears once in its row, and a vertical merge appears only in its first row. A table nested
  in a cell is flattened into that cell's text. Rows with no text are skipped.
- **Tracked changes are accepted.** Inserted text is kept and deleted text is dropped, so the text
  matches the document with all changes accepted. python-docx's `paragraph.text` would silently
  drop insertions.
- **Content controls** (`<w:sdt>`, common in templates) are read, both inline and around whole paragraphs.
- **Not read:** headers, footers, footnotes, endnotes, comments and text boxes. A clause that sits
  only in a text box or footnote is missed.
- Word files have no fixed pages (pagination depends on the renderer), so `page_of` returns `None`.
  Upload a PDF export when page numbers matter.

## Offsets and page numbers

- `page_starts[i]` is the offset in `Document.text` where page `i + 1` begins. Page numbers are
  physical, 1-based page positions in the file, not printed page labels such as "iv" or "A-3".
- **Blank pages keep their number.** A page with no text layer still gets a start offset, so later
  pages aren't shifted. `Document.blank_pages()` lists them, for example image-only signature pages,
  or scanned pages mixed into an otherwise digital PDF. OCR runs only when every page is blank;
  mixed PDFs are not partially OCR'd.
- With the current evidence extractor a quote never crosses a page break, because
  `PAGE_SEPARATOR` contains a line break and the extractor splits sentences on line breaks.
  `tests/test_ingest_grounding.py` checks this.

## Optional OCR

A PDF where no page has a text layer is sent to `ocr_pages`. That function renders each page at
300 dpi with `pypdfium2` and reads it with `pytesseract`, which needs the separate Tesseract
program. Neither package is in `requirements.txt`. Without them the reviewer is told what to
install; if OCR finds no text either, the reviewer is told that. OCR output depends on the
Tesseract version, so `sha256` is what ties stored offsets to a given OCR run. Tests cover this path
with mocked modules only (`tests/test_ingest_documents.py`). Real OCR quality has not been measured.

## Limits

| Limit | Value | Constant |
|---|---|---|
| File size | 25 MB (checked before the file is read) | `MAX_FILE_BYTES` |
| PDF pages (text layer and OCR) | 1,000 | `MAX_PAGES` |
| Extracted text | 2,000,000 characters (the longest contract in the CUAD training file has 338,211) | `MAX_TEXT_CHARS` |
| `.docx` expanded size | 100 MB, at most 5,000 archive entries | `MAX_DOCX_UNCOMPRESSED_BYTES`, `MAX_DOCX_ENTRIES` |

The demo page also caps uploads at `MAX_FILE_SIZE = "25mb"` in `clausetriage/ui/app.py`. The two values are
set independently.

## Untrusted files

Every upload is treated as hostile. In order, the reader:

1. Refuses unsupported extensions before reading anything.
2. Refuses missing, unreadable, empty (0-byte) and oversized files using the file size alone.
3. Checks that the first bytes match the extension (`%PDF-` for PDF, a ZIP header for `.docx`, no
   binary data for `.txt`). The error says what the file looks like instead, for example "it looks
   like a PDF". A password-protected Word file is an OLE container, not a ZIP, and gets its own message.
4. Parses inside a catch-all. Any `pypdf` / `python-docx` / `zipfile` exception becomes a
   `ValueError` with a plain message ("This PDF is damaged or incomplete…", "Page 7 of this PDF is
   damaged…", "This PDF is password-protected…"). The original exception is kept as `__cause__` for
   debugging, so reviewers never see a stack trace.
5. Opens PDFs encrypted only with an owner password (to restrict editing or printing) using the
   empty user password, as any PDF viewer does. PDFs that need a password to open are refused.
6. Checks a `.docx`'s declared uncompressed size before parsing (zip bombs). Python's `zipfile`
   never inflates an entry past its declared size. python-docx parses XML with
   `resolve_entities=False`, so external entities (XXE) aren't fetched. `pypdf` 7.1 has its own caps on
   decompressed stream sizes.
7. Never writes the upload to disk or runs anything from it. Only the filename's stem is kept, as
   `Document.name`. Directory parts are dropped.

**Not protected:** time. A crafted PDF can still be slow to parse within the size limit, and there is
no timeout. The API should run ingestion in a worker with a time limit.

## Extraction fidelity

`python -m scripts.evaluate_pdf_ingestion` reads
`data/demo_contracts/cuad_validation_fuelcell_development_agreement.pdf` with `read_document`. It
then loads the same contract's original `context` and gold spans from
`train_separate_questions.json` (it refuses `test.json`). The script confirms the contract is in
the seed-42 validation split and counts how many gold spans still occur in the extracted text.

Measured on this branch (pypdf 7.1.0):

| | Value |
|---|---|
| Original CUAD `context` | 105,662 characters |
| Extracted PDF text | 105,213 characters, 32 pages, no blank pages |
| Text identical / identical after whitespace normalization | No / No |
| Gold answer spans (distinct spans) | 60 (48) |
| Found **verbatim** | **7 / 60** |
| Found after **whitespace normalization** (runs collapsed to one space) | **36 / 60** |
| Found ignoring whitespace entirely (diagnostic) | 51 / 60 |
| Found ignoring whitespace and curly-vs-straight quotes (diagnostic) | 60 / 60 |

Where the losses come from:

- **Line wraps (53 spans not verbatim).** The PDF wraps lines where CUAD has spaces, so most spans
  longer than a line contain `\n` in the extracted text. Collapsing whitespace recovers 29 of them.
- **Hyphen wraps (15 more spans).** A line that breaks after a hyphen extracts as `non-\nexclusive`,
  which normalizes to `non- exclusive`, not `non-exclusive`. This hits license-grant spans the most.
  Removing the whitespace after each hyphen, on top of normalization, finds exactly 51 spans, the
  same as ignoring whitespace entirely. So hyphen wraps account for all 15.
- **Apostrophes (the last 9 spans, exactly the 9 that contain `'`).** The PDF's Helvetica font
  declares no encoding, so the PDF standard encoding applies. In that encoding byte `0x27` is a
  right single quote, so every `'` comes back as `’` (87 times in this contract).
  This is a property of how the sample PDF was rendered, not of `pypdf`. CUAD's 4 bullets and 1
  ellipsis didn't survive the rendering either, but no gold span contains them.

What this means:

- **Grounding is not affected.** Evidence is selected from, and validated against, the extracted
  text. `tests/test_ingest_grounding.py` runs the real chunker and evidence extractor (every
  category predicted for every chunk) over all three sample PDFs and a hand-made PDF with a blank
  page. Every quote satisfies `text[start:end] == quote` and is found on the page `page_of` reports,
  checked against page offsets rebuilt from `pypdf`'s per-page text. On the three sample PDFs that
  is 316, 424 and 1,087 grounded quotes, with 0 span mismatches and 0 quotes crossing a page.
- **CUAD labels don't transfer to PDF text verbatim.** Evaluating a model on PDF-extracted text, or
  projecting gold spans onto it, needs whitespace and quote normalization with an offset map back
  to the extracted text. Only 7 of 60 spans can be located without one.
- **Model inputs differ from training text.** The classifier was trained on CUAD `context` strings
  and sees line-wrapped text from PDFs. How much that changes detection quality has not been
  measured.
- **The sample is small.** This is one contract, rendered by one tool. The NASA sample PDFs have no
  CUAD source text to compare against.

## Tests

- `tests/test_ingest_documents.py`: the `Document` model (hash, validation, `page_of`,
  `blank_pages`, positional compatibility), PDF reading (page starts, blank pages, determinism,
  owner-password and locked PDFs, truncated and garbage files, page and text limits, an unreadable
  page), mocked OCR, Word (tables, merged and nested cells, tracked changes, content controls,
  damaged, locked and zip-bomb files), text decoding, wrong extensions, empty and oversized files,
  and pasted text.
- `tests/test_ingest_grounding.py`: the end-to-end grounding and page invariant described above.
  It needs the pinned DistilRoBERTa tokenizer in the local Hugging Face cache.
- `tests/test_evaluate_pdf_ingestion.py`: the fidelity script on a synthetic CUAD file.
- `tests/test_ui_documents.py`: the `clausetriage/ui/documents.py` re-export shim.
