"""Tiny hand-written PDFs for tests, so no PDF-writing library is needed (pypdf only for encryption)."""
from pathlib import Path


def _escape(line: str) -> bytes:
    """A line of text as a PDF string body: backslashes and parentheses escaped, Latin-1 encoded."""
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1")


def _page_content(text: str) -> bytes:
    """Content stream drawing text in Helvetica, one PDF text line per '\\n'-separated line."""
    lines = b" T* ".join(b"(%s) Tj" % _escape(line) for line in text.split("\n"))
    return b"BT /F1 12 Tf 14 TL 72 720 Td " + lines + b" ET"


def pdf_bytes(pages: list[str | None]) -> bytes:
    """A PDF with one page per entry: the text drawn on it ('\\n' starts a new line), or None for no text layer."""
    objects = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids = []
    for index, text in enumerate(pages):
        page_id, content_id = 4 + 2 * index, 5 + 2 * index
        content = _page_content(text) if text else b""
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"
        objects[page_id] = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
                            b"/Resources << /Font << /F1 3 0 R >> >> >>" % content_id)
        kids.append(b"%d 0 R" % page_id)
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(kids), len(kids))
    out, offsets = bytearray(b"%PDF-1.4\n"), {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"
    xref, size = len(out), max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    out += b"".join(b"%010d 00000 n \n" % offsets[n] for n in range(1, size))
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref)
    return bytes(out)


def write_pdf(path: Path, pages: list[str | None]) -> None:
    """Write pdf_bytes(pages) to path."""
    path.write_bytes(pdf_bytes(pages))


def write_encrypted_pdf(path: Path, pages: list[str | None], user_password: str, owner_password: str) -> None:
    """Write an RC4-encrypted copy of pdf_bytes(pages); an empty user_password opens without a password."""
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf_bytes(pages))))
    writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm="RC4-128")
    with open(path, "wb") as file:
        writer.write(file)
