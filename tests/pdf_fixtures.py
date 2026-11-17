"""Tiny hand-written PDFs for tests, so no PDF-writing library is needed."""
from pathlib import Path


def write_pdf(path: Path, pages: list[str | None]) -> None:
    """Write a PDF with one page per entry: the text drawn on it, or None for a page with no text layer."""
    objects = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids = []
    for index, text in enumerate(pages):
        page_id, content_id = 4 + 2 * index, 5 + 2 * index
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
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
    path.write_bytes(bytes(out))
