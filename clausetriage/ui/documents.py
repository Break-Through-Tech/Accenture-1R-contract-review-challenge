"""
Compatibility shim: document reading is now a core pipeline stage in clausetriage/ingest/documents.py.
Existing imports from clausetriage.ui.documents keep working; new code should import from clausetriage.ingest.documents.
"""
from clausetriage.ingest.documents import (PAGE_SEPARATOR, SUPPORTED_SUFFIXES, Document, document_from_bytes, ocr_pages,
                                  pasted_document, read_document)

__all__ = ["PAGE_SEPARATOR", "SUPPORTED_SUFFIXES", "Document", "document_from_bytes", "ocr_pages",
           "pasted_document", "read_document"]
