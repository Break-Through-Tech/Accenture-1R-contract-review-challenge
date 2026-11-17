"""
Compatibility shim: document reading is now a core pipeline stage in src/ingest/documents.py.
Existing imports from src.ui.documents keep working; new code should import from src.ingest.documents.
"""
from src.ingest.documents import (PAGE_SEPARATOR, SUPPORTED_SUFFIXES, Document, document_from_bytes, ocr_pages,
                                  pasted_document, read_document)

__all__ = ["PAGE_SEPARATOR", "SUPPORTED_SUFFIXES", "Document", "document_from_bytes", "ocr_pages",
           "pasted_document", "read_document"]
