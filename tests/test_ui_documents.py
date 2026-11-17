"""The reader tests live in tests/test_ingest_documents.py; this module only checks the src.ui.documents shim."""
import pytest

import src.ingest.documents as ingest
import src.ui.documents as shim


# ----- Re-exports -----
@pytest.mark.parametrize("name", shim.__all__)
def test_shim_reexports_the_ingest_objects(name):
    """Every name the shim exports is the very object from src.ingest.documents."""
    assert getattr(shim, name) is getattr(ingest, name)


def test_shim_document_accepts_legacy_positional_construction():
    """src/ui/report.py and its tests build Document(name, text, page_starts) positionally through the shim."""
    document = shim.Document("risky", "Clause one.\n\nClause two.", (0, 13))
    assert document.page_of(13) == 2 and document.sha256 == ingest.text_sha256(document.text)
