import subprocess
import sys
from pathlib import Path

from src.ingest.documents import Document, pasted_document
from src.ui.app import score_documents

ROOT = Path(__file__).resolve().parents[1]


class RecordingPipeline:
    """Stands in for ContractInferencePipeline and records the contract ids it was given."""
    def __init__(self):
        self.ids = []

    def predict_contract(self, contract_id, text):
        """Return a minimal result carrying the id."""
        self.ids.append(contract_id)
        return {"contract_id": contract_id}


# ----- Scoring uploads -----
def test_score_documents_makes_repeated_names_unique():
    """Two uploads with the same name get distinct contract ids, in upload order."""
    pipeline = RecordingPipeline()
    uploads = [Document(name="nda", text="a"), Document(name="lease", text="b"), Document(name="nda", text="c")]
    scored = score_documents(pipeline, uploads)
    assert pipeline.ids == ["nda", "lease", "nda (2)"]
    assert [item.document.text for item in scored] == ["a", "b", "c"]


def test_score_documents_scores_pasted_text_unchanged():
    """Pasted text reaches the pipeline exactly as typed, under the 'Pasted text' id."""
    pipeline = RecordingPipeline()
    scored = score_documents(pipeline, [pasted_document("Clause one.\r\n")])
    assert pipeline.ids == ["Pasted text"] and scored[0].document.text == "Clause one.\r\n"


# ----- The demo-only install -----
def test_demo_page_imports_without_the_store_or_api_layers():
    """python -m src.ui.app needs no SQLAlchemy (requirements-demo.txt has none): importing it loads neither layer."""
    check = ("import sys, src.ui.app; "
             "print(sorted(m for m in ('sqlalchemy', 'src.store', 'src.api', 'src.ui.reviewer') if m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", check], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
