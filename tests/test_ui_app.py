from src.ingest.documents import Document, pasted_document
from src.ui.app import score_documents


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
