from src.ui.app import score_documents
from src.ui.documents import Document


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
    scored = score_documents(pipeline, [Document("nda", "a"), Document("lease", "b"), Document("nda", "c")])
    assert pipeline.ids == ["nda", "lease", "nda (2)"]
    assert [item.document.text for item in scored] == ["a", "b", "c"]
