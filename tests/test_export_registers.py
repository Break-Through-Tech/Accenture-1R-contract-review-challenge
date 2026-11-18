import json

import pytest

from scripts.export_registers import (DEFAULT_OUTPUT, parse_args, score_contracts, validation_contracts,
                                      write_registers)
from clausetriage.evidence.extractor import GroundedEvidence, Unknown
from clausetriage.evidence.schema import Evidence
from clausetriage.labels.loader import Contract
from clausetriage.register.export import build_registers, register_file_stem
from clausetriage.risk.schema import DEFAULT_WEIGHTS_PATH, ClauseScore, WeightSet

# An explicit weight set, so a change of the repo's default weights can't move these values
WEIGHTS = WeightSet(
    weight_version_id="test-export", created_by="tests", default_severity=0.5,
    signal_weights={"category_severity": 0.55, "one_sidedness": 0.15, "magnitude": 0.15, "contract_context": 0.15},
    medium_threshold=0.35, high_threshold=0.55,
)
SIGNALS = {"category_severity": 1.0, "one_sidedness": 0.5, "magnitude": 0.0, "contract_context": 0.0}
QUOTE = "Liability is unlimited."


def _clause(contract_id: str, category: str, composite: float, rating: str, has_evidence: bool) -> ClauseScore:
    """A hand-built chunk-0 ClauseScore under WEIGHTS."""
    return ClauseScore(contract_id=contract_id, chunk_index=0, category=category, probability=0.9, signals=SIGNALS,
                       composite_score=composite, rating=rating, has_evidence=has_evidence,
                       weight_version_id="test-export", weight_set_sha256=WEIGHTS.content_sha256)


def _registers() -> list:
    """Two contracts ranked together: 'Alpha Co.' with a quoted High clause, 'Beta' with an unquoted Low one."""
    alpha = {"contract_id": "Alpha Co.", "model_name": "m", "seed": 1,
             "grounded": [GroundedEvidence(contract_id="Alpha Co.", chunk_index=0, category="Uncapped Liability",
                                           evidence=Evidence(quote=QUOTE, start=0, end=len(QUOTE)))],
             "unknown": [], "clause_scores": [_clause("Alpha Co.", "Uncapped Liability", 0.6, "High", True)]}
    beta = {"contract_id": "Beta", "model_name": "m", "seed": 1, "grounded": [],
            "unknown": [Unknown(contract_id="Beta", chunk_index=0, category="Insurance", reason="no_cue_match")],
            "clause_scores": [_clause("Beta", "Insurance", 0.3, "Low", False)]}
    return build_registers([beta, alpha], WEIGHTS, texts={"Alpha Co.": QUOTE})


# ----- Contract selection -----
def test_validation_contracts_uses_evaluate_triage_split(tmp_path):
    """The 10-contract file's 20% validation split (seed 42) is the same two contracts, and limit keeps the first."""
    entries = [{"title": f"c{i}", "paragraphs": [{"context": f"Contract {i}.", "qas": []}]} for i in range(10)]
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"data": entries}), encoding="utf-8")
    # Hand-checked against sklearn train_test_split(test_size=0.2, random_state=42) on c0..c9
    assert [c.contract_id for c in validation_contracts(path)] == ["c8", "c1"]
    assert [c.contract_id for c in validation_contracts(path, limit=1)] == ["c8"]


def test_score_contracts_scores_each_contract_in_order_with_progress():
    """Each contract goes through predict_contract once, in order, with a progress line."""
    class FakePipeline:
        """Records the contracts it was asked to score."""
        def predict_contract(self, contract_id, text):
            """Echo the contract id and text back as the result."""
            return {"contract_id": contract_id, "text": text}

    contracts = [Contract(contract_id=cid, text=f"text of {cid}", spans=[]) for cid in ("a", "b")]
    messages = []
    results = score_contracts(FakePipeline(), contracts, progress=messages.append)
    assert results == [{"contract_id": "a", "text": "text of a"}, {"contract_id": "b", "text": "text of b"}]
    assert messages == ["[1/2] a", "[2/2] b"]


# ----- Writing -----
def test_write_registers_writes_each_register_and_a_ranked_index(tmp_path):
    """Two registers give four register files plus index.csv and index.json, ranked Alpha then Beta."""
    index = write_registers(_registers(), tmp_path, {"weight_version_id": "test-export", "test_data_used": False})
    alpha, beta = register_file_stem("Alpha Co."), register_file_stem("Beta")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"{alpha}.csv", f"{alpha}.json", f"{beta}.csv", f"{beta}.json", "index.csv", "index.json"])
    assert list(index["contract_id"]) == ["Alpha Co.", "Beta"] and list(index["rank"]) == [1, 2]

    # index.json carries the run metadata, totals and the same rows
    summary = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert (summary["weight_version_id"], summary["test_data_used"]) == ("test-export", False)
    assert (summary["contracts"], summary["clause_rows"], summary["manual_review_rows"]) == (2, 2, 1)
    assert summary["index"][1]["file_stem"] == beta and summary["index"][1]["unknown_count"] == 1


def test_write_registers_index_csv_matches_golden(tmp_path):
    """index.csv rows hold the hand-computed triage scores (0.5 * worst + 0.5 * worst / 3)."""
    write_registers(_registers(), tmp_path, {})
    alpha, beta = register_file_stem("Alpha Co."), register_file_stem("Beta")
    assert (tmp_path / "index.csv").read_text(encoding="utf-8") == (
        "rank,contract_id,triage_score,high_count,medium_count,low_count,unknown_count,clause_count,"
        "manual_review_count,file_stem\n"
        f"1,Alpha Co.,0.4,1,0,0,0,1,0,{alpha}\n"   # 0.3 + 0.1
        f"2,Beta,0.2,0,0,1,1,1,1,{beta}\n"          # 0.15 + 0.05
    )


def test_write_registers_refuses_two_registers_with_one_file_stem(tmp_path):
    """The same contract twice would overwrite its own files, so nothing is written."""
    register = _registers()[0]
    with pytest.raises(ValueError, match="same file stem"):
        write_registers([register, register], tmp_path / "out", {})
    assert not (tmp_path / "out").exists()


# ----- Options -----
def test_parse_args_defaults_write_to_the_ignored_outputs_folder():
    """By default registers go to outputs/registers with the default weights and every contract."""
    args = parse_args([])
    assert args.output == DEFAULT_OUTPUT and DEFAULT_OUTPUT.parent.name == "outputs"
    assert args.weights == DEFAULT_WEIGHTS_PATH and args.limit is None and args.bundle is None
