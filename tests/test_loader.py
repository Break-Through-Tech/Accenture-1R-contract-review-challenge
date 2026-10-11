import json
from pathlib import Path
import pytest
from src.labels.loader import AnswerSpan, load_cuad


def write_cuad_json(tmp_path, context, qas, title="T") -> Path:
    """Write a one contract CUAD shaped JSON file under tmp_path and return its path."""
    data = {
        "data": [
            {
                "title": title,
                "paragraphs": [
                    {"context": context, "qas": qas}
                ]
            }
        ]
    }

    path = tmp_path / "cuad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ----- Loader: spans built from a handmade CUAD JSON example ----- 
def test_load_cuad_hand_made_json_returns_exact_spans(tmp_path):
    """Checks load_cuad returns the exact hand-computed spans and skips unanswered questions."""
    # "Alpha Beta Gamma": Alpha = [0:5], Beta = [6:10], Gamma = [11:16]
    context = "Alpha Beta Gamma"
    qas = [
        {"id": "T__Parties_0", "answers": [{"text": "Alpha", "answer_start": 0}]},
        {"id": "T__Parties_1", "answers": [{"text": "Gamma", "answer_start": 11}]},
        {"id": "T__Governing Law_0", "answers": []},
    ]
    path = write_cuad_json(tmp_path, context, qas)
    contracts = load_cuad(path)

    assert len(contracts) == 1
    assert contracts[0].contract_id == "T"
    assert contracts[0].text == context
    assert contracts[0].spans == [
        AnswerSpan(category="Parties", start=0, end=5, text="Alpha"),
        AnswerSpan(category="Parties", start=11, end=16, text="Gamma"),
    ]


# ----- Loader: category name is parsed from the question ID -----
@pytest.mark.parametrize("qa_id, expected", [
    ("T__Parties_1", "Parties"),
    ("T__Document Name_0", "Document Name"),
    ("T__Rofr/Rofo/Rofn_0", "Rofr/Rofo/Rofn"),
    ("A_B__Parties_3", "Parties"),
    ("T__Parties_12", "Parties"),
])
def test_load_cuad_question_id_yields_category_name(tmp_path, qa_id, expected):
    """Checks the category is the part after the last '__' with the trailing _N removed."""
    context = "Alpha"
    qas = [{"id": qa_id, "answers": [{"text": "Alpha", "answer_start": 0}]}]
    path = write_cuad_json(tmp_path, context, qas)

    assert load_cuad(path)[0].spans[0].category == expected


# ----- Loader: ungrounded spans are rejected -----
def test_load_cuad_misaligned_offset_raises_value_error(tmp_path):
    """Checks load_cuad raises ValueError when answer_start points at the wrong text."""
    # "Beta" starts at 6, so claiming 0 makes the slice context[0:4] = "Alph"
    context = "Alpha Beta"
    qas = [{"id": "T__Parties_0", "answers": [{"text": "Beta", "answer_start": 0}]}]
    path = write_cuad_json(tmp_path, context, qas)

    with pytest.raises(ValueError, match="not grounded"):
        load_cuad(path)