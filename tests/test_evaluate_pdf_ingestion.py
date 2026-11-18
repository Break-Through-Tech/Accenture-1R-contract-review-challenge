import json
from pathlib import Path

import pytest

from scripts.evaluate_pdf_ingestion import (count_recovered, evaluate, main, normalize_whitespace,
                                            remove_whitespace, straighten_quotes)
from clausetriage.labels.loader import load_cuad
from clausetriage.labels.splits import split_by_contract
from tests.pdf_fixtures import write_pdf

CONTEXT = "This Agreement is governed by Delaware law. Either party may terminate."
PAGES = ["This Agreement is governed by Delaware law.", "Either party may terminate."]


def _write_cuad(path: Path) -> None:
    """A five-contract CUAD-format file; 'target' has two gold spans, the second crossing the PDF page break."""
    entries = []
    for index in range(5):
        title = "target" if index == 0 else f"other-{index}"
        answers = [("Governing Law", "governed by Delaware law"), ("Termination For Convenience", "law. Either party")]
        qas = [{"id": f"{title}__{category}_0", "answers": [{"text": text, "answer_start": CONTEXT.index(text)}]}
               for category, text in answers]
        entries.append({"title": title, "paragraphs": [{"context": CONTEXT, "qas": qas}]})
    path.write_text(json.dumps({"data": entries}), encoding="utf-8")


# ----- Text transforms -----
@pytest.mark.parametrize("function,text,expected", [
    (normalize_whitespace, "  a \n\n b\tc ", "a b c"),
    (remove_whitespace, " a \n b\tc", "abc"),
    (straighten_quotes, "‘a’ “b”", "'a' \"b\""),
])
def test_text_transforms_golden(function, text, expected):
    """Each transform gives the hand-computed result."""
    assert function(text) == expected


# ----- Counting recovered spans -----
def test_count_recovered_counts_each_level():
    """Hand-built case: one span per level, so the counts climb 1, 2, 3, 4."""
    extracted = "The Licensee\nshall pay’s fees. non-\nexclusive"
    spans = ["The Licensee",        # verbatim
             "Licensee shall pay",  # only after collapsing the line break to a space
             "non-exclusive",       # only once the wrap after the hyphen is dropped
             "pay's fees"]          # only with the curly apostrophe straightened
    assert count_recovered(extracted, spans) == {
        "verbatim": 1, "whitespace_normalized": 2, "ignoring_whitespace": 3,
        "ignoring_whitespace_and_quote_style": 4}


# ----- Whole report -----
def test_evaluate_reports_fidelity_for_a_rendered_contract(tmp_path):
    """Two-page PDF of CONTEXT: the page break breaks one span verbatim, whitespace normalization recovers it."""
    _write_cuad(tmp_path / "train.json")
    write_pdf(tmp_path / "target.pdf", PAGES)
    report = evaluate(tmp_path / "target.pdf", tmp_path / "train.json", "target")

    # "...law." + "\n\n" + "Either..." is 72 characters against the original 71
    assert (report.original_chars, report.extracted_chars, report.pages) == (71, 72, 2)
    assert not report.text_identical and report.whitespace_normalized_identical
    assert report.gold_spans == 2 and report.distinct_gold_spans == 2
    assert report.found == {"verbatim": 1, "whitespace_normalized": 2, "ignoring_whitespace": 2,
                            "ignoring_whitespace_and_quote_style": 2}
    assert report.missing_after_whitespace_normalization == []
    _, validation = split_by_contract(load_cuad(tmp_path / "train.json"))
    assert report.in_validation_split == ("target" in [c.contract_id for c in validation])


def test_evaluate_refuses_test_json(tmp_path):
    """The held-out test file is refused before it is opened."""
    write_pdf(tmp_path / "target.pdf", PAGES)
    with pytest.raises(ValueError, match="Refusing to read test.json"):
        evaluate(tmp_path / "target.pdf", tmp_path / "test.json", "target")


def test_evaluate_unknown_contract_raises(tmp_path):
    """A contract id that isn't in the file is a clear error."""
    _write_cuad(tmp_path / "train.json")
    write_pdf(tmp_path / "target.pdf", PAGES)
    with pytest.raises(ValueError, match="not in train.json"):
        evaluate(tmp_path / "target.pdf", tmp_path / "train.json", "missing")


def test_main_writes_json_report(tmp_path, capsys):
    """--output writes the same JSON that is printed."""
    _write_cuad(tmp_path / "train.json")
    write_pdf(tmp_path / "target.pdf", PAGES)
    report = main(["--pdf", str(tmp_path / "target.pdf"), "--cuad", str(tmp_path / "train.json"),
                   "--contract-id", "target", "--output", str(tmp_path / "report.json")])
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written == report.model_dump() == json.loads(capsys.readouterr().out)
