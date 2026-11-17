import json

import pandas as pd
import pytest

from scripts.build_advisor_packet import (
    KEY_FILE,
    PACKET_FILE,
    Candidate,
    build_packet,
    write_packet,
)
from scripts.evaluate_advisor_agreement import flagged_wrong_category, main
from src.evidence.schema import Evidence
from src.risk.agreement import AdvisorRanking, read_packet_csv
from src.risk.schema import SIGNAL_NAMES, ClauseInput, WeightSet
from src.risk.scoring import score_contract_clauses

# Composite = category severity: Uncapped Liability 0.8 and Exclusivity 0.6 (High), Audit Rights 0.4 (Medium),
# Governing Law 0.1 (Low), Parties 0.0 (Low)
WEIGHTS = WeightSet(
    weight_version_id="e2e-test-v1",
    created_by="tests",
    signal_weights={name: 0.0 for name in SIGNAL_NAMES} | {"category_severity": 1.0},
    medium_threshold=0.35,
    high_threshold=0.55,
    category_severity={"Uncapped Liability": 0.8, "Exclusivity": 0.6, "Audit Rights": 0.4, "Governing Law": 0.1,
                       "Parties": 0.0},
    default_severity=0.2,
)
CATEGORIES = ["Uncapped Liability", "Exclusivity", "Audit Rights", "Governing Law", "Parties"]
# A perfect advisor: buckets as the model, ranks in descending score order
ANSWERS = {"Uncapped Liability": ("High", "1"), "Exclusivity": ("High", "2"), "Audit Rights": ("Medium", "3"),
           "Governing Law": ("Low", "4"), "Parties": ("Low", "5")}


@pytest.fixture
def packet_dir(tmp_path):
    """A five-clause packet over five synthetic contracts, written by the packet script's own writer."""
    texts, clauses = {}, []
    for i, category in enumerate(CATEGORIES):
        quote = f"Clause {i} says the {category.lower()} terms apply here in full."
        texts[f"contract-{i}"] = f"Recitals. {quote} Signatures."
        clauses.append(ClauseInput(contract_id=f"contract-{i}", chunk_index=0, category=category, probability=0.8,
                                   evidence=Evidence(quote=quote, start=10, end=10 + len(quote))))
    candidates = [Candidate(c, s) for c, s in zip(clauses, score_contract_clauses(clauses, WEIGHTS))]
    packet, key = build_packet(candidates, texts, {f"contract-{i}": [c] for i, c in enumerate(CATEGORIES)}, {},
                               WEIGHTS, seed=1)
    write_packet(tmp_path, packet, key)
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(WEIGHTS.model_dump_json(), encoding="utf-8")
    return tmp_path


def _fill(folder, answers=ANSWERS, notes: dict[str, str] | None = None) -> None:
    """Fill the packet's advisor columns by category, as the advisor would, and save it as returned.csv."""
    packet = read_packet_csv(folder / PACKET_FILE)
    packet["advisor_bucket"] = [answers[c][0] for c in packet["category"]]
    packet["advisor_rank"] = [answers[c][1] for c in packet["category"]]
    packet["advisor_notes"] = [(notes or {}).get(c, "") for c in packet["category"]]
    packet.sort_values("advisor_rank").to_csv(folder / "returned.csv", index=False)


def _run(folder, *extra: str):
    """Run the evaluation script on the packet folder and return (report, report.json contents)."""
    report = main(["--key", str(folder / KEY_FILE), "--returned", str(folder / "returned.csv"), "--reviewer",
                   "advisor", "--output", str(folder / "out"), "--n-boot", "200", *extra])
    return report, json.loads((folder / "out" / "report.json").read_text(encoding="utf-8"))


# ----- End to end -----
def test_perfect_advisor_end_to_end(packet_dir):
    """A perfect advisor gives rho = 1 and full bucket agreement, and every output file is written."""
    _fill(packet_dir)
    report, saved = _run(packet_dir)
    assert report.spearman_rank.value == pytest.approx(1.0) and report.exact_rate.value == 1.0
    assert saved["exact_rate"]["value"] == 1.0 and saved["inputs"]["weights"] is None and not saved["rescored"]
    assert saved["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 2]]
    assert {p.name for p in (packet_dir / "out").iterdir()} == {
        "report.json", "items.csv", "confusion_matrix.csv", "threshold_sweep.csv", "advisor_rankings.csv"}


def test_advisor_rankings_csv_has_the_table_columns(packet_dir):
    """advisor_rankings.csv holds one record per judged clause in the AdvisorRanking columns, ranking_set = packet id."""
    _fill(packet_dir)
    report, _ = _run(packet_dir)
    rankings = pd.read_csv(packet_dir / "out" / "advisor_rankings.csv")
    assert list(rankings.columns) == list(AdvisorRanking.model_fields) and len(rankings) == 5
    assert set(rankings["ranking_set"]) == {report.packet_id} and set(rankings["reviewer"]) == {"advisor"}


def test_rescoring_under_another_weight_set_needs_no_model(packet_dir):
    """--weights with a High threshold of 0.7 drops Exclusivity (0.6) to Medium; the same set reproduces the key."""
    _fill(packet_dir)
    stricter = WEIGHTS.model_copy(update={"weight_version_id": "e2e-test-v2", "high_threshold": 0.7})
    (packet_dir / "v2.json").write_text(stricter.model_dump_json(), encoding="utf-8")
    report, saved = _run(packet_dir, "--weights", str(packet_dir / "v2.json"))
    assert saved["weight_version_id"] == "e2e-test-v2" and saved["rescored"]
    assert report.exact_rate.value == 0.8 and report.high_recall.value == 0.5
    same, _ = _run(packet_dir, "--weights", str(packet_dir / "weights.json"))
    assert same.scores_reproduced is True and same.exact_rate.value == 1.0


def test_exclude_wrong_category_drops_flagged_clauses(packet_dir):
    """A clause whose note starts WRONG CATEGORY is listed, and dropped only with --exclude-wrong-category."""
    _fill(packet_dir, notes={"Exclusivity": "wrong category: this is a licence grant"})
    kept, saved = _run(packet_dir)
    assert kept.n_judged == 5 and len(saved["inputs"]["flagged_wrong_category"]) == 1
    dropped, saved = _run(packet_dir, "--exclude-wrong-category")
    assert dropped.n_judged == 4 and "Exclusivity" not in {i.category for i in dropped.items}


def test_malformed_returned_packet_fails_clearly(packet_dir):
    """A bucket that isn't Low, Medium or High stops the run with the item named."""
    _fill(packet_dir, answers=ANSWERS | {"Parties": ("Severe", "5")})
    with pytest.raises(ValueError, match="advisor_bucket must be Low, Medium or High"):
        _run(packet_dir)


@pytest.mark.parametrize("notes, expected", [
    ("WRONG CATEGORY", True), ("  wrong category - a licence", True), ("not the wrong category", False), ("", False),
])
def test_flagged_wrong_category_reads_the_note_prefix(notes, expected):
    """Only a note that starts with WRONG CATEGORY (any case) flags the clause."""
    assert flagged_wrong_category(notes) is expected
