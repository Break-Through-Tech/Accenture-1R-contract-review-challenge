import hashlib
import json
import shutil
import subprocess

import pytest

from src.labels.categories import CATEGORIES
from src.labels.loader import Contract
from src.sealed_run import (
    ATTEMPTS_FILE, PREFLIGHT_FILE, REPORT_FILE, SealedRunError, check_mode, check_sealed_output,
    check_sealed_preconditions, classify_data_path, git_blob_id, git_history_has, git_state, git_tracked,
    ids_sha256,
    preflight_failures, start_attempt, structure_report, text_file_sha256,
)

CLEAN_GIT = {"commit": "abc", "branch": "b", "dirty": False}


def _raw(contracts: list[tuple[str, str, list[tuple[str, int, str]]]], suffix: bool = True) -> dict:
    """A CUAD-format dict: (title, context, [(category, answer_start, text)]); every category gets a question."""
    data = []
    for title, context, answers in contracts:
        qas = []
        for category in CATEGORIES:
            found = [{"text": text, "answer_start": start} for name, start, text in answers if name == category]
            qas.append({"id": f"{title}__{category}" + ("_0" if suffix else ""), "question": "q",
                        "answers": found, "is_impossible": not found})
        data.append({"title": title, "paragraphs": [{"context": context, "qas": qas}]})
    return {"version": "v", "data": data}


def _contract(title: str, text: str) -> Contract:
    """A Contract with no spans, for overlap checks."""
    return Contract(contract_id=title, text=text, spans=[])


# ----- Data path classification -----
@pytest.mark.parametrize("name, kind", [("test.json", "sealed"), ("TEST.JSON", "sealed"),
                                        ("train_separate_questions.json", "development"),
                                        ("development_holdout.json", "development")])
def test_classify_data_path_marks_only_test_json_as_sealed(tmp_path, name, kind):
    """test.json (any case) is sealed; anything else is development. The file need not exist."""
    assert classify_data_path(tmp_path / name) == kind


@pytest.mark.parametrize("name", ["CUADv1.json", "cuadv1.json", "cuad-v1.zip", "data.ZIP"])
def test_classify_data_path_refuses_files_that_contain_the_test_contracts(tmp_path, name):
    """CUADv1.json and zip archives are refused in every mode, before anything is opened."""
    with pytest.raises(SealedRunError, match="never read"):
        classify_data_path(tmp_path / name)


@pytest.mark.parametrize("kind, flag, message", [("sealed", False, "--sealed-test-run"),
                                                 ("development", True, "only for")])
def test_check_mode_needs_the_flag_exactly_for_the_sealed_file(kind, flag, message):
    """test.json without the opt-in flag, or the flag with another file, is refused."""
    with pytest.raises(SealedRunError, match=message):
        check_mode(kind, flag)


@pytest.mark.parametrize("kind, flag", [("sealed", True), ("development", False)])
def test_check_mode_allows_matching_flag_and_file(kind, flag):
    """The two consistent combinations pass."""
    check_mode(kind, flag)


def test_check_sealed_output_allows_only_the_fixed_folder(tmp_path):
    """The sealed run writes to one folder."""
    check_sealed_output(tmp_path / "a", tmp_path / "a")
    with pytest.raises(SealedRunError):
        check_sealed_output(tmp_path / "b", tmp_path / "a")


# ----- Sealed-run preconditions -----
def _write_preflight(folder, passed=True, sha="d" * 64):
    """Write a preflight record into folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / PREFLIGHT_FILE).write_text(json.dumps({"passed": passed, "data": {"sha256": sha}}), encoding="utf-8")


def _preconditions(folder, **overrides):
    """Call check_sealed_preconditions with passing defaults, overridden per test."""
    arguments = dict(output=folder, git=CLEAN_GIT, report_in_history=False, protocol_tracked=True,
                     preflight_tracked=True, data_sha256="d" * 64, preflight_only=False)
    arguments.update(overrides)
    check_sealed_preconditions(**arguments)


def test_check_sealed_preconditions_pass_with_committed_code_and_passing_preflight(tmp_path):
    """Clean tree, committed protocol, committed passing preflight for the same hash: allowed."""
    _write_preflight(tmp_path)
    _preconditions(tmp_path)


@pytest.mark.parametrize("overrides, message", [
    ({"report_in_history": True}, "already exists"),
    ({"git": {**CLEAN_GIT, "dirty": True}}, "uncommitted"),
    ({"protocol_tracked": False}, "protocol"),
    ({"preflight_tracked": False}, "preflight"),
    ({"data_sha256": "e" * 64}, "changed"),
])
def test_check_sealed_preconditions_refuse_each_missing_condition(tmp_path, overrides, message):
    """Each precondition on its own blocks the run."""
    _write_preflight(tmp_path)
    with pytest.raises(SealedRunError, match=message):
        _preconditions(tmp_path, **overrides)


def test_check_sealed_preconditions_refuse_an_existing_report_on_disk(tmp_path):
    """A report.json in the folder means the run already happened."""
    _write_preflight(tmp_path)
    (tmp_path / REPORT_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(SealedRunError, match="already exists"):
        _preconditions(tmp_path)


def test_check_sealed_preconditions_refuse_a_failed_or_missing_preflight(tmp_path):
    """No preflight, or one that failed, blocks the evaluation."""
    with pytest.raises(SealedRunError, match="preflight"):
        _preconditions(tmp_path)
    _write_preflight(tmp_path, passed=False)
    with pytest.raises(SealedRunError, match="did not pass"):
        _preconditions(tmp_path)


def test_check_sealed_preconditions_skip_the_preflight_requirement_for_the_preflight_itself(tmp_path):
    """The preflight run needs a clean tree and the protocol, but no earlier preflight."""
    _preconditions(tmp_path, preflight_only=True, preflight_tracked=False)


# ----- Attempt log -----
def test_start_attempt_logs_the_first_attempt(tmp_path):
    """The first attempt is recorded before any data is read."""
    attempts = start_attempt(tmp_path, commit="abc", mode="sealed_test", resume_reason=None)
    assert [a["attempt"] for a in attempts] == [1]
    assert json.loads((tmp_path / ATTEMPTS_FILE).read_text(encoding="utf-8"))[0]["commit"] == "abc"


def test_start_attempt_refuses_a_second_attempt_without_a_crash_reason(tmp_path):
    """An unfinished earlier attempt can only be resumed with an explicit, recorded reason."""
    start_attempt(tmp_path, commit="abc", mode="sealed_test", resume_reason=None)
    with pytest.raises(SealedRunError, match="--resume-after-crash"):
        start_attempt(tmp_path, commit="abc", mode="sealed_test", resume_reason=None)
    attempts = start_attempt(tmp_path, commit="def", mode="sealed_test", resume_reason="tokenizer cache missing")
    assert [(a["attempt"], a["resume_reason"]) for a in attempts] == [(1, None), (2, "tokenizer cache missing")]


def test_start_attempt_refuses_a_crash_reason_when_nothing_crashed(tmp_path):
    """A resume reason can't be given in advance."""
    with pytest.raises(SealedRunError, match="only for an attempt that crashed"):
        start_attempt(tmp_path, commit="abc", mode="sealed_test", resume_reason="pre-filled")


# ----- Git provenance -----
@pytest.fixture
def repo(tmp_path):
    """A throwaway git repository with one committed file."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    def git(*args):
        """Run git in the throwaway repository."""
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "protocol.md").write_text("v1\n", encoding="utf-8")
    git("add", "protocol.md")
    git("commit", "-q", "-m", "init")
    return tmp_path


def test_git_state_and_tracking_follow_uncommitted_changes(repo):
    """A modified tracked file makes the tree dirty and no longer counts as committed."""
    assert git_state(repo)["dirty"] is False and git_tracked(repo, repo / "protocol.md")
    (repo / "protocol.md").write_text("v2\n", encoding="utf-8")
    assert git_state(repo)["dirty"] is True and not git_tracked(repo, repo / "protocol.md")
    assert not git_tracked(repo, repo / "untracked.json")


def test_git_blob_id_matches_the_index_entry(repo):
    """The working file's object id equals the blob git recorded for it (what the preflight compares)."""
    index = subprocess.run(["git", "ls-files", "-s", "protocol.md"], cwd=repo, capture_output=True, text=True,
                           check=True).stdout.split()[1]
    assert git_blob_id(repo, repo / "protocol.md") == index


def test_git_history_has_finds_a_deleted_file(repo):
    """A result committed and then deleted still counts as having existed."""
    (repo / "report.json").write_text("{}", encoding="utf-8")
    subprocess.run(["git", "add", "report.json"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "r"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "rm", "-q", "report.json"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "rm"], cwd=repo, check=True, capture_output=True)
    assert git_history_has(repo, repo / "report.json")
    assert not git_history_has(repo, repo / "never.json")


def test_text_file_sha256_is_the_same_for_crlf_and_lf_checkouts(tmp_path):
    """A Windows (CRLF) and a Linux (LF) checkout of one file hash alike: SHA-256 of b'a\\nb\\n'."""
    (tmp_path / "lf.txt").write_bytes(b"a\nb\n")
    (tmp_path / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    expected = hashlib.sha256(b"a\nb\n").hexdigest()
    assert text_file_sha256(tmp_path / "lf.txt") == text_file_sha256(tmp_path / "crlf.txt") == expected


def test_ids_sha256_golden_value():
    """Newline-joined ids, as the saved reports record them: SHA-256 of b'a\nb'."""
    assert ids_sha256(["a", "b"]) == "7e18f737311b2dc3b2f269dd78396b0351f14fb66efa879f768cb23181883c78"


# ----- Structure-only preflight -----
def test_structure_report_counts_and_grounding():
    """Counts, category names, suffixes and grounding are reported; no contract text is returned."""
    text = "The parties agree. Governed by Delaware law."
    raw = _raw([("C1", text, [("Governing Law", 19, "Governed by Delaware law.")]),
                ("C2", "Other text.", [("Parties", 0, "Wrong")])])
    report = structure_report(raw, {"training": [], "calibration": [], "evaluation": []})
    assert (report["contracts"], report["unique_titles"], report["paragraphs_per_contract"]) == (2, 2, {"1": 2})
    assert (report["questions"], report["question_ids_with_numeric_suffix"]) == (82, 82)
    assert (report["categories_seen"], report["unknown_category_count"], report["categories_missing"]) == (41, 0, [])
    assert (report["answers"], report["answers_grounded"], report["answers_empty"]) == (2, 1, 0)
    assert report["contracts_with_each_category_once"] == 2
    assert text not in json.dumps(report)


def test_structure_report_accepts_ids_without_a_numeric_suffix():
    """'<title>__<Category>' without _N still maps to the category, as the loader does."""
    report = structure_report(_raw([("C1", "x", [])], suffix=False), {})
    assert (report["question_ids_with_numeric_suffix"], report["unknown_category_count"]) == (0, 0)


def test_structure_report_lists_unknown_category_names():
    """A question id naming a category outside CATEGORIES is reported."""
    raw = _raw([("C1", "x", [])])
    raw["data"][0]["paragraphs"][0]["qas"].append({"id": "C1__Made Up Clause_0", "answers": []})
    report = structure_report(raw, {})
    assert report["unknown_category_names"] == ["Made Up Clause"]


def test_structure_report_counts_overlap_by_title_and_text_per_partition():
    """Shared titles and identical texts are counted against each training-file partition."""
    raw = _raw([("C1", "same text", []), ("C2", "unique", [])])
    partitions = {"training": [_contract("C1", "different")], "calibration": [_contract("X", "same text")]}
    overlap = structure_report(raw, partitions)["overlap_with_training_file"]
    assert overlap == {"training": {"titles": 1, "texts": 0}, "calibration": {"titles": 0, "texts": 1}}


def test_structure_report_counts_duplicate_answers():
    """The same answer listed twice for a category is counted once as a duplicate."""
    raw = _raw([("C1", "abc", [("Parties", 0, "abc"), ("Parties", 0, "abc")])])
    assert structure_report(raw, {})["duplicate_answers"] == 1


def _passing_structure(**changes) -> dict:
    """A structure report that meets every criterion, with some fields changed."""
    structure = {"contracts": 102, "unique_titles": 102, "paragraphs_per_contract": {"1": 102},
                 "unknown_category_count": 0, "answers": 10, "answers_grounded": 10, "answers_empty": 0,
                 "overlap_with_training_file": {"training": {"titles": 0, "texts": 0},
                                                "calibration": {"titles": 0, "texts": 0},
                                                "evaluation": {"titles": 0, "texts": 0}}}
    structure.update(changes)
    return structure


def test_preflight_failures_empty_for_a_passing_file():
    """A file meeting every criterion has no failures."""
    assert preflight_failures(_passing_structure(), expected_contracts=102, loader_error=None, blob_matches=True) == []


@pytest.mark.parametrize("changes, kwargs, message", [
    ({"contracts": 101, "unique_titles": 101}, {}, "expected 102"),
    ({"paragraphs_per_contract": {"1": 100, "2": 2}}, {}, "one paragraph"),
    ({"unique_titles": 100}, {}, "not unique"),
    ({"unknown_category_count": 1}, {}, "outside"),
    ({"answers_grounded": 9}, {}, "grounded"),
    ({"overlap_with_training_file": {"training": {"titles": 1, "texts": 0}}}, {}, "training partition"),
    ({"overlap_with_training_file": {"calibration": {"titles": 0, "texts": 2}}}, {}, "calibration partition"),
    ({}, {"loader_error": "ValueError"}, "load_cuad"),
    ({}, {"blob_matches": False}, "committed blob"),
])
def test_preflight_failures_flag_each_criterion(changes, kwargs, message):
    """Each pre-registered criterion fails the preflight on its own."""
    arguments = {"expected_contracts": 102, "loader_error": None, "blob_matches": True, **kwargs}
    failures = preflight_failures(_passing_structure(**changes), **arguments)
    assert len(failures) == 1 and message in failures[0]


def test_preflight_failures_allow_overlap_with_the_development_evaluation_partition():
    """Sharing contracts with the 82 development evaluation contracts did not shape the model, so it passes."""
    structure = _passing_structure(overlap_with_training_file={"evaluation": {"titles": 82, "texts": 82}})
    assert preflight_failures(structure, expected_contracts=None, loader_error=None, blob_matches=None) == []
