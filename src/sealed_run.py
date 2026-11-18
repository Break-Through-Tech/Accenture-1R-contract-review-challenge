"""
Guards, provenance and the structure-only preflight for the one-time sealed test evaluation
(docs/final-test-evaluation.md). Nothing here computes a prediction or a metric.

How everything comes together:
classify_data_path(): "sealed" for test.json, "development" otherwise; refuses CUADv1.json and zip archives
check_mode(): the sealed file needs the explicit opt-in flag, and the flag needs the sealed file
check_sealed_preconditions(): before the sealed run reads test.json: clean committed code, a committed
                              passing preflight for the same file, and no earlier result
start_attempt(): appends to the attempt log before any data is read; a later attempt needs a recorded
                 crash reason
git_state() / environment(): commit, branch, dirty flag and library versions for the report
structure_report(): keys, counts, category names, grounding and overlap with the training file; no
                    contract text is returned
"""
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Literal

from src.labels.categories import CATEGORIES
from src.labels.loader import Contract

# File names that hold the CUAD test contracts. test.json is the sealed file; the others are never read
SEALED_DATA_NAME = "test.json"
FORBIDDEN_DATA_NAMES = frozenset({"cuadv1.json"})
FORBIDDEN_SUFFIXES = frozenset({".zip"})
ATTEMPTS_FILE, PREFLIGHT_FILE, REPORT_FILE = "run_attempts.json", "preflight.json", "report.json"
LIBRARIES = ("numpy", "scipy", "scikit-learn", "pandas", "joblib", "transformers", "tokenizers", "pydantic")


class SealedRunError(RuntimeError):
    """Raised when a guard refuses to run; raised before the sealed file is opened."""


# ----- What the data path is, and whether this invocation may read it -----
def classify_data_path(path: str | Path) -> Literal["sealed", "development"]:
    """
    Return "sealed" for a file named test.json and "development" for anything else. CUADv1.json and
    zip archives also contain the test contracts, so they raise SealedRunError in every mode.
    """
    path = Path(path)
    name = path.name.lower()
    if name in FORBIDDEN_DATA_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise SealedRunError(f"{path.name} contains the test contracts and is never read by this script")
    return "sealed" if name == SEALED_DATA_NAME else "development"


def check_mode(data_kind: str, sealed_flag: bool) -> None:
    """Refuse test.json without the opt-in flag, and the opt-in flag with any other file."""
    if data_kind == "sealed" and not sealed_flag:
        raise SealedRunError("test.json is sealed. Reading it needs --sealed-test-run, and the protocol "
                             "allows exactly one evaluation (docs/final-test-evaluation.md)")
    if data_kind != "sealed" and sealed_flag:
        raise SealedRunError("--sealed-test-run is only for data/cuad/test.json")


def check_sealed_output(output: Path, expected_output: Path) -> None:
    """The sealed run writes to one fixed folder, and development runs never write there."""
    if output.resolve() != expected_output.resolve():
        raise SealedRunError(f"the sealed run writes only to {expected_output}")


# ----- Preconditions for the sealed run -----
def check_sealed_preconditions(*, output: Path, git: Mapping, report_in_history: bool,
                               protocol_tracked: bool, preflight_tracked: bool,
                               data_sha256: str | None, preflight_only: bool) -> None:
    """
    Raise SealedRunError unless the sealed run may proceed:
    - no result exists, on disk or anywhere in git history;
    - tracked files have no uncommitted changes and the protocol is committed;
    - for the evaluation (not the preflight): a committed preflight that passed, for the same file hash.
    """
    if (output / REPORT_FILE).exists() or report_in_history:
        raise SealedRunError("a sealed result already exists; the evaluation is never run twice")
    if git.get("dirty"):
        raise SealedRunError("tracked files have uncommitted changes; commit first so the recorded commit "
                             "describes the code that ran")
    if not protocol_tracked:
        raise SealedRunError("the protocol docs/final-test-evaluation.md must be committed before the run")
    if preflight_only:
        return
    preflight_path = output / PREFLIGHT_FILE
    if not preflight_path.is_file() or not preflight_tracked:
        raise SealedRunError("run and commit the preflight (--preflight) before the evaluation")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("passed"):
        raise SealedRunError("the preflight did not pass; the protocol says the run does not start")
    if data_sha256 is not None and preflight.get("data", {}).get("sha256") != data_sha256:
        raise SealedRunError("the data file changed since the preflight")


def start_attempt(output: Path, *, commit: str, mode: str, resume_reason: str | None) -> list[dict]:
    """
    Record an attempt in the attempt log before any data is read, and return the whole log.
    A previous attempt without a result means a crash: continuing needs --resume-after-crash REASON,
    and a reason given when there was no crash is refused (so it can't be pre-filled).
    """
    log_path = output / ATTEMPTS_FILE
    attempts = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else []
    if attempts and not resume_reason:
        raise SealedRunError(f"{len(attempts)} earlier attempt(s) did not finish. Only a crash before any "
                             "metric was produced may be resumed: pass --resume-after-crash \"<what failed and "
                             "what was fixed>\"; it is recorded as a deviation")
    if not attempts and resume_reason:
        raise SealedRunError("--resume-after-crash is only for an attempt that crashed")
    attempts.append({"attempt": len(attempts) + 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                     "commit": commit, "mode": mode, "resume_reason": resume_reason})
    output.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
    return attempts


# ----- Provenance -----
def _git(root: Path, *args: str) -> str:
    """Run a git command in root and return its stripped output ('' if git fails)."""
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def git_state(root: Path) -> dict:
    """Commit, branch, and whether tracked files differ from that commit (untracked files don't count)."""
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=no")),
    }


def _relative(root: Path, path: Path) -> str | None:
    """path relative to root in POSIX form, or None when it lies outside the repository."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def git_tracked(root: Path, path: Path) -> bool:
    """True when path is tracked and identical to the committed version (False outside the repository)."""
    relative = _relative(root, path)
    if relative is None or not _git(root, "ls-files", relative):
        return False
    return not _git(root, "status", "--porcelain", "--", relative)


def git_history_has(root: Path, path: Path) -> bool:
    """True if any commit on any branch ever touched path (so a deleted result still counts)."""
    relative = _relative(root, path)
    return relative is not None and bool(_git(root, "log", "--all", "--format=%H", "--", relative))


def git_blob_id(root: Path, path: Path) -> str:
    """The git object id of a working file, with the same line-ending filters `git add` applies."""
    return _git(root, "hash-object", "--", str(path))


def last_commit_of(root: Path, path: Path) -> str:
    """The last commit that changed path ('' if none or outside the repository)."""
    relative = _relative(root, path)
    return _git(root, "log", "-1", "--format=%H", "--", relative) if relative is not None else ""


def environment() -> dict:
    """Python, platform and the versions of the libraries that decide the numbers."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "libraries": versions}


def text_file_sha256(path: str | Path) -> str:
    """
    SHA-256 of a text file with CRLF read as LF, so a Windows checkout (core.autocrlf) and a Linux
    checkout of the same commit give the same value.
    """
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a string's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_sha256(ids: Sequence[str]) -> str:
    """SHA-256 of newline-joined contract ids, as scripts/evaluate_word_char_tfidf.py records them."""
    return sha256_text("\n".join(ids))


# ----- Structure-only preflight -----
def _category_of(question_id: str) -> str:
    """'<title>__<Category>_<N>' -> '<Category>', exactly as src.labels.loader.load_cuad derives it."""
    return question_id.rsplit("__", 1)[-1].rsplit("_", 1)[0]


def _keys(items: Sequence[Mapping]) -> list[str]:
    """Sorted union of the keys of a list of dicts."""
    return sorted(set().union(*(item.keys() for item in items))) if items else []


def structure_report(raw: Mapping, partitions: Mapping[str, Sequence[Contract]]) -> dict:
    """
    Describe a CUAD-format JSON without computing any prediction or returning any contract text:
    keys, counts, category names derived from question ids, answer grounding, and how many
    contracts share a title or an exact text with each training-file partition.
    Category names outside CATEGORIES are listed (truncated), since projection would reject them.
    """
    entries = list(raw.get("data", []))
    paragraphs = [p for e in entries for p in e.get("paragraphs", [])]
    questions = [q for p in paragraphs for q in p.get("qas", [])]
    answers = [(p.get("context", ""), a) for p in paragraphs for q in p.get("qas", []) for a in q.get("answers", [])]

    # Categories per contract, from the first paragraph as the loader reads it
    per_contract = [Counter(_category_of(q.get("id", "")) for q in e["paragraphs"][0].get("qas", []))
                    for e in entries if e.get("paragraphs")]
    seen = set().union(*per_contract) if per_contract else set()
    unknown = sorted(seen - set(CATEGORIES))

    # Answers: grounded means context[start:start + len(text)] == text, the loader's own check
    grounded = sum(isinstance(a.get("answer_start"), int) and isinstance(a.get("text"), str)
                   and context[a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
                   for context, a in answers)
    empty = sum(not a.get("text") for _, a in answers)
    answer_keys = [(e.get("title"), _category_of(q.get("id", "")), a.get("answer_start"), a.get("text"))
                   for e in entries for p in e.get("paragraphs", []) for q in p.get("qas", [])
                   for a in q.get("answers", [])]

    # Overlap with the training file, by title and by exact text hash, per partition
    titles = [e.get("title") for e in entries]
    text_hashes = {sha256_text(e["paragraphs"][0].get("context", "")) for e in entries if e.get("paragraphs")}
    overlap = {name: {"titles": len(set(titles) & {c.contract_id for c in group}),
                      "texts": len(text_hashes & {sha256_text(c.text) for c in group})}
               for name, group in partitions.items()}

    return {
        "top_level_keys": sorted(raw.keys()),
        "version": raw.get("version"),
        "contracts": len(entries),
        "unique_titles": len(set(titles)),
        "paragraphs_per_contract": {str(k): v for k, v in sorted(Counter(len(e.get("paragraphs", []))
                                                                         for e in entries).items())},
        "entry_keys": _keys(entries), "paragraph_keys": _keys(paragraphs),
        "question_keys": _keys(questions), "answer_keys": _keys([a for _, a in answers]),
        "questions": len(questions),
        "questions_per_contract": {str(k): v for k, v in sorted(Counter(sum(c.values()) for c in per_contract)
                                                                .items())},
        "question_ids_with_numeric_suffix": sum(bool(re.search(r"_\d+$", q.get("id", ""))) for q in questions),
        "categories_seen": len(seen),
        "categories_missing": [c for c in CATEGORIES if c not in seen],
        "unknown_category_names": [name[:60] for name in unknown[:10]],
        "unknown_category_count": len(unknown),
        "contracts_with_each_category_once": sum(set(c) == set(CATEGORIES) and max(c.values()) == 1
                                                 for c in per_contract),
        "answered_questions": sum(bool(q.get("answers")) for q in questions),
        "answers": len(answers), "answers_grounded": int(grounded), "answers_empty": int(empty),
        "duplicate_answers": len(answer_keys) - len(set(answer_keys)),
        "contexts_with_crlf": sum("\r\n" in p.get("context", "") for p in paragraphs),
        "overlap_with_training_file": overlap,
    }


def preflight_failures(structure: Mapping, *, expected_contracts: int | None, loader_error: str | None,
                       blob_matches: bool | None) -> list[str]:
    """The pre-registered pass criteria; an empty list means the preflight passed."""
    failures = []
    if expected_contracts is not None and structure["contracts"] != expected_contracts:
        failures.append(f"expected {expected_contracts} contracts, found {structure['contracts']}")
    if set(structure["paragraphs_per_contract"]) != {"1"}:
        failures.append("every contract must have exactly one paragraph")
    if structure["unique_titles"] != structure["contracts"]:
        failures.append("contract titles are not unique")
    if structure["unknown_category_count"]:
        failures.append("question ids name categories outside the 41 CATEGORIES")
    if structure["answers_grounded"] != structure["answers"] or structure["answers_empty"]:
        failures.append("some answers are empty or not grounded in their context")
    for name in ("training", "calibration"):
        if any(structure["overlap_with_training_file"].get(name, {}).values()):
            failures.append(f"contracts overlap the {name} partition the model or thresholds were fitted on")
    if loader_error:
        failures.append(f"load_cuad failed ({loader_error})")
    if blob_matches is False:
        failures.append("the file differs from the committed blob pinned in the protocol")
    return failures
