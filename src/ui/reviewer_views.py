"""
Pure helpers behind the reviewer page (src/ui/reviewer.py). They turn what ClauseTriageService
returns into the tables and text the page shows, and turn what a reviewer typed into a candidate
weight set or a passage location. No Gradio, no store access and no I/O except reading the CUAD
category descriptions once.

How everything comes together:
ranking_table(), contract_choices(): the ranked contract list (dataset contracts and uploads together)
rank_changes(), preview_summary(): what a weight preview moved, compared with the version shown
jobs_table(): the scoring status of uploads
scored_contract(), detail_summary(), detection_choices(), feedback_table(): one stored contract,
                 shown with the demo page's clause table and highlighted text (src/ui/report.py)
severity_table(), candidate_weight_set(), describe_changes(): the weights editor
queue_table(), item_choices(), item_summary(), item_segments(): the review queue
locate_phrase(), chunk_for_span(): where a clause the reviewer flags as missing is
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.api.jobs import JobRecord
from src.api.schemas import ContractDetailResponse, ContractRanking, RankedContract
from src.evidence.cues import CATEGORY_COL, CATEGORY_DESCRIPTIONS_PATH, DESCRIPTION_COL, _strip_label, normalize_category
from src.evidence.extractor import GroundedEvidence, Unknown
from src.evidence.schema import Evidence
from src.ingest.documents import Document
from src.labels.categories import CATEGORIES
from src.review.schema import ReviewItem, ReviewQueuePage
from src.risk.schema import SIGNAL_NAMES, WeightSet
from src.store.records import ChunkRecord, FeedbackRecord
from src.ui.report import ScoredContract, verdict

# ----- Labels the page uses -----
SOURCE_LABELS = {"cuad": "CUAD dataset", "pdf": "PDF upload", "docx": "Word upload", "txt": "Text file upload",
                 "pasted": "Pasted text"}
ALL_CONTRACTS, DATASET_ONLY, UPLOADS_ONLY = "All contracts", "CUAD dataset", "Uploads"
SOURCE_FILTERS = (ALL_CONTRACTS, DATASET_ONLY, UPLOADS_ONLY)
RANKING_COLUMNS = ["Rank", "Contract", "Source", "Verdict", "Triage score", "High", "Medium", "Low", "No quote",
                   "Pages", "ID"]
CHANGE_COLUMNS = ["Rank in preview", "Rank now", "Move", "Contract", "Score in preview", "Score now", "ID"]
JOB_COLUMNS = ["Submitted (UTC)", "Contract", "Source", "Status", "Problem"]
JOB_STATUS_LABELS = {"queued": "Waiting to be scored", "running": "Scoring now", "complete": "Done: in the ranking",
                     "failed": "Failed"}
FEEDBACK_COLUMNS = ["When (UTC)", "Reviewer", "Verdict", "Category", "Passage", "Note"]
SIGNAL_TITLES = {"category_severity": "Category severity", "one_sidedness": "One-sidedness",
                 "magnitude": "Magnitude", "contract_context": "Contract context"}
SEVERITY_COLUMNS = ["Category", "Severity (0 to 1)", "Reason"]
USES_DEFAULT = "(uses the default severity)"
QUEUE_COLUMNS = ["#", "Status", "Contract", "Category", "Probability", "Why it is here", "Pages",
                 "Verdicts so far", "ID"]
STATUS_LABELS = {"pending": "Needs review", "reviewed": "Reviewed", "conflict": "Reviewers disagree"}
REASON_LABELS = {"no_cue_match": "No quote found", "span_mismatch": "Quote failed its check",
                 "reviewer_conflict": "Reviewers disagree"}
VERDICT_MEANINGS = {"accept": "the clause is in this passage", "reject": "the clause is not in this passage",
                    "missing": "the clause is in this passage and the model missed it"}
QUOTE_LABEL, SEARCH_LABEL = "Model's quote", "Your search"


# ----- Ranking -----
def display_name(contract: RankedContract) -> str:
    """What the reviewer calls a contract: its file or dataset name, else its id."""
    return contract.source_name or contract.contract_id


def ranked_verdict(contract: RankedContract) -> str:
    """'Open first' with any High clause type, 'Review' with any Medium, else 'Low priority' (as on the demo page)."""
    if contract.high_count:
        return "Open first"
    return "Review" if contract.medium_count else "Low priority"


def filter_contracts(ranking: ContractRanking, source_filter: str = ALL_CONTRACTS) -> list[RankedContract]:
    """The ranked contracts from the dataset, from uploads, or both; ranks stay those of the whole ranking."""
    if source_filter == DATASET_ONLY:
        return [c for c in ranking.contracts if c.source_type == "cuad"]
    if source_filter == UPLOADS_ONLY:
        return [c for c in ranking.contracts if c.source_type != "cuad"]
    return list(ranking.contracts)


def ranking_table(ranking: ContractRanking, source_filter: str = ALL_CONTRACTS) -> pd.DataFrame:
    """One row per contract in rank order; the last column is the contract id the page selects by."""
    return pd.DataFrame([{
        "Rank": c.rank, "Contract": display_name(c), "Source": SOURCE_LABELS.get(c.source_type or "", "Stored"),
        "Verdict": ranked_verdict(c), "Triage score": round(c.triage_score, 3), "High": c.high_count,
        "Medium": c.medium_count, "Low": c.low_count, "No quote": c.unknown_count, "Pages": c.page_count or "",
        "ID": c.contract_id,
    } for c in filter_contracts(ranking, source_filter)], columns=RANKING_COLUMNS)


def contract_choices(ranking: ContractRanking, source_filter: str = ALL_CONTRACTS) -> list[tuple[str, str]]:
    """(label, contract id) pairs for the contract picker, in rank order."""
    return [(f"#{c.rank} · {display_name(c)}", c.contract_id) for c in filter_contracts(ranking, source_filter)]


def move_label(rank_now: int | None, rank_preview: int) -> str:
    """'up 2', 'down 1' or 'no change' (toward rank 1 is up); 'new' for a contract the current ranking lacks."""
    if rank_now is None:
        return "new"
    places = rank_now - rank_preview
    if places > 0:
        return f"up {places}"
    return f"down {-places}" if places < 0 else "no change"


def rank_changes(current: ContractRanking, preview: ContractRanking) -> pd.DataFrame:
    """Each previewed contract's rank and score under the preview next to the version shown, in preview order."""
    now = {c.contract_id: c for c in current.contracts}
    rows = []
    for contract in sorted(preview.contracts, key=lambda c: (c.rank, display_name(c))):
        before = now.get(contract.contract_id)
        rows.append({
            "Rank in preview": contract.rank, "Rank now": before.rank if before else None,
            "Move": move_label(before.rank if before else None, contract.rank), "Contract": display_name(contract),
            "Score in preview": round(contract.triage_score, 3),
            "Score now": round(before.triage_score, 3) if before else None, "ID": contract.contract_id,
        })
    return pd.DataFrame(rows, columns=CHANGE_COLUMNS)


def preview_summary(current: ContractRanking, preview: ContractRanking, changes: Sequence[str] = ()) -> str:
    """Markdown: how many contracts changed rank, the largest move, and what was edited. Says nothing was saved."""
    now = {c.contract_id: c.rank for c in current.contracts}
    moved = [(now[c.contract_id] - c.rank, c) for c in preview.contracts
             if c.contract_id in now and now[c.contract_id] != c.rank]
    lines = [f"**Preview only: nothing was saved.** Ranked {preview.ranked_out_of} contract(s) under the edited "
             f"weights, compared with version `{current.weight_version_id}`: {len(moved)} changed rank."]
    if moved:
        places, contract = max(moved, key=lambda pair: (abs(pair[0]), -pair[1].rank))
        lines.append(f"Largest move: **{display_name(contract)}**, {move_label(now[contract.contract_id], contract.rank)} "
                     f"(rank {now[contract.contract_id]} to {contract.rank}).")
    if changes:
        lines.append("What you changed: " + "; ".join(changes) + ".")
    return "\n\n".join(lines)


# ----- Scoring jobs -----
def jobs_table(jobs: Iterable[JobRecord], limit: int = 20) -> pd.DataFrame:
    """The most recent upload jobs (dataset contracts are left out), newest first, with any failure reason."""
    uploads = [job for job in jobs if job.source_type != "cuad"][:limit]
    return pd.DataFrame([{
        "Submitted (UTC)": job.submitted_at.strftime("%Y-%m-%d %H:%M:%S"),
        "Contract": job.source_name or job.contract_id,
        "Source": SOURCE_LABELS.get(job.source_type or "", ""), "Status": JOB_STATUS_LABELS[job.status],
        "Problem": job.error or "",
    } for job in uploads], columns=JOB_COLUMNS)


def jobs_signature(jobs: Iterable[JobRecord]) -> list[list[str]]:
    """What changes whenever any job is queued, starts, finishes or fails (to know when to refresh)."""
    return [[job.contract_id, job.status, job.updated_at.isoformat()] for job in jobs]


# ----- One contract -----
def scored_contract(detail: ContractDetailResponse) -> ScoredContract:
    """A stored contract as the demo page's ScoredContract, so its clause table and highlights are the same views."""
    contract = detail.contract
    document = Document(name=contract.source_name or contract.contract_id, text=contract.text,
                        page_starts=contract.page_starts or (), ocr=bool(contract.ocr_used))
    grounded = [GroundedEvidence(contract_id=contract.contract_id, chunk_index=e.chunk_index, category=e.category,
                                 evidence=Evidence(quote=e.quote, start=e.start, end=e.end)) for e in detail.evidence]
    unknown = [Unknown(contract_id=contract.contract_id, chunk_index=u.chunk_index, category=u.category,
                       reason=u.reason) for u in detail.unknowns]
    result = {"contract_id": contract.contract_id, "model_name": detail.model_name, "grounded": grounded,
              "unknown": unknown, "clause_scores": detail.clause_scores}
    return ScoredContract(document, result)


def detail_summary(detail: ContractDetailResponse, disclaimer: str) -> str:
    """Markdown headline: verdict, triage score and rank, rating counts, source, pages, model and weight version."""
    contract, score = detail.contract, detail.contract_score
    name = contract.source_name or contract.contract_id
    pages = (f"{contract.page_count} pages" + (" (read with OCR)" if contract.ocr_used else "")
             if contract.page_count else "no page numbers")
    return (
        f"## {name}\n"
        f"### {verdict(score)} · triage score {score.triage_score:.2f} · rank {detail.rank} of {detail.ranked_out_of}\n"
        f"**{score.high_count} High · {score.medium_count} Medium · {score.low_count} Low** clause types "
        f"({score.unknown_count} without a quote) · {SOURCE_LABELS.get(contract.source_type or '', 'stored')} · "
        f"{pages} · id `{contract.contract_id}`\n\n"
        f"Model `{detail.model_name}` (version `{detail.model_version}`) · weights `{detail.weight_version_id}` "
        f"(hash `{detail.weight_set_sha256[:12]}`)\n\n{disclaimer}"
    )


def _page_label(document: Document, start: int, end: int) -> str:
    """'page 3', 'pages 3-4' or '' for a span of a document without pages."""
    if not document.page_starts or start >= len(document.text):
        return ""
    first, last = document.page_of(start), document.page_of(max(start, end - 1))
    return f"page {first}" if first == last else f"pages {first}-{last}"


def detection_choices(detail: ContractDetailResponse) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """
    Every detection (grounded or Unknown), riskiest first, as (label, value) picker choices plus a
    Markdown description of each by value. The value is '<chunk_id>|<category>'.
    """
    item = scored_contract(detail)
    chunk_of = {chunk.chunk_index: chunk for chunk in detail.chunks}
    evidence = {(e.chunk_index, e.category): e for e in detail.evidence}
    threshold = {(p.chunk_index, p.category): p.threshold for p in detail.predictions}
    choices, views = [], {}
    for clause in sorted(detail.clause_scores, key=lambda c: (-c.composite_score, c.category, c.chunk_index)):
        chunk = chunk_of[clause.chunk_index]
        found = evidence.get((clause.chunk_index, clause.category))
        span = (found.start, found.end) if found else (chunk.start, chunk.end)
        where = _page_label(item.document, *span)
        value = f"{chunk.chunk_id}|{clause.category}"
        label = " · ".join(part for part in (clause.rating, clause.category, where,
                                             "quote" if found else "no quote", f"passage {clause.chunk_index}") if part)
        choices.append((label, value))
        body = (f"> {found.quote}" if found else
                "*No quote could be grounded for this detection. The passage the model read:*\n\n> "
                + chunk.text.replace("\n", "\n> "))
        views[value] = (f"**{clause.category}** · rated {clause.rating} (risk score {clause.composite_score:.2f}) · "
                        f"model probability {clause.probability:.2f} (threshold "
                        f"{threshold.get((clause.chunk_index, clause.category), float('nan')):.2f}) · "
                        f"{where or 'no page numbers'} · passage {clause.chunk_index}\n\n{body}")
    return choices, views


def feedback_table(records: Iterable[FeedbackRecord]) -> pd.DataFrame:
    """Recorded verdicts, newest first."""
    return pd.DataFrame([{
        "When (UTC)": record.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "Reviewer": record.reviewer,
        "Verdict": record.verdict, "Category": record.category, "Passage": record.chunk_id.rsplit(":", 1)[-1],
        "Note": record.note or "",
    } for record in sorted(records, key=lambda r: r.id, reverse=True)], columns=FEEDBACK_COLUMNS)


def locate_phrase(text: str, phrase: str) -> tuple[int, int, int] | None:
    """
    (start, end, count) of the first place phrase occurs in text, ignoring case and treating any run
    of whitespace as one space (so a phrase copied across a line break still matches); count is how
    many times it occurs. None when it doesn't occur or is blank.
    """
    words = phrase.split()
    if not words:
        return None
    matches = list(re.finditer(r"\s+".join(re.escape(word) for word in words), text, re.IGNORECASE))
    return (matches[0].start(), matches[0].end(), len(matches)) if matches else None


def chunk_for_span(chunks: Sequence[ChunkRecord], start: int, end: int) -> ChunkRecord | None:
    """The first chunk holding all of text[start:end], else the first one holding its start; None if neither."""
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_index)
    whole = next((chunk for chunk in ordered if chunk.start <= start and end <= chunk.end), None)
    return whole or next((chunk for chunk in ordered if chunk.start <= start < chunk.end), None)


# ----- Weights editor -----
class WeightEditError(ValueError):
    """An edit that can't become a weight set, with a message written for the reviewer."""


def severity_table(weight_set: WeightSet) -> pd.DataFrame:
    """All 41 categories with their severity under weight_set and the reason it gives, in CUAD order."""
    return pd.DataFrame([{
        "Category": category,
        "Severity (0 to 1)": weight_set.category_severity.get(category, weight_set.default_severity),
        "Reason": weight_set.severity_rationale.get(category, "") if category in weight_set.category_severity
                  else USES_DEFAULT,
    } for category in CATEGORIES], columns=SEVERITY_COLUMNS)


def _number(value: Any, label: str) -> float:
    """A finite number typed into the editor; WeightEditError naming the field otherwise."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise WeightEditError(f"{label} is empty; enter a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise WeightEditError(f"{label} must be a number (you entered {value!r}).") from None
    if not math.isfinite(number):
        raise WeightEditError(f"{label} must be a number (you entered {value!r}).")
    return number


def _severities(base: WeightSet, table: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    """
    Severities and reasons from the editor's table. A category keeps an explicit severity if the base
    version set one or the reviewer changed it from the base default; the rest keep following the default.
    """
    if list(table.columns[:3]) != SEVERITY_COLUMNS[:3] or sorted(table.iloc[:, 0]) != sorted(CATEGORIES):
        raise WeightEditError("The severity table must list each of the 41 CUAD categories once; "
                              "press Reset to start again from the version shown.")
    severity, rationale = dict(base.category_severity), {}
    for category, value, reason in table.iloc[:, :3].itertuples(index=False):
        number = _number(value, f"The severity of {category}")
        if category in base.category_severity or number != base.default_severity:
            severity[category] = number
            reason = "" if reason is None or reason == USES_DEFAULT else str(reason).strip()
            if reason:
                rationale[category] = reason
    return severity, rationale


_FIELD_TITLES = {"medium_threshold": "The Medium threshold", "high_threshold": "The High threshold",
                 "default_severity": "The default severity", "triage_top_k": "Clauses averaged",
                 "triage_max_weight": "The weight of the worst clause", "weight_version_id": "The version id",
                 "created_by": "Your name"}


def describe_weight_errors(error: ValidationError) -> str:
    """A WeightSet ValidationError in plain words, one sentence per problem."""
    sentences = []
    for item in error.errors():
        loc, kind, message = tuple(item.get("loc", ())), item.get("type", ""), str(item.get("msg", ""))
        if loc[:1] == ("signal_weights",) and len(loc) > 1:
            title = f"The {SIGNAL_TITLES.get(str(loc[1]), str(loc[1])).lower()} weight"
        elif loc[:1] == ("category_severity",) and len(loc) > 1:
            title = f"The severity of {loc[1]}"
        else:
            title = _FIELD_TITLES.get(str(loc[0]) if loc else "", "")
        total = re.search(r"must sum to 1\.0, got (\S+)", message)
        if total:
            sentences.append(f"The four signal weights must add up to 1 (they add up to {float(total.group(1)):.4g}).")
        elif "medium_threshold must be below high_threshold" in message:
            sentences.append("The Medium threshold must be below the High threshold.")
        elif kind in ("less_than_equal", "greater_than_equal", "less_than", "greater_than") and title:
            if loc[:1] == ("triage_top_k",):
                sentences.append(f"{title} must be at least 1.")
            else:
                sentences.append(f"{title} must be between 0 and 1 (you entered {item.get('input')}).")
        else:
            sentences.append(f"{title}: {message.removeprefix('Value error, ')}" if title
                             else message.removeprefix("Value error, "))
    return " ".join(sentences)


def describe_changes(base: WeightSet, candidate: WeightSet) -> list[str]:
    """Short plain-language list of the values candidate changes from base (empty when only the id differs)."""
    changes = []
    for name in SIGNAL_NAMES:
        if candidate.signal_weights[name] != base.signal_weights[name]:
            changes.append(f"{SIGNAL_TITLES[name].lower()} weight {base.signal_weights[name]:g} to "
                           f"{candidate.signal_weights[name]:g}")
    for field, title in (("medium_threshold", "Medium threshold"), ("high_threshold", "High threshold"),
                         ("default_severity", "default severity"), ("triage_top_k", "clauses averaged"),
                         ("triage_max_weight", "weight of the worst clause")):
        if getattr(candidate, field) != getattr(base, field):
            changes.append(f"{title} {getattr(base, field):g} to {getattr(candidate, field):g}")
    moved = [category for category in CATEGORIES
             if candidate.category_severity.get(category, candidate.default_severity)
             != base.category_severity.get(category, base.default_severity)]
    if moved:
        changes.append("severity of " + ", ".join(
            f"{category} {base.category_severity.get(category, base.default_severity):g} to "
            f"{candidate.category_severity.get(category, candidate.default_severity):g}" for category in moved))
    if dict(candidate.severity_rationale) != dict(base.severity_rationale):
        changes.append("reasons for " + ", ".join(
            category for category in CATEGORIES
            if candidate.severity_rationale.get(category) != base.severity_rationale.get(category)))
    return changes


def candidate_weight_set(base: WeightSet, *, weight_version_id: str, created_by: str, description: str,
                         signal_weights: Mapping[str, Any], medium_threshold: Any, high_threshold: Any,
                         default_severity: Any, triage_top_k: Any, triage_max_weight: Any,
                         severities: pd.DataFrame) -> WeightSet:
    """
    A new WeightSet: base with the editor's values. Context rules, phrase lists and duration rules are
    carried over from base unchanged. description defaults to one naming base, created_by and the
    changes. WeightEditError, in plain words, for anything the WeightSet would refuse.
    """
    data = base.model_dump(mode="json")
    data["signal_weights"] = {name: _number(signal_weights.get(name), f"The {SIGNAL_TITLES[name].lower()} weight")
                              for name in SIGNAL_NAMES}
    data["medium_threshold"] = _number(medium_threshold, "The Medium threshold")
    data["high_threshold"] = _number(high_threshold, "The High threshold")
    data["default_severity"] = _number(default_severity, "The default severity")
    top_k = _number(triage_top_k, "Clauses averaged")
    if top_k != int(top_k):
        raise WeightEditError(f"Clauses averaged must be a whole number (you entered {triage_top_k}).")
    data["triage_top_k"] = int(top_k)
    data["triage_max_weight"] = _number(triage_max_weight, "The weight of the worst clause")
    data["category_severity"], data["severity_rationale"] = _severities(base, severities)
    data["weight_version_id"], data["created_by"] = weight_version_id, created_by
    try:
        candidate = WeightSet.model_validate(data)
    except ValidationError as error:
        raise WeightEditError(describe_weight_errors(error)) from None
    # The description is part of the version, so it is settled before the final validation
    changes = describe_changes(base, candidate)
    data["description"] = description.strip() or (
        f"Derived from {base.weight_version_id} in the reviewer page by {created_by} on "
        f"{datetime.now().date().isoformat()}. Changed: {'; '.join(changes) or 'no value changed'}. Context rules, "
        "phrase lists and duration rules are those of the base version. Not legal advice.")
    return WeightSet.model_validate(data)


# ----- Review queue -----
@lru_cache(maxsize=1)
def category_descriptions() -> dict[str, str]:
    """Each canonical CUAD category's description, from data/cuad/category_descriptions.csv."""
    canonical = {normalize_category(category): category for category in CATEGORIES}
    frame = pd.read_csv(CATEGORY_DESCRIPTIONS_PATH, encoding="utf-8-sig")
    descriptions = {}
    for name, description in frame[[CATEGORY_COL, DESCRIPTION_COL]].itertuples(index=False):
        category = canonical.get(normalize_category(_strip_label(name, "Category")))
        if category:
            descriptions[category] = _strip_label(description, "Description")
    return descriptions


def _pages(item: ReviewItem) -> str:
    """'p. 3', 'pp. 3-4' or '' without page numbers."""
    if item.first_page is None:
        return ""
    return f"p. {item.first_page}" if item.first_page == item.last_page else f"pp. {item.first_page}-{item.last_page}"


def _votes(item: ReviewItem) -> str:
    """Each reviewer's current verdict, e.g. 'Ana: accept; Ben: reject'."""
    return "; ".join(f"{vote.reviewer}: {vote.verdict}" for vote in item.current_verdicts)


def queue_table(page: ReviewQueuePage) -> pd.DataFrame:
    """One row per item on the page, numbered from the page's offset; the last column is the item id."""
    return pd.DataFrame([{
        "#": page.offset + number, "Status": STATUS_LABELS[item.status],
        "Contract": item.source_name or item.contract_id, "Category": item.category,
        "Probability": round(item.probability, 3), "Why it is here": REASON_LABELS.get(item.reason, item.reason),
        "Pages": _pages(item), "Verdicts so far": _votes(item), "ID": item.item_id,
    } for number, item in enumerate(page.items, start=1)], columns=QUEUE_COLUMNS)


def item_choices(page: ReviewQueuePage) -> list[tuple[str, str]]:
    """(label, item id) pairs for the item picker, in queue order."""
    return [(f"{page.offset + number}. {item.source_name or item.contract_id} · {item.category}", item.item_id)
            for number, item in enumerate(page.items, start=1)]


def item_summary(item: ReviewItem) -> str:
    """Markdown with everything needed to decide one item: what, where, why it is queued, and what each verdict means."""
    threshold = f" (threshold {item.threshold:.2f})" if item.threshold is not None else ""
    where = _pages(item) or "no page numbers"
    description = category_descriptions().get(item.category, "")
    lines = [
        f"### {item.category}\n**Contract:** {item.source_name or item.contract_id} · {where} · passage "
        f"{item.chunk_index} (characters {item.chunk_start:,} to {item.chunk_end:,})",
        f"**What this category means (CUAD):** {description}" if description else "",
        f"**Why it is here:** {item.reason_text}",
        f"**Model probability:** {item.probability:.2f}{threshold} · **Status:** {STATUS_LABELS[item.status]}"
        + (f" · **Verdicts so far:** {_votes(item)}" if item.current_verdicts else ""),
    ]
    if item.evidence_quote is None:
        lines.append("There is no quote to check: the model did not point to a sentence. Read the passage below; "
                     "use **Highlight in passage** to find words such as a defined term.")
    lines.append("**Your options:** " + "; ".join(f"**{v}**: {VERDICT_MEANINGS[v]}" for v in item.allowed_verdicts)
                 + ".")
    return "\n\n".join(line for line in lines if line)


def item_segments(item: ReviewItem, search: str = "") -> list[tuple[str, str | None]]:
    """
    The item's passage as (segment, label) pairs for gr.HighlightedText: the model's quote (when it has
    one) and every match of the reviewer's search, which never overlaps the quote.
    """
    text, spans = item.chunk_text, []
    if item.evidence_start is not None and item.evidence_end is not None:
        spans.append((item.evidence_start - item.chunk_start, item.evidence_end - item.chunk_start, QUOTE_LABEL))
    words = search.split()
    if words:
        for match in re.finditer(r"\s+".join(re.escape(word) for word in words), text, re.IGNORECASE):
            if all(match.end() <= start or match.start() >= end for start, end, _ in spans):
                spans.append((match.start(), match.end(), SEARCH_LABEL))
    segments, position = [], 0
    for start, end, label in sorted(spans):
        if start > position:
            segments.append((text[position:start], None))
        segments.append((text[start:end], label))
        position = end
    if position < len(text):
        segments.append((text[position:], None))
    return segments
