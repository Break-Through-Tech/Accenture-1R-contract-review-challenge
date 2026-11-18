"""
The reviewer page: one Gradio page over ClauseTriageService for the whole triage loop on stored data.
python -m src.api mounts it at /ui/ (src/api/app.py). It reads and writes only through the service,
the same methods the HTTP endpoints call, so everything shown here is what the API returns.

Tabs:
Contracts  add contracts (upload or paste; scored in the background, persisted, with status and
           failures shown), the ranked list of every stored contract (CUAD dataset and uploads), one
           contract's detail (score, rank, risky clauses with pages, quotes and signals, highlighted
           text, register downloads), and verdicts on its clauses or a clause the model missed
Weights    edit the weight version shown into a candidate, preview the re-ranking it gives (nothing is
           saved), or save it as a new version with the reviewer as its creator
Review     the review queue (Unknowns and reviewer conflicts) with each item's passage, and verdicts

How everything comes together:
ReviewerPage: the handlers, one method per page action, each returning the values of its outputs
              (tests call them directly; build_reviewer_app() wires them to components)
build_reviewer_app(): the gr.Blocks page for one service

Uploads are scored on one background thread owned by the page (run_job can be swapped, for example
to score synchronously in tests). A timer polls the job list and refreshes the ranking when a job
changes. Every score shown names its weight version; nothing on this page changes a stored version.
No authentication: names typed here are self-declared. Not legal advice.
"""
from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.api.schemas import MAX_NAME_CHARS, ContractRanking
from src.api.service import ClauseTriageService, ServiceError
from src.ingest.documents import SUPPORTED_SUFFIXES
from src.labels.categories import CATEGORIES
from src.register.export import page_of, register_file_stem, register_frame, to_json
from src.register.schema import ContractRegister
from src.review.schema import ReviewItem
from src.risk.schema import DEFAULT_WEIGHTS_PATH, SIGNAL_NAMES, V1_WEIGHTS_PATH, WeightSet, load_weight_set
from src.ui.report import CLAUSE_COLUMNS, clause_table, highlighted_text
from src.ui.reviewer_views import (ALL_CONTRACTS, CHANGE_COLUMNS, FEEDBACK_COLUMNS, JOB_COLUMNS, QUEUE_COLUMNS,
                                   QUOTE_LABEL, RANKING_COLUMNS, SEARCH_LABEL, SEVERITY_COLUMNS, SIGNAL_TITLES,
                                   SOURCE_FILTERS, VERDICT_MEANINGS, WeightEditError, candidate_weight_set,
                                   chunk_for_span, contract_choices, describe_changes, detail_summary,
                                   detection_choices, display_name, feedback_table, filter_contracts, item_choices,
                                   item_segments, item_summary, jobs_signature, jobs_table, locate_phrase,
                                   preview_summary, queue_table, rank_changes, ranking_table, scored_contract,
                                   severity_table)

RATING_COLORS = {"High": "#e5484d", "Medium": "#f5a524", "Low": "#46a758"}
PASSAGE_COLORS = {QUOTE_LABEL: "#3b82f6", SEARCH_LABEL: "#f5a524"}
STATUS_FILTERS = {"Needs review": "pending", "Reviewers disagree": "conflict", "Reviewed": "reviewed", "All": None}
QUEUE_PAGE_SIZE = 25
POLL_SECONDS = 2.0
VERDICTS = ("accept", "reject", "missing")
NAME_FIRST = "Type your name in **Your name** at the top of the page first; it is saved with what you record."
# Plain-language lead for the store's verdict refusals (the store's own reason follows it)
REFUSAL_HINTS = (
    ("'missing' is for clauses the model did not predict",
     "The model already detected this category in this passage, so choose Accept or Reject instead."),
    ("needs a prediction", "The model did not detect this category in this passage, so it can only be flagged as "
                           "missing (and then disputed with Reject)."),
    ("is earlier than", "You already gave a later verdict on this item."),
)
# Weight sets shipped in src/risk/weights/: set by the team and not validated by a legal expert
TEAM_WEIGHT_FILES = (DEFAULT_WEIGHTS_PATH, V1_WEIGHTS_PATH)


def _frame(value: Any, columns: list[str]) -> pd.DataFrame:
    """A table from Gradio (a DataFrame, or a {'headers', 'data'} dict from an API client) as a DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, dict):
        return pd.DataFrame(value.get("data") or [], columns=value.get("headers") or columns)
    return pd.DataFrame(value or [], columns=columns)


class ReviewerPage:
    """
    The reviewer page's actions over one ClauseTriageService. Each public method is one page action
    and returns the values (or gr.update()s) of its outputs, in the order build_reviewer_app wires them.
    """

    def __init__(self, service: ClauseTriageService, run_job: Callable[[str], Any] | None = None) -> None:
        """Wrap service. run_job(contract_id) scores a queued upload; default: on one background thread."""
        self.service = service
        self._executor = None if run_job else ThreadPoolExecutor(max_workers=1, thread_name_prefix="clausetriage-ui")
        self._run_job = run_job or (lambda contract_id: self._executor.submit(service.run_job, contract_id))
        self._team_hashes = {load_weight_set(path).content_sha256 for path in TEAM_WEIGHT_FILES if path.exists()}

    # ----- Header -----
    def disclaimer(self) -> str:
        """The prototype, unvalidated-weights and no-accounts notice shown at the top and with every contract."""
        default = self.service.default_weight_set
        if default.content_sha256 in self._team_hashes:
            status = (f"The default weights (`{default.weight_version_id}`) were set by the team and **have not been "
                      "validated by a legal expert**.")
        else:
            status = (f"The default weights (`{default.weight_version_id}`, created by {default.created_by}) were "
                      "given to the server as a file and are not one of the team's shipped weight sets.")
        return (f"> **Research prototype, not legal advice.** Ratings come from rules with adjustable weights. {status} "
                "Names typed on this page are self-declared; there are no accounts.")

    def version_choices(self) -> list[tuple[str, str]]:
        """(label, id) for every stored weight version, oldest first; the default is marked."""
        listing = self.service.list_weights()
        return [(f"{record.weight_version_id} · by {record.created_by}"
                 + (" · default" if record.weight_version_id == listing.default_weight_version_id else ""),
                 record.weight_version_id) for record in listing.versions]

    def version_note(self, version_id: str | None) -> str:
        """Who made the weight version shown, when, and whether it is one of the team's shipped weight sets."""
        try:
            detail = self.service.get_weights(version_id or self.service.default_weight_set.weight_version_id)
        except ServiceError as error:
            return f"**Can't show this weight version:** {error}"
        record, weight_set = detail.version, detail.weight_set
        if weight_set.content_sha256 in self._team_hashes:
            origin = "set by the team, **not validated by a legal expert**"
        else:
            origin = (f"created by **{record.created_by}** (a self-declared name); not validated by a "
                      "legal expert")
        default = " It is the default: new uploads are scored with it." if detail.is_default else ""
        return (f"Scores below use weight version **`{record.weight_version_id}`**: {origin}, stored "
                f"{record.created_at:%Y-%m-%d %H:%M} UTC, hash `{record.content_sha256[:12]}`.{default}")

    def start(self) -> tuple:
        """Page load: the stored weight versions (showing the default), its note, and the upload jobs."""
        default = self.service.default_weight_set.weight_version_id
        jobs = self.service.list_jobs()
        return (gr.update(choices=self.version_choices(), value=default), self.version_note(default),
                jobs_table(jobs), jobs_signature(jobs))

    # ----- Ranking -----
    def ranking_view(self, version_id: str | None, source_filter: str, selected: str | None) -> tuple:
        """The ranking note, the ranked table (dataset contracts and uploads together) and the contract picker."""
        try:
            ranking = self.service.list_ranked(weight_version_id=version_id or None)
        except ServiceError as error:
            return (f"**Can't show the ranking:** {error}", pd.DataFrame(columns=RANKING_COLUMNS),
                    gr.update(choices=[], value=None))
        choices = contract_choices(ranking, source_filter)
        ids = [value for _, value in choices]
        shown = len(filter_contracts(ranking, source_filter))
        note = (f"**{ranking.ranked_out_of} contract(s)** ranked under weights `{ranking.weight_version_id}` and "
                f"model `{ranking.model_version}`; rank 1 is the one to open first.")
        if source_filter != ALL_CONTRACTS:
            note += f" Showing the {shown} from: {source_filter} (ranks are among all of them)."
        if not ranking.contracts:
            note += ("\n\nNothing is stored yet. Upload a contract above, or load the CUAD validation contracts "
                     "with `python -m scripts.load_dataset_contracts` while the server is stopped.")
        value = selected if selected in ids else (ids[0] if ids else None)
        return note, ranking_table(ranking, source_filter), gr.update(choices=choices, value=value)

    def pick_ranked_row(self, evt: gr.SelectData) -> str | Any:
        """The contract id of the clicked ranking row (its last column)."""
        row = evt.row_value or []
        return row[-1] if row else gr.skip()

    # ----- Uploads -----
    def _job_line(self, name: str, job: Any, queued: bool) -> str:
        """One line of the submit message for an accepted upload."""
        if queued:
            return f"- **{name}**: queued for scoring. Its progress is in the table below; it joins the ranking when done."
        if job.status == "complete":
            return f"- **{name}**: already scored (the same text is stored), so it is already in the ranking."
        return f"- **{name}**: already being scored."

    def submit_contracts(self, paths: list[str] | None, pasted: str | None, paste_name: str | None) -> tuple:
        """
        Submit every uploaded file and the pasted text through the service and queue new ones for
        background scoring. Returns the message (one line per input, with the reason for any refused),
        the jobs table, and the cleared file and paste inputs (pasted text is kept if it was refused).
        """
        lines, keep_pasted = [], pasted
        for path in paths or []:
            name = Path(path).name
            try:
                job, queued = self.service.submit_upload(Path(path).read_bytes(), name)
            except ServiceError as error:
                lines.append(f"- **{name}**: not accepted. {error}")
                continue
            except OSError:
                lines.append(f"- **{name}**: not accepted. The uploaded file could not be read; upload it again.")
                continue
            lines.append(self._job_line(name, job, queued))
            if queued:
                self._run_job(job.contract_id)
        if pasted and pasted.strip():
            label = (paste_name or "").strip() or "Pasted text"
            try:
                job, queued = self.service.submit_text(pasted, paste_name)
            except ServiceError as error:
                lines.append(f"- **{label}**: not accepted. {error}")
            else:
                keep_pasted = ""
                lines.append(self._job_line(label, job, queued))
                if queued:
                    self._run_job(job.contract_id)
        message = "\n".join(lines) or "**Nothing to score.** Choose files, or paste contract text."
        return message, jobs_table(self.service.list_jobs()), gr.update(value=None), gr.update(value=keep_pasted)

    def poll_jobs(self, seen: list | None, version_id: str | None, source_filter: str, selected: str | None) -> tuple:
        """Timer tick: when any job changed, the new jobs table and a refreshed ranking; otherwise nothing."""
        jobs = self.service.list_jobs()
        signature = jobs_signature(jobs)
        if signature == seen:
            return gr.skip(), seen, gr.skip(), gr.skip(), gr.skip()
        return (jobs_table(jobs), signature, *self.ranking_view(version_id, source_filter, selected))

    # ----- One contract -----
    def _empty_contract(self, message: str = "") -> tuple:
        """The contract outputs with nothing selected."""
        return (message, pd.DataFrame(columns=CLAUSE_COLUMNS), gr.HighlightedText(value=[]),
                gr.update(value=None, interactive=False), gr.update(value=None, interactive=False),
                gr.update(choices=[], value=None), {}, "", pd.DataFrame(columns=FEEDBACK_COLUMNS))

    @staticmethod
    def _write_register(register: ContractRegister) -> tuple[str, str]:
        """The register as CSV and JSON files in a new temporary folder, byte for byte as the API serves them."""
        folder = Path(tempfile.mkdtemp(prefix="clausetriage-register-"))
        stem = register_file_stem(register.header.contract_id)
        csv_path, json_path = folder / f"{stem}.csv", folder / f"{stem}.json"
        csv_path.write_bytes(register_frame([register]).to_csv(index=False, lineterminator="\n").encode("utf-8"))
        json_path.write_bytes(to_json(register).encode("utf-8"))
        return str(csv_path), str(json_path)

    def contract_view(self, contract_id: str | None, version_id: str | None, show_harmless: bool) -> tuple:
        """
        One contract under the weight version shown: headline, risky clause table, highlighted text,
        register downloads (CSV, JSON), the clause picker for verdicts with each clause's description,
        the first clause's description, and the verdicts recorded on the contract.
        """
        if not contract_id:
            return self._empty_contract()
        try:
            detail = self.service.get_contract_detail(contract_id, weight_version_id=version_id or None)
            register = self.service.get_register(contract_id, weight_version_id=version_id or None)
            feedback = self.service.list_feedback(contract_id)
        except ServiceError as error:
            return self._empty_contract(f"**Can't show this contract:** {error}")
        item = scored_contract(detail)
        segments = highlighted_text(item)
        colors = {label: RATING_COLORS[label.split(":")[0]] for _, label in segments if label}
        csv_path, json_path = self._write_register(register)
        choices, views = detection_choices(detail)
        first = choices[0][1] if choices else None
        return (detail_summary(detail, self.disclaimer()), clause_table(item, show_harmless),
                gr.HighlightedText(value=segments, color_map=colors), gr.update(value=csv_path, interactive=True),
                gr.update(value=json_path, interactive=True), gr.update(choices=choices, value=first), views,
                views.get(first, "*This contract has no detected clauses to give a verdict on.*"),
                feedback_table(feedback))

    @staticmethod
    def detection_view(views: dict | None, value: str | None) -> str:
        """The description of the picked clause."""
        return (views or {}).get(value or "", "")

    # ----- Verdicts -----
    def _record(self, verdict: str, chunk_id: str, category: str, reviewer: str | None, note: str | None,
                where: str, model_version: str | None = None) -> tuple[bool, str]:
        """Submit one verdict through the service; (recorded, message) with the reason when refused."""
        if not (reviewer or "").strip():
            return False, f"**Not recorded.** {NAME_FIRST}"
        try:
            record = self.service.submit_feedback({"chunk_id": chunk_id, "category": category, "verdict": verdict,
                                                   "reviewer": reviewer, "note": note, "model_version": model_version})
        except ServiceError as error:
            hint = next((text for pattern, text in REFUSAL_HINTS if pattern in str(error)), "")
            return False, f"**Not recorded.** {hint + ' ' if hint else ''}Reason: {error}"
        return True, (f"Recorded **{record.verdict}** ({VERDICT_MEANINGS[record.verdict]}) for *{record.category}* "
                      f"{where}, by {record.reviewer} at {record.timestamp:%H:%M:%S} UTC. Scores and rankings do not "
                      "change.")

    def clause_verdict(self, verdict: str, contract_id: str | None, detection: str | None, reviewer: str | None,
                       note: str | None) -> tuple:
        """Accept or reject a detected clause of the contract shown; the message, its verdicts, and the note box."""
        if not contract_id or not detection:
            return "**Not recorded.** Pick a contract and one of its clauses first.", gr.skip(), note
        chunk_id, category = detection.rsplit("|", 1)
        recorded, message = self._record(verdict, chunk_id, category, reviewer, note,
                                         f"in passage {chunk_id.rsplit(':', 1)[-1]}")
        return message, self._feedback_table(contract_id), "" if recorded else note

    def accept_clause(self, contract_id, detection, reviewer, note) -> tuple:
        """Accept: the picked clause is there."""
        return self.clause_verdict("accept", contract_id, detection, reviewer, note)

    def reject_clause(self, contract_id, detection, reviewer, note) -> tuple:
        """Reject: the picked clause is not there."""
        return self.clause_verdict("reject", contract_id, detection, reviewer, note)

    def flag_missing(self, contract_id: str | None, category: str | None, phrase: str | None, reviewer: str | None,
                     note: str | None) -> tuple:
        """
        Flag a clause the model missed: find the reviewer's words in the contract, take the passage
        holding them, and record 'missing' for the category there. The message, verdicts, and note box.
        """
        if not contract_id:
            return "**Not recorded.** Pick a contract first.", gr.skip(), note
        if category not in CATEGORIES:
            return "**Not recorded.** Pick the category of the missing clause.", gr.skip(), note
        if not (phrase or "").strip():
            return ("**Not recorded.** Paste a few words from where the clause is, so the passage can be found.",
                    gr.skip(), note)
        try:
            detail = self.service.get_contract_detail(contract_id)
        except ServiceError as error:
            return f"**Not recorded.** {error}", gr.skip(), note
        found = locate_phrase(detail.contract.text, phrase)
        if found is None:
            return ("**Not recorded.** Those words are not in the contract text. Copy them from the highlighted "
                    "text above (line breaks and capitals don't matter).", gr.skip(), note)
        start, end, count = found
        chunk = chunk_for_span(detail.chunks, start, end)
        page = page_of(start, detail.contract.page_starts or ())
        where = f"in passage {chunk.chunk_index}" + (f" (page {page})" if page else "")
        recorded, message = self._record("missing", chunk.chunk_id, category, reviewer, note, where)
        if recorded and count > 1:
            message += f" Those words occur {count} times; the first was used."
        return message, self._feedback_table(contract_id), "" if recorded else note

    def _feedback_table(self, contract_id: str) -> pd.DataFrame:
        """The verdicts recorded on a contract, newest first."""
        return feedback_table(self.service.list_feedback(contract_id))

    # ----- Weights -----
    def editor_view(self, version_id: str | None) -> tuple:
        """
        The editor filled from the weight version shown: note, the four signal weights, Medium and High
        thresholds, default severity, clauses averaged, weight of the worst clause, severity table,
        an empty new id and description, and an empty preview table.
        """
        try:
            weight_set = self.service.get_weights(version_id or self.service.default_weight_set.weight_version_id
                                                  ).weight_set
        except ServiceError as error:
            return (f"**Can't load this weight version:** {error}", *([gr.skip()] * 12),
                    pd.DataFrame(columns=CHANGE_COLUMNS))
        note = (f"Editing starts from **`{weight_set.weight_version_id}`**, the version shown at the top. Its "
                f"{len(weight_set.cooccurrence_rules)} contract-context rules, phrase lists and "
                f"{len(weight_set.magnitude_rules)} duration rules are kept as they are (edit those through the API's "
                "weight JSON). The team's v2 proposal uses five severity tiers: None 0, Low 0.25, Medium 0.5, "
                "High 0.75, Critical 1.0.")
        return (note, *(weight_set.signal_weights[name] for name in SIGNAL_NAMES), weight_set.medium_threshold,
                weight_set.high_threshold, weight_set.default_severity, weight_set.triage_top_k,
                weight_set.triage_max_weight, severity_table(weight_set), "", "", pd.DataFrame(columns=CHANGE_COLUMNS))

    def _candidate(self, version_id: str | None, new_id: str | None, reviewer: str | None, description: str | None,
                   values: tuple, saving: bool) -> tuple[WeightSet, WeightSet]:
        """(base, candidate) from the editor; WeightEditError or ServiceError with a reviewer-readable reason."""
        base = self.service.get_weights(version_id or self.service.default_weight_set.weight_version_id).weight_set
        name, new_id = (reviewer or "").strip(), (new_id or "").strip()
        if saving and not name:
            raise WeightEditError(NAME_FIRST + " It is recorded as the new version's creator.")
        if len(name) > MAX_NAME_CHARS:
            raise WeightEditError(f"Your name is longer than {MAX_NAME_CHARS} characters.")
        if saving and not new_id:
            raise WeightEditError("Give the new version an id, for example `v3-team-draft`. Existing versions "
                                  "never change, so the id must be new.")
        *signals, medium, high, default_severity, top_k, max_weight, severities = values
        candidate = candidate_weight_set(
            base, weight_version_id=new_id or f"{base.weight_version_id}-preview", created_by=name or "preview",
            description=description or "", signal_weights=dict(zip(SIGNAL_NAMES, signals)), medium_threshold=medium,
            high_threshold=high, default_severity=default_severity, triage_top_k=top_k,
            triage_max_weight=max_weight, severities=_frame(severities, SEVERITY_COLUMNS))
        return base, candidate

    def preview_weights(self, version_id, new_id, reviewer, description, *values) -> tuple:
        """
        Re-rank every stored contract under the edited weights from cached probabilities, saving
        nothing, and compare with the ranking under the version shown. Returns the message and the table.
        """
        try:
            base, candidate = self._candidate(version_id, new_id, reviewer, description, values, saving=False)
            current: ContractRanking = self.service.list_ranked(weight_version_id=base.weight_version_id)
            preview = self.service.preview(candidate)
        except (WeightEditError, ServiceError) as error:
            return f"**Can't preview:** {error}", pd.DataFrame(columns=CHANGE_COLUMNS)
        return preview_summary(current, preview, describe_changes(base, candidate)), rank_changes(current, preview)

    def save_weights(self, version_id, new_id, reviewer, description, *values) -> tuple:
        """
        Store the edited weights as a new version created by the reviewer (scoring every stored contract
        with it) and switch the page to it. Never changes an existing version. Returns the message and
        the version picker (unchanged when refused).
        """
        try:
            base, candidate = self._candidate(version_id, new_id, reviewer, description, values, saving=True)
            created = self.service.create_weights(candidate)
        except (WeightEditError, ServiceError) as error:
            return f"**Not saved:** {error}", gr.skip()
        new_id = created.version.weight_version_id
        if created.created:
            message = (f"Saved new weight version **`{new_id}`**, created by {created.version.created_by}. "
                       f"{created.rescored_contracts} stored contract(s) were scored with it from cached "
                       f"probabilities. `{base.weight_version_id}` is unchanged. The page now shows `{new_id}`.")
        else:
            message = (f"Version `{new_id}` already exists with exactly these values, so nothing was saved. "
                       f"The page now shows it.")
        return message, gr.update(choices=self.version_choices(), value=new_id)

    # ----- Review queue -----
    def contract_filter_choices(self) -> list[tuple[str, str]]:
        """('All contracts', '') then every stored contract in rank order under the default weights."""
        try:
            ranking = self.service.list_ranked()
        except ServiceError:
            return [(ALL_CONTRACTS, "")]
        return [(ALL_CONTRACTS, ""), *((display_name(c), c.contract_id) for c in ranking.contracts)]

    def queue_view(self, status_label: str, contract_filter: str | None, page_number: float | None,
                   current_item: str | None) -> tuple:
        """
        One page of the review queue under the filters: counts note, table, item picker (keeping the
        current item if it is still listed, else the first) and the items by id.
        """
        size = QUEUE_PAGE_SIZE
        page_number = max(1, int(page_number or 1))
        try:
            page = self.service.review_queue(contract_id=contract_filter or None,
                                             status=STATUS_FILTERS.get(status_label), offset=(page_number - 1) * size,
                                             limit=size)
        except ServiceError as error:
            return (f"**Can't show the queue:** {error}", pd.DataFrame(columns=QUEUE_COLUMNS),
                    gr.update(choices=[], value=None), {})
        counts = page.status_counts
        pages = max(1, math.ceil(page.total / size))
        note = (f"**{counts['pending']} need review · {counts['conflict']} where reviewers disagree · "
                f"{counts['reviewed']} reviewed**" + (" for this contract" if contract_filter else "")
                + f". Page {page_number} of {pages}; {page.total} item(s) match the filter.")
        if not page.items:
            note += "\n\nNothing to show here." + (" Try **All**, or another page." if page.total else "")
        items = {item.item_id: item.model_dump(mode="json") for item in page.items}
        value = current_item if current_item in items else next(iter(items), None)
        return note, queue_table(page), gr.update(choices=item_choices(page), value=value), items

    def review_tab_view(self, status_label, contract_filter, page_number, current_item) -> tuple:
        """Opening the Review tab: refreshed contract filter choices, then the queue."""
        choices = self.contract_filter_choices()
        contract_filter = contract_filter if contract_filter in {value for _, value in choices} else ""
        return (gr.update(choices=choices, value=contract_filter),
                *self.queue_view(status_label, contract_filter, page_number, current_item))

    def pick_queue_row(self, evt: gr.SelectData) -> str | Any:
        """The item id of the clicked queue row (its last column)."""
        row = evt.row_value or []
        return row[-1] if row else gr.skip()

    def item_view(self, item_id: str | None, items: dict | None, search: str | None) -> tuple:
        """The picked item's summary, its passage (quote and search matches highlighted) and which verdict buttons show."""
        item_data = (items or {}).get(item_id or "")
        if item_data is None:
            return ("*Pick an item in the queue.*", gr.HighlightedText(value=[]),
                    *(gr.update(visible=False) for _ in VERDICTS))
        item = ReviewItem.model_validate(item_data)
        return (item_summary(item), gr.HighlightedText(value=item_segments(item, search or ""), color_map=PASSAGE_COLORS),
                *(gr.update(visible=verdict in item.allowed_verdicts) for verdict in VERDICTS))

    def item_verdict(self, verdict: str, item_id: str | None, items: dict | None, reviewer: str | None,
                     note: str | None, status_label: str, contract_filter: str | None, page_number) -> tuple:
        """
        Record a verdict on the picked queue item, then re-read the queue. Returns the message, the
        queue outputs (see queue_view) and the note box (cleared once recorded).
        """
        item_data = (items or {}).get(item_id or "")
        if item_data is None:
            recorded, message = False, "**Not recorded.** Pick an item in the queue first."
        else:
            item = ReviewItem.model_validate(item_data)
            recorded, message = self._record(
                verdict, item.chunk_id, item.category, reviewer, note,
                f"in passage {item.chunk_index} of {item.source_name or item.contract_id}", item.model_version)
        queue = self.queue_view(status_label, contract_filter, page_number, item_id)
        return (message, *queue, "" if recorded else note)

    def accept_item(self, *args) -> tuple:
        """Accept: the queue item's clause is in its passage."""
        return self.item_verdict("accept", *args)

    def reject_item(self, *args) -> tuple:
        """Reject: the queue item's clause is not in its passage."""
        return self.item_verdict("reject", *args)

    def missing_item(self, *args) -> tuple:
        """Missing: the clause is in the passage and the model did not predict it."""
        return self.item_verdict("missing", *args)


REVIEW_NOTE = (
    "**What a verdict does.** It is saved in this server's local database with your name and the time, and the "
    "history is never edited. It does **not** change any score, rating or ranking, and it does **not** retrain the "
    "model. Verdicts on the CUAD validation contracts are left out of training exports on purpose (the model is "
    "evaluated on those contracts), and uploaded contracts are left out unless the team adds them deliberately. "
    "Two reviewers who disagree on an item are shown as a conflict; nothing is settled by majority."
)


def build_reviewer_app(service: ClauseTriageService, run_job: Callable[[str], Any] | None = None) -> gr.Blocks:
    """The reviewer page for service: header, then the Contracts, Weights and Review tabs (see the module docstring)."""
    page = ReviewerPage(service, run_job)
    default_version = service.default_weight_set.weight_version_id

    with gr.Blocks(title="ClauseTriage reviewer") as app:
        gr.Markdown("# ClauseTriage reviewer\nRank contracts for review, check the clauses behind each score, try "
                    "other risk weights, and record verdicts on what the model found.\n\n" + page.disclaimer())
        with gr.Row():
            reviewer = gr.Textbox(label="Your name", max_length=MAX_NAME_CHARS, scale=1,
                                  info="Saved with your verdicts and any weight version you create. No login.")
            version = gr.Dropdown(label="Weight version shown", choices=page.version_choices(), value=default_version,
                                  scale=2, info="Every score on the page uses this version. Switching never changes it.")
        version_note = gr.Markdown()

        with gr.Tabs():
            # Contracts: add, rank, inspect, and give verdicts on one contract's clauses
            with gr.Tab("Contracts"):
                with gr.Accordion("Add contracts", open=False):
                    with gr.Row():
                        files = gr.File(label="Contracts (PDF, Word or text)", file_count="multiple",
                                        file_types=list(SUPPORTED_SUFFIXES), type="filepath")
                        with gr.Column():
                            pasted = gr.Textbox(label="Or paste contract text", lines=6)
                            paste_name = gr.Textbox(label="Name for the pasted text (optional)",
                                                    max_length=MAX_NAME_CHARS)
                    submit = gr.Button("Score and add to the ranking", variant="primary")
                    submit_message = gr.Markdown()
                    jobs = gr.Dataframe(headers=JOB_COLUMNS, label="Uploads and their scoring status (newest first)",
                                        interactive=False, wrap=True)
                ranking_note = gr.Markdown()
                source_filter = gr.Radio(list(SOURCE_FILTERS), value=ALL_CONTRACTS, label="Show")
                ranking = gr.Dataframe(headers=RANKING_COLUMNS, label="Contracts, riskiest first (click a row to open it)",
                                       interactive=False, max_height=420)
                with gr.Row():
                    selected = gr.Dropdown(label="Contract", choices=[], scale=3)
                    show_harmless = gr.Checkbox(label="Show harmless clause types (Parties, dates)", value=False,
                                                scale=1)
                summary = gr.Markdown()
                with gr.Row():
                    # Always rendered and enabled only once a contract is shown. Hiding both and then
                    # showing them in one update hangs the browser in Gradio 6.17.3: a Row whose
                    # children all start hidden never finishes rendering when several appear together.
                    csv_download = gr.DownloadButton("Clause register (CSV)", interactive=False)
                    json_download = gr.DownloadButton("Clause register (JSON)", interactive=False)
                clauses = gr.Dataframe(headers=CLAUSE_COLUMNS, label="Risky clauses (worst per type)", wrap=True,
                                       interactive=False)
                text = gr.HighlightedText(label="Contract text with detected clauses highlighted", combine_adjacent=False)
                with gr.Accordion("Verdicts on this contract", open=False):
                    gr.Markdown(REVIEW_NOTE)
                    detection = gr.Dropdown(label="A detected clause", choices=[])
                    detection_note = gr.Markdown()
                    clause_note = gr.Textbox(label="Note (optional)", lines=1)
                    with gr.Row():
                        accept_clause = gr.Button("Accept: the clause is there")
                        reject_clause = gr.Button("Reject: it is not")
                    gr.Markdown("**A clause the model missed:** pick its category and paste a few words from it.")
                    with gr.Row():
                        missing_category = gr.Dropdown(label="Category", choices=list(CATEGORIES))
                        missing_phrase = gr.Textbox(label="Words from the clause", scale=2)
                    flag_missing = gr.Button("Flag as missing")
                    feedback_message = gr.Markdown()
                    contract_feedback = gr.Dataframe(headers=FEEDBACK_COLUMNS, label="Verdicts on this contract",
                                                     interactive=False, wrap=True)

            # Weights: edit the version shown, preview the re-ranking, save as a new version
            with gr.Tab("Weights"):
                editor_note = gr.Markdown()
                gr.Markdown("**Signal weights** (each between 0 and 1; the four must add up to 1). A clause's risk "
                            "score is the weighted sum of its four signals.")
                with gr.Row():
                    signals = [gr.Number(label=SIGNAL_TITLES[name], minimum=0, maximum=1, step=0.05)
                               for name in SIGNAL_NAMES]
                gr.Markdown("**Ratings and contract ranking.** A clause rates Medium from the Medium threshold and High "
                            "from the High threshold. A contract's triage score mixes its worst clause with the average "
                            "of its worst few clause types.")
                with gr.Row():
                    medium = gr.Number(label="Medium threshold", minimum=0, maximum=1, step=0.025)
                    high = gr.Number(label="High threshold", minimum=0, maximum=1, step=0.025)
                    top_k = gr.Number(label="Clauses averaged", precision=0, minimum=1)
                    max_weight = gr.Number(label="Weight of the worst clause", minimum=0, maximum=1, step=0.05)
                    default_severity = gr.Number(label="Default severity", minimum=0, maximum=1, step=0.05,
                                                 info="For categories without their own severity")
                severities = gr.Dataframe(headers=SEVERITY_COLUMNS, label="Severity of each category (0 to 1)",
                                          interactive=True, static_columns=[0], wrap=True, max_height=420,
                                          datatype=["str", "number", "str"])
                with gr.Row():
                    new_id = gr.Textbox(label="New version id", placeholder="for example v3-team-draft",
                                        info="Letters, digits, '.', '_' and '-'. Existing versions never change.")
                    description = gr.Textbox(label="Description (optional)",
                                             info="Left empty, it says which version it came from, who made it and "
                                                  "what changed.")
                with gr.Row():
                    preview = gr.Button("Preview ranking (saves nothing)")
                    save = gr.Button("Save as new version", variant="primary")
                    reset = gr.Button("Reset to the version shown")
                weights_message = gr.Markdown()
                changes = gr.Dataframe(headers=CHANGE_COLUMNS, label="Preview: ranking under the edited weights",
                                       interactive=False, max_height=420)

            # Review: the queue of Unknowns and conflicts, one passage at a time
            with gr.Tab("Review") as review_tab:
                gr.Markdown("Items the model could not back up with a quote (Unknowns) and items where reviewers "
                            "disagree. Read the passage and record whether the clause is there.\n\n" + REVIEW_NOTE)
                with gr.Row():
                    status_filter = gr.Radio(list(STATUS_FILTERS), value="Needs review", label="Show", scale=2)
                    contract_filter = gr.Dropdown(label="Contract", choices=[(ALL_CONTRACTS, "")], value="", scale=2)
                    page_number = gr.Number(label="Page", value=1, precision=0, minimum=1, scale=1)
                    refresh = gr.Button("Refresh", scale=1)
                queue_note = gr.Markdown()
                queue = gr.Dataframe(headers=QUEUE_COLUMNS, label="Review queue (click a row to open it)",
                                     interactive=False, wrap=True, max_height=360)
                items = gr.State({})
                item = gr.Dropdown(label="Item", choices=[])
                item_note = gr.Markdown()
                search = gr.Textbox(label="Highlight in passage", placeholder="a word or phrase to find")
                passage = gr.HighlightedText(label="Passage", combine_adjacent=False, color_map=PASSAGE_COLORS)
                verdict_note = gr.Textbox(label="Note (optional)", lines=1)
                with gr.Row():
                    accept_item = gr.Button("Accept: the clause is there", visible=False)
                    reject_item = gr.Button("Reject: it is not", visible=False)
                    missing_item = gr.Button("Missing: it is there, the model missed it", visible=False)
                verdict_message = gr.Markdown()

        jobs_seen, detection_views = gr.State(None), gr.State({})
        timer = gr.Timer(POLL_SECONDS)

        # Wiring: what each output list is, then the events
        ranking_outputs = [ranking_note, ranking, selected]
        contract_inputs = [selected, version, show_harmless]
        contract_outputs = [summary, clauses, text, csv_download, json_download, detection, detection_views,
                            detection_note, contract_feedback]
        editor_outputs = [editor_note, *signals, medium, high, default_severity, top_k, max_weight, severities,
                          new_id, description, changes]
        weight_inputs = [version, new_id, reviewer, description, *signals, medium, high, default_severity, top_k,
                         max_weight, severities]
        queue_inputs = [status_filter, contract_filter, page_number, item]
        queue_outputs = [queue_note, queue, item, items]
        item_inputs = [item, items, search]
        item_outputs = [item_note, passage, accept_item, reject_item, missing_item]
        verdict_inputs = [item, items, reviewer, verdict_note, status_filter, contract_filter, page_number]

        def show_version(event):
            """Chain the refresh of everything that depends on the weight version shown after event."""
            return (event.then(page.ranking_view, [version, source_filter, selected], ranking_outputs)
                    .then(page.contract_view, contract_inputs, contract_outputs)
                    .then(page.editor_view, [version], editor_outputs))

        show_version(app.load(page.start, None, [version, version_note, jobs, jobs_seen]))
        # A switch by the reviewer or by a successful save (a refused save leaves the picker alone)
        show_version(version.change(page.version_note, [version], [version_note]))
        source_filter.change(page.ranking_view, [version, source_filter, selected], ranking_outputs)
        ranking.select(page.pick_ranked_row, None, selected)
        selected.change(page.contract_view, contract_inputs, contract_outputs, api_name="contract")
        show_harmless.change(page.contract_view, contract_inputs, contract_outputs)
        submit.click(page.submit_contracts, [files, pasted, paste_name], [submit_message, jobs, files, pasted],
                     api_name="submit_contracts")
        timer.tick(page.poll_jobs, [jobs_seen, version, source_filter, selected],
                   [jobs, jobs_seen, *ranking_outputs], show_progress="hidden")
        detection.change(page.detection_view, [detection_views, detection], detection_note)
        clause_feedback_inputs = [selected, detection, reviewer, clause_note]
        clause_feedback_outputs = [feedback_message, contract_feedback, clause_note]
        accept_clause.click(page.accept_clause, clause_feedback_inputs, clause_feedback_outputs)
        reject_clause.click(page.reject_clause, clause_feedback_inputs, clause_feedback_outputs)
        flag_missing.click(page.flag_missing, [selected, missing_category, missing_phrase, reviewer, clause_note],
                           clause_feedback_outputs)

        preview.click(page.preview_weights, weight_inputs, [weights_message, changes], api_name="preview_weights")
        save.click(page.save_weights, weight_inputs, [weights_message, version], api_name="save_weights")
        reset.click(page.editor_view, [version], editor_outputs).then(lambda: "", None, weights_message)

        review_tab.select(page.review_tab_view, queue_inputs, [contract_filter, *queue_outputs]).then(
            page.item_view, item_inputs, item_outputs)
        for control in (status_filter, contract_filter, page_number):
            control.change(page.queue_view, queue_inputs, queue_outputs).then(page.item_view, item_inputs, item_outputs)
        refresh.click(page.queue_view, queue_inputs, queue_outputs, api_name="review_queue").then(
            page.item_view, item_inputs, item_outputs)
        queue.select(page.pick_queue_row, None, item)
        item.change(page.item_view, item_inputs, item_outputs)
        search.change(page.item_view, item_inputs, item_outputs)
        for button, handler in ((accept_item, page.accept_item), (reject_item, page.reject_item),
                                (missing_item, page.missing_item)):
            button.click(handler, verdict_inputs, [verdict_message, *queue_outputs, verdict_note]).then(
                page.item_view, item_inputs, item_outputs)
    return app
