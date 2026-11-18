"""
ClauseTriage demo page: upload one or more contracts (PDF, Word or text, or paste text), rank them
for review, and drill into any contract's risky clauses with page numbers, highlighted quotes and
the signals behind each score.

How everything comes together:
score_documents(): runs every Document through the ContractInferencePipeline
build_app(): the Gradio page wired to one pipeline (views come from src/ui/report.py)

Run from the repo root (the model downloads from the GitHub Release on first run):
    python -m src.ui.app
    python -m src.ui.app --registry-version latest       # use the local MLflow registry instead
    python -m src.ui.app --bundle path/to/bundle.joblib   # use a specific bundle file (trusted files only)

PROTOTYPE: scores use the team's v2 weights, which no legal expert has validated; they are not a
real risk assessment.
"""
import argparse
import tempfile
from collections import Counter
from pathlib import Path

import gradio as gr
import pandas as pd
from transformers import AutoTokenizer

from src.chunking.token_chunker import TOKENIZER_NAME, TOKENIZER_REVISION
from src.evidence.cues import load_category_cues
from src.inference import ContractInferencePipeline
from src.ingest.documents import SUPPORTED_SUFFIXES, Document, pasted_document, read_document
from src.registry.demo_model import TOKENIZER_CACHE, load_demo_bundle
from src.ui.report import (CLAUSE_COLUMNS, CONTRACT_COLUMNS, ScoredContract, clause_register, clause_table,
                           contract_table, highlighted_text, rank_contracts, verdict)

DEMO_DIR = Path(__file__).resolve().parents[2] / "data" / "demo_contracts"
MAX_FILE_SIZE = "25mb"
RATING_COLORS = {"High": "#e5484d", "Medium": "#f5a524", "Low": "#46a758"}
DISCLAIMER = "> Research prototype with team-set weights that no legal expert has validated: not legal advice or a real risk assessment."


def score_documents(pipeline: ContractInferencePipeline, documents: list[Document],
                    progress=lambda fraction, desc: None) -> list[ScoredContract]:
    """Score each document; repeated names get a suffix so contract ids stay unique."""
    seen, scored = Counter(), []
    for index, document in enumerate(documents):
        progress(index / len(documents), f"Scoring {document.name}")
        seen[document.name] += 1
        contract_id = document.name if seen[document.name] == 1 else f"{document.name} ({seen[document.name]})"
        scored.append(ScoredContract(document, pipeline.predict_contract(contract_id, document.text)))
    return scored


def build_app(pipeline: ContractInferencePipeline) -> gr.Blocks:
    """The demo page: inputs, the ranked contract list, one contract's details, and the CSV download."""
    def analyze(paths, pasted, show_harmless, progress=gr.Progress()):
        """Read and score every input, rank them together, and show the top contract."""
        documents, skipped = [], []
        for path in paths or []:
            try:
                documents.append(read_document(path))
            except Exception as error:  # one unreadable file shouldn't stop the others
                skipped.append(f"- **{Path(path).name}**: {error}")
        if pasted and pasted.strip():
            try:
                documents.append(pasted_document(pasted))
            except ValueError as error:  # e.g. over the length limit
                skipped.append(f"- **Pasted text**: {error}")
        if not documents:
            note = "\n".join(["**Nothing to score.** Upload a contract or paste its text.", *skipped])
            return None, pd_empty(CONTRACT_COLUMNS), gr.Dropdown(choices=[], value=None), note, \
                pd_empty(CLAUSE_COLUMNS), gr.HighlightedText(value=[]), None

        scored = score_documents(pipeline, documents, progress)
        ranked = rank_contracts(scored, pipeline.weight_set)
        state = {"scored": {item.result["contract_id"]: item for item in scored},
                 "ranked": {c.contract_id: c for c in ranked}}
        register = Path(tempfile.mkdtemp(prefix="clausetriage-")) / "clause_register.csv"
        clause_register(scored, ranked).to_csv(register, index=False)

        names = [c.contract_id for c in ranked]
        overview = f"Ranked **{len(ranked)}** contract(s); rank 1 is the one to open first."
        if skipped:
            overview += "\n\nSkipped:\n" + "\n".join(skipped)
        summary, clauses, text = render(state, names[0], show_harmless)
        return (state, contract_table(ranked), gr.Dropdown(choices=names, value=names[0]),
                overview + "\n\n" + summary, clauses, text, str(register))

    def render(state, name, show_harmless):
        """One contract's headline, clause table and highlighted text."""
        if not state or name not in state["scored"]:
            return "", pd_empty(CLAUSE_COLUMNS), gr.HighlightedText(value=[])
        item, score = state["scored"][name], state["ranked"][name]
        document = item.document
        pages = f"{len(document.page_starts)} pages" + (" (read with OCR)" if document.ocr else "") \
            if document.page_starts else "no page numbers"
        summary = (
            f"## {name}\n"
            f"### {verdict(score)} · triage score {score.triage_score:.2f} · rank {score.rank} of {len(state['ranked'])}\n"
            f"**{score.high_count} High · {score.medium_count} Medium · {score.low_count} Low** clause types "
            f"({score.unknown_count} without a quote) · {pages} · model `{item.result['model_name']}` · "
            f"weights `{score.weight_version_id}`\n\n{DISCLAIMER}"
        )
        segments = highlighted_text(item)
        colors = {label: RATING_COLORS[label.split(":")[0]] for _, label in segments if label}
        return summary, clause_table(item, show_harmless), gr.HighlightedText(value=segments, color_map=colors)

    with gr.Blocks(title="ClauseTriage") as app:
        gr.Markdown("# ClauseTriage\nUpload contracts to rank them for review and see each one's riskiest "
                    "clauses, where they are, and why they scored that way.\n\n" + DISCLAIMER)
        state = gr.State()
        with gr.Row():
            files = gr.File(label="Contracts (PDF, Word or text)", file_count="multiple",
                            file_types=list(SUPPORTED_SUFFIXES), type="filepath")
            pasted = gr.Textbox(label="Or paste contract text", lines=6)
        with gr.Row():
            score_button = gr.Button("Score contracts", variant="primary")
            sample_button = gr.Button("Try the sample contracts", visible=DEMO_DIR.exists())
            show_harmless = gr.Checkbox(label="Show harmless clause types (Parties, dates)", value=False)
        contracts = gr.Dataframe(headers=CONTRACT_COLUMNS, label="Contracts, riskiest first", interactive=False)
        download = gr.File(label="Clause register (CSV)")
        selected = gr.Dropdown(label="Contract details", choices=[])
        summary = gr.Markdown()
        clauses = gr.Dataframe(headers=CLAUSE_COLUMNS, label="Risky clauses (worst per type)", wrap=True,
                               interactive=False)
        text = gr.HighlightedText(label="Contract text with detected clauses highlighted", combine_adjacent=False)

        outputs = [state, contracts, selected, summary, clauses, text, download]
        score_button.click(analyze, inputs=[files, pasted, show_harmless], outputs=outputs)
        sample_button.click(lambda harmless, progress=gr.Progress(): analyze(
            sorted(str(p) for p in DEMO_DIR.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES), "", harmless,
            progress), inputs=show_harmless, outputs=outputs)
        for control in (selected, show_harmless):
            control.change(render, inputs=[state, selected, show_harmless], outputs=[summary, clauses, text])
    return app


def pd_empty(columns: list[str]) -> pd.DataFrame:
    """An empty table with the given columns."""
    return pd.DataFrame(columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ClauseTriage demo page")
    parser.add_argument("--bundle", type=Path, help="ClauseModelBundle .joblib to load (only load trusted files)")
    parser.add_argument("--registry-version", help="Load this version ('latest' or a number) from local MLflow")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    bundle = load_demo_bundle(args.bundle, args.registry_version)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=TOKENIZER_REVISION, use_fast=True,
                                              cache_dir=TOKENIZER_CACHE)
    pipeline = ContractInferencePipeline(bundle, tokenizer, load_category_cues())
    build_app(pipeline).launch(server_port=args.port, max_file_size=MAX_FILE_SIZE)  # local only: no share link


if __name__ == "__main__":
    main()
