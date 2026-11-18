import json
from typing_extensions import Self
from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class AnswerSpan(BaseModel):
    """
    One annotated CUAD answer span for a single clause category.
    start and end are absolute, end-exclusive character offsets into the original
    contract text, so text == contract_text[start:end].
    """
    category: str
    start: int = Field(ge=0)
    end: int
    text: str

    # Model Validator; ensures the span is non-empty & that text matches it exactly
    @model_validator(mode="after")
    def check_span_consistency(self) -> Self:
        if self.end <= self.start:
            raise ValueError("End index of span must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"Text length ({len(self.text)}) must equal end - start ({self.end - self.start})"
            )
        return self


class Contract(BaseModel):
    """
    One CUAD contract with all of its annotated answer spans.
    text is the full, unaltered original contract text (the single text source the
    chunker also reads), and every span's offsets index into it.
    """
    contract_id: str
    text: str
    spans: list[AnswerSpan]

    # Model Validator; ensures every span is grounded: text[start:end] == span.text
    @model_validator(mode="after")
    def check_spans_grounded(self) -> Self:
        for span in self.spans:
            actual = self.text[span.start:span.end]
            if actual != span.text:
                raise ValueError(
                    f"Span not grounded in contract '{self.contract_id}': "
                    f"category '{span.category}' at [{span.start}:{span.end}] "
                    f"expected {span.text!r} but found {actual!r}"
                )
        return self


def load_cuad(path: str | Path) -> list[Contract]:
    """
    Load a CUAD JSON file into one Contract per document, with its answer spans.
    The category is taken from each question ID by dropping the contract title
    prefix and the trailing _N suffix. Unanswered questions yield no spans, and
    any ungrounded or empty span raises a ValueError.
    """
    # Read the whole JSON file as UTF-8
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    contracts = []
    for entry in data["data"]:
        # Each contract has a single paragraph holding its full text and all questions
        paragraph = entry["paragraphs"][0]
        text = paragraph["context"]
        spans = []
        for qa in paragraph["qas"]:
            # "<title>__<Category>_<N>" -> "<Category>"
            category = qa["id"].rsplit("__", 1)[-1].rsplit("_", 1)[0]

            # Turn each answer into an end-exclusive span (no answers -> no spans)
            for answer in qa["answers"]:
                answer_text = answer["text"]
                start = answer["answer_start"]
                end = start + len(answer_text)
                spans.append(AnswerSpan(category=category, start=start, end=end, text=answer_text))
        contracts.append(Contract(contract_id=entry["title"], text=text, spans=spans))

    return contracts
