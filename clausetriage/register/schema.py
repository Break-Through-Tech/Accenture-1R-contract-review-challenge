"""
The clause register is a named deliverable: for each contract, every clause the pipeline detected,
with its risk rating, the signals behind it, where it sits in the contract and the quote that
supports it. These Pydantic models are the register's single definition, shared by the batch
script, the demo page and the future API, so all of them export the same columns.

How everything comes together:
RegisterRow: one scored clause (one ClauseScore plus its evidence, page and provenance)
RegisterHeader: the contract's triage summary (score, rank, rating counts, model, weight version)
ContractRegister: a header plus its rows; JSON written from it validates back into an equal model

Rows without a grounded quote (the Unknown queue) are kept, flagged needs_manual_review, and carry
no quote, offsets or page: the register never drops a detected clause and never invents a quote.

Counting rules (the same as risk.contract.ContractScore):
- high_count, medium_count, low_count and unknown_count count distinct categories, using each
  category's worst clause, because that is what the triage score counts
- clause_count and manual_review_count count rows, i.e. every detected (chunk, category) clause
"""
from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from clausetriage.evidence.schema import Evidence
from clausetriage.labels.categories import CATEGORIES
from clausetriage.risk.schema import SIGNAL_NAMES, Rating, SignalName, UnitFloat

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
# Why a clause needs manual review: the two evidence-extractor outcomes, or no evidence record at all
ReviewReason = Literal["no_cue_match", "span_mismatch", "no_evidence"]


class RegisterRow(BaseModel):
    """
    One scored clause of one contract.
    With evidence, quote/start/end are set (contract text[start:end] == quote) and page is set when
    the document has pages. Without evidence, all four are None, needs_manual_review is True and
    review_reason says why.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = Field(min_length=1)
    category: str
    rating: Rating
    composite_score: UnitFloat
    signals: dict[SignalName, UnitFloat]
    probability: UnitFloat
    chunk_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = None
    quote: str | None = None
    has_evidence: bool
    needs_manual_review: bool
    review_reason: ReviewReason | None = None
    model_name: str = Field(min_length=1)
    weight_version_id: str = Field(min_length=1)
    weight_set_sha256: Sha256

    @field_validator("category")
    @classmethod
    def check_category(cls, name: str) -> str:
        """Only the 41 canonical CUAD category names are allowed."""
        if name not in CATEGORIES:
            raise ValueError(f"Unknown CUAD category: {name!r}")
        return name

    @field_validator("signals")
    @classmethod
    def check_signals(cls, signals: dict[str, float]) -> dict[str, float]:
        """Every risk signal must be present, so each export has the same columns."""
        if set(signals) != set(SIGNAL_NAMES):
            raise ValueError(f"signals must have exactly {list(SIGNAL_NAMES)}, got {sorted(signals)}")
        return {name: signals[name] for name in SIGNAL_NAMES}

    # A grounded row has a consistent span; an unquoted row has no span and is flagged for review
    @model_validator(mode="after")
    def check_evidence_fields(self) -> Self:
        """Evidence fields are all set (and consistent) with evidence, and all None without it."""
        if self.has_evidence:
            if self.quote is None or self.start is None or self.end is None:
                raise ValueError("a row with evidence needs quote, start and end")
            Evidence(quote=self.quote, start=self.start, end=self.end)  # raises on a bad span
            if self.needs_manual_review or self.review_reason is not None:
                raise ValueError("a row with evidence is not flagged for manual review")
        else:
            if any(value is not None for value in (self.quote, self.start, self.end, self.page)):
                raise ValueError("a row without evidence must not carry a quote, offsets or page")
            if not self.needs_manual_review or self.review_reason is None:
                raise ValueError("a row without evidence must be flagged for manual review with a reason")
        return self


class RegisterHeader(BaseModel):
    """
    One contract's triage summary under one model and one weight version.
    rank 1 means open first; ranked_out_of is how many contracts were ranked together (None if unknown).
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str = Field(min_length=1)
    triage_score: UnitFloat
    rank: int = Field(ge=1)
    ranked_out_of: int | None = Field(default=None, ge=1)
    high_count: int = Field(ge=0)
    medium_count: int = Field(ge=0)
    low_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    clause_count: int = Field(ge=0)
    manual_review_count: int = Field(ge=0)
    model_name: str = Field(min_length=1)
    model_seed: int | None = None
    weight_version_id: str = Field(min_length=1)
    weight_set_sha256: Sha256

    @model_validator(mode="after")
    def check_rank(self) -> Self:
        """A rank can't exceed the number of contracts ranked."""
        if self.ranked_out_of is not None and self.rank > self.ranked_out_of:
            raise ValueError(f"rank {self.rank} is beyond ranked_out_of {self.ranked_out_of}")
        return self


class ContractRegister(BaseModel):
    """
    A contract's register: its header and every scored clause, riskiest first.
    Every row must belong to the header's contract, model and weight version, and the header's counts
    must agree with the rows, so a register stitched together from mismatched runs fails validation.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    header: RegisterHeader
    rows: list[RegisterRow]

    # Rows share the header's identity and provenance
    @model_validator(mode="after")
    def check_rows_match_header(self) -> Self:
        """Rows carry the header's contract, model and weight version; counts match the rows."""
        header = self.header
        for row in self.rows:
            if (row.contract_id, row.model_name, row.weight_version_id, row.weight_set_sha256) != (
                    header.contract_id, header.model_name, header.weight_version_id, header.weight_set_sha256):
                raise ValueError(f"row {row.category!r} (chunk {row.chunk_index}) doesn't match the header's "
                                 "contract, model or weight version")
        if header.clause_count != len(self.rows):
            raise ValueError(f"clause_count is {header.clause_count} but there are {len(self.rows)} rows")
        flagged = sum(row.needs_manual_review for row in self.rows)
        if header.manual_review_count != flagged:
            raise ValueError(f"manual_review_count is {header.manual_review_count} but {flagged} rows are flagged")

        # Rating counts are over each category's worst clause (tied clauses share a rating)
        worst: dict[str, RegisterRow] = {}
        for row in self.rows:
            if row.category not in worst or row.composite_score > worst[row.category].composite_score:
                worst[row.category] = row
        ratings = Counter(row.rating for row in worst.values())
        counts = {"High": header.high_count, "Medium": header.medium_count, "Low": header.low_count}
        if counts != {rating: ratings[rating] for rating in counts}:
            raise ValueError(f"header rating counts {counts} don't match the rows' worst clauses {dict(ratings)}")
        return self
