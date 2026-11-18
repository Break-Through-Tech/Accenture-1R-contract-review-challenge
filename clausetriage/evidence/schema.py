"""
Every piece of evidence shown to a reviewer or used for risk scoring needs to be
verifiable. It must sit at [start:end] in the original contract. This model acts
as the single shared definition of "grounded evidence" so that every piece
(extractor, risk scorer, clause-register export, reviewer UI, etc) can all agree
on what an evidence object looks like.

'start' and 'end' are absolute, end-exclusive character offsets into the full contract text.

There are two layers of checking
1. At construction time the span must be non-empty, start must be >= 0, and the quote
length must equal end - start. Pydantic can only see the object itself here, not the contract.
2. validate_against verifies that the quote exists as [start:end] in the full contract
text. This is a separate method, because contract text isn't stored on the model.

The extractor calls validate_against on every result, and anything that fails gets routed
to the Unknown queue.
"""
from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator


class Evidence(BaseModel):
    """
    A quote from a contract plus its absolute, end-exclusive offsets.
    A valid Evidence satisfies contract_text[start:end] == quote (see validate_against).
    """
    quote: str  # exact text of supporting sentence
    start: int = Field(ge=0)  # character offset of where quote begins (inclusive)
    end: int  # character offset of where the quote ends (exclusive)

    # Model Validator; ensures the span is non-empty & that the quote length matches it
    @model_validator(mode="after")
    def check_span_consistency(self) -> Self:
        if self.end <= self.start:
            raise ValueError("End must be after start.")
        if len(self.quote) != self.end - self.start:
            raise ValueError(
                f"Quote length ({len(self.quote)}) must equal end - start ({self.end - self.start})"
            )
        return self

    def validate_against(self, source_text: str) -> bool:
        """Return True only if source_text[start:end] reproduces the quote exactly."""
        return self.end <= len(source_text) and source_text[self.start:self.end] == self.quote
