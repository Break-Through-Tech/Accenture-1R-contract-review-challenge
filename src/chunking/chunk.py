from typing_extensions import Self
from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """
    A contiguous slice of one contract's text, with its position in that contract.
    start and end are absolute, end-exclusive character offsets into the original
    contract text, so text == original_text[start:end]. chunk_index is the chunk's
    position within its contract (0-based).
    """
    contract_id: str
    start: int = Field(ge=0)
    end: int
    text: str
    chunk_index: int = Field(ge=0)

    # Model Validator; ensures the span is non-empty & that text matches it exactly
    @model_validator(mode="after")
    def check_span_consistency(self) -> Self:
        if self.end <= self.start:
            raise ValueError("End index of chunk must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"Text length ({len(self.text)}) must equal end - start ({self.end - self.start})"
            )
        return self