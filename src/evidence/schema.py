from pydantic import BaseModel, field_validator

class Evidence(BaseModel):
    """
    A grounded quote from a chunk, with the character span it came from.
    Guarantees that quote == chunk_text[start:end]
    """
    quote: str
    start: int
    end: int

    @field_validator("end")
    @classmethod
    def end_after_start(cls, end, info):
        start = info.data.get("start")
        if start is not None and end <= start:
            raise ValueError("End must be after start.")
        return end

    # Returns true only if evidence's span exactly reproduces quote in the chunk text
    def validate_against(self, chunk_text: str) -> bool:
        return 0 <= self.start < self.end <= len(chunk_text) and chunk_text[self.start:self.end] == self.quote