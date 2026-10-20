from pydantic import BaseModel, field_validator

"""
Every piece of evidence shown to a reviewer or used for risk scoring needs to be 
verifiable. It must sit at [start:end] in the original contract. This model acts 
as the single shared definition of "grounded evidence" so that every piece 
(extractor, risk scorere, clause-register export, reviewer UI, etc) can all agree 
on what an evidence object looks like.

'start' and 'end' are character offsets in the full contract text

There are two layers of checking
1. 'end' must be greater than 'start'. Pydantic can only see the object itself here,
not the contract, so this is the only check at contruction-time.
2. validate_against verifies that the quote exists as [start:end] in the full contract
text. This is a seperate method, because contract text isn't stored on the model.

Extractor will call validate_against on every result, and anything that fils gets routed
to an unknown queue
"""

class Evidence(BaseModel):
    quote: str # exact text of supporting sentence
    start: int # character offset of where quote begins (inclusive)
    end: int # character offset of where the quote ends (exclusive)

    @field_validator("end")
    @classmethod
    def end_after_start(cls, end, info):
        # info.data holds fields already validated, so 'start' is available
        start = info.data.get("start")
        if start is not None and end <= start:
            raise ValueError("End must be after start.")
        return end

    # Returns true only if evidence's span exactly reproduces quote in the source text
    def validate_against(self, source_text: str) -> bool:
        return 0 <= self.start < self.end <= len(source_text) and source_text[self.start:self.end] == self.quote