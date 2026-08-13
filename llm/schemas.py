from typing import List, Literal

from pydantic import BaseModel, Field


class SmishingVerdict(BaseModel):
    verdict: Literal["위험", "주의", "안전"]
    scam_type: str
    scam_type_ko: str
    summary: str
    reasons: List[str] = Field(min_length=1)
    action_guide: List[str] = Field(min_length=1)
    used_sources: List[str]
