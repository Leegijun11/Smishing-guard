from typing import List, Literal

from pydantic import BaseModel


class SmishingVerdict(BaseModel):
    verdict: Literal["위험", "주의", "안전"]
    scam_type: str
    scam_type_ko: str
    summary: str
    reasons: List[str]
    action_guide: List[str]
    used_sources: List[str]
