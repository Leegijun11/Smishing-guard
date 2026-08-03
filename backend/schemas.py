from typing import List

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="분석할 문자 메시지 원문")


class BertDistributionItem(BaseModel):
    label: str
    label_ko: str
    score: float


class BertPrediction(BaseModel):
    label: str
    label_ko: str
    confidence: float
    distribution: List[BertDistributionItem]


class ReferenceChunk(BaseModel):
    text: str
    section: str
    label: str
    label_ko: str
    distance: float


class AnalyzeResponse(BaseModel):
    input_text: str
    verdict: str
    scam_type: str
    scam_type_ko: str
    summary: str
    reasons: List[str]
    action_guide: List[str]
    used_sources: List[str]
    bert_prediction: BertPrediction
    reference_chunks: List[ReferenceChunk]


class LabelInfo(BaseModel):
    key: str
    ko: str
    description: str


class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
