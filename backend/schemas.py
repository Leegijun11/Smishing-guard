from typing import List

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="분석할 문자 메시지 원문")
# 클리이언트가 /analyze 에 보내는 요청 검증용

class BertDistributionItem(BaseModel):
    label: str
    label_ko: str
    score: float
# 라벨 하나의 예측 확률 정보

class BertPrediction(BaseModel):
    label: str
    label_ko: str
    confidence: float
    distribution: List[BertDistributionItem]
# 라벨에 순위별 예측정보를 가진 사용자 텍스트의 입력에 대한 bert 분류

class ReferenceChunk(BaseModel):
    text: str
    section: str
    label: str
    label_ko: str
    distance: float
# retriever.search() 결과의 타입 정의

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
# /analyze 엔드포인트가 클라이언트에 돌려줄 최종 응답 구조

class LabelInfo(BaseModel):
    key: str
    ko: str
    description: str
# ㅣlabels 엔드포인트용

class HealthResponse(BaseModel):
    status: str
    agent_ready: bool
