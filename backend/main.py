import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    LabelInfo,
)
from common.labels import LABELS
from common.settings import BACKEND_HOST, BACKEND_PORT
from llm.agent import get_agent, is_ready

app = FastAPI(
    title="Smishing Guard API",
    description="문자 메시지의 스미싱 여부를 BERT 분류 + RAG + LLM으로 판단하는 API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_up():
    """서버 기동 시 BERT/ChromaDB/OpenAI 클라이언트를 미리 로드해 첫 요청 지연을 없앤다."""
    try:
        get_agent()
    except Exception as e:
        print(f"[startup] 에이전트 초기화 실패 (요청 시 재시도됩니다): {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if is_ready() else "warming_up", agent_ready=is_ready())


@app.get("/labels", response_model=List[LabelInfo])
def list_labels():
    return [LabelInfo(key=l.key, ko=l.ko, description=l.description) for l in LABELS]


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    try:
        agent = get_agent()
        result = agent.analyze(payload.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)