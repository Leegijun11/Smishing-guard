import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from pydantic import ValidationError

from common.settings import (
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from llm.schemas import SmishingVerdict
from ml_model.classifier import get_classifier
from vector_db.retriever import get_retriever

MAX_RETRIES = 3

class SmishingAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요."
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.classifier = get_classifier()
        self.retriever = get_retriever()

    def analyze(self, text: str, top_k: int = 4) -> Dict:
        text = text.strip()
        if not text:
            raise ValueError("분석할 텍스트가 비어 있습니다.")

        bert_result = self.classifier.predict(text)

        reference_chunks = []
        if bert_result["label"] != "normal":
            reference_chunks = self.retriever.search(
                query_text=text, label=bert_result["label"], top_k=top_k
            )

        user_prompt = build_user_prompt(text, bert_result, reference_chunks)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        llm_result = self._call_llm_with_retry(messages)

        return {
            "input_text": text,
            "verdict": llm_result.verdict,
            "scam_type": llm_result.scam_type,
            "scam_type_ko": llm_result.scam_type_ko,
            "summary": llm_result.summary,
            "reasons": llm_result.reasons,
            "action_guide": llm_result.action_guide,
            "used_sources": llm_result.used_sources,
            "bert_prediction": bert_result,
            "reference_chunks": reference_chunks,
        }

    def _call_llm_with_retry(self, messages: List[Dict[str, str]]) -> SmishingVerdict:
        last_error: Optional[str] = None

        for _ in range(MAX_RETRIES):
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=messages,
            )
            raw_content = response.choices[0].message.content

            try:
                return self._parse_and_validate(raw_content)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                messages.append({"role": "assistant", "content": raw_content or ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"이전 응답이 요구된 JSON 스키마와 맞지 않습니다: {last_error}\n"
                            "코드블록이나 다른 텍스트 없이, 지정된 JSON 스키마 형식으로만 다시 답변하세요."
                        ),
                    }
                )

        raise RuntimeError(
            f"LLM이 {MAX_RETRIES}회 시도에도 유효한 JSON 스키마 응답을 반환하지 못했습니다. "
            f"마지막 오류: {last_error}"
        )

    @staticmethod
    def _parse_and_validate(raw_content: Optional[str]) -> SmishingVerdict:
        parsed = json.loads(raw_content or "")
        return SmishingVerdict.model_validate(parsed)


_singleton: Optional[SmishingAgent] = None


def get_agent() -> SmishingAgent:
    global _singleton
    if _singleton is None:
        _singleton = SmishingAgent()
    return _singleton


def is_ready() -> bool:
    return _singleton is not None


if __name__ == "__main__":
    agent = get_agent()
    sample = (
        "[Web발신] CJ대한통운 고객님의 택배가 주소지 불명으로 보관중입니다. "
        "주소 재확인 http://cj-parcel.info/kr"
    )
    result = agent.analyze(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))
