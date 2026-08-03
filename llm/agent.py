import json
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from common.settings import (
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from ml_model.classifier import get_classifier
from vector_db.retriever import get_retriever


class SmishingAgent:
    """텍스트 -> (BERT 분류 + RAG 검색) -> LLM 최종 판단."""

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

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            max_tokens=OPENAI_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_content = response.choices[0].message.content
        llm_result = json.loads(raw_content)

        return {
            "input_text": text,
            "verdict": llm_result["verdict"],
            "scam_type": llm_result.get("scam_type", bert_result["label"]),
            "scam_type_ko": llm_result.get("scam_type_ko", bert_result["label_ko"]),
            "summary": llm_result.get("summary", ""),
            "reasons": llm_result.get("reasons", []),
            "action_guide": llm_result.get("action_guide", []),
            "used_sources": llm_result.get("used_sources", []),
            "bert_prediction": bert_result,
            "reference_chunks": reference_chunks,
        }


if __name__ == "__main__":
    agent = SmishingAgent()
    sample = (
        "[Web발신] CJ대한통운 고객님의 택배가 주소지 불명으로 보관중입니다. "
        "주소 재확인 http://cj-parcel.info/kr"
    )
    result = agent.analyze(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))