import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from pydantic import ValidationError

from common.classification import is_ambiguous
from common.labels import ko_name
from common.settings import (
    FAST_PATH_NORMAL_THRESHOLD,
    INJECTION_GUARD,
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from llm.injection import detect_injection
from llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_GUARDED, build_user_prompt
from llm.schemas import SmishingVerdict
from ml_model.classifier import get_classifier
from vector_db.retriever import get_retriever

MAX_RETRIES = 3


@dataclass(frozen=True)
class PipelineOptions:
    """파이프라인 구성 스위치. 기본값은 운영(서비스) 구성이며,
    evaluation/ 벤치마크가 같은 코드 경로에서 부품을 하나씩 끄며 비교할 때 쓴다."""

    use_bert: bool = True
    use_rag: bool = True
    # 1·2순위가 근소할 때 2순위 라벨도 검색하는 분기 (README 트러블슈팅 1, 2번)
    use_ambiguity_branch: bool = True
    # BERT가 normal을 이 확신도 이상으로 판단하면 LLM 호출을 생략. None이면 비활성화.
    fast_path_threshold: Optional[float] = FAST_PATH_NORMAL_THRESHOLD
    # None이면 settings.OPENAI_TEMPERATURE 사용
    temperature: Optional[float] = None
    # 프롬프트 인젝션 방어 (문자 격리 프롬프트, 인젝션 문구 탐지 시 "위험" 강제, 탐지되면 Fast Path 차단)
    injection_guard: bool = INJECTION_GUARD


class SmishingAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요."
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.classifier = get_classifier()
        self.retriever = get_retriever()

    def analyze(
        self,
        text: str,
        top_k: int = 4,
        options: Optional[PipelineOptions] = None,
    ) -> Dict:
        options = options or PipelineOptions()

        text = text.strip()
        if not text:
            raise ValueError("분석할 텍스트가 비어 있습니다.")
        if options.use_rag and not options.use_bert:
            raise ValueError("RAG 검색은 BERT 분류 라벨이 필요합니다 (use_bert=True).")

        started = time.perf_counter()
        metrics: Dict = {
            "bert_ms": 0.0,
            "retrieval_ms": 0.0,
            "llm_ms": 0.0,
            "total_ms": 0.0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        # 인젝션 방어: AI에게 하는 말이 문자에 들어 있는지 (코드로 탐지, LLM을 거치지 않는다)
        injection_hits: List[str] = detect_injection(text) if options.injection_guard else []

        bert_result: Optional[Dict] = None
        if options.use_bert:
            t0 = time.perf_counter()
            bert_result = self.classifier.predict(text)
            metrics["bert_ms"] = (time.perf_counter() - t0) * 1000

            # 인젝션 문구가 있으면 BERT가 정상을 확신해도 LLM 없이 "안전"으로 통과시키지 않는다
            if not injection_hits and self._is_fast_path(bert_result, options):
                metrics["total_ms"] = (time.perf_counter() - started) * 1000
                return self._fast_path_response(text, bert_result, metrics)

        reference_chunks: List[Dict] = []
        if options.use_rag:
            t0 = time.perf_counter()
            reference_chunks = self._retrieve(
                text, bert_result, top_k, options.use_ambiguity_branch
            )
            metrics["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        user_prompt = build_user_prompt(
            text, bert_result, reference_chunks, rag_enabled=options.use_rag,
            guard=options.injection_guard,
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT_GUARDED if options.injection_guard else SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        t0 = time.perf_counter()
        llm_result = self._call_llm_with_retry(messages, metrics, options.temperature)
        metrics["llm_ms"] = (time.perf_counter() - t0) * 1000
        metrics["total_ms"] = (time.perf_counter() - started) * 1000

        result = {
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
            "fast_path": False,
            "injection_detected": bool(injection_hits),
            "metrics": metrics,
        }
        if injection_hits:
            metrics["injection_hits"] = injection_hits
            self._apply_injection_override(result, bert_result)
        return result

    @staticmethod
    def _apply_injection_override(result: Dict, bert_result: Optional[Dict]) -> None:
        """인젝션 문구가 탐지된 문자는 LLM이 속아서 "안전"이라고 해도 "위험"으로 올린다.

        정상 문자는 AI에게 판정 방법을 지시하지 않으므로 그 자체가 사기 신호다. 유형이 normal로 나왔으면
        BERT가 본 가장 유력한 사기 유형으로 바로잡아 verdict와 scam_type이 모순되지 않게 한다.
        """
        result["verdict"] = "위험"
        result["reasons"] = [
            "문자 안에 AI의 판정을 조작하려는 지시문이 포함되어 있습니다 (프롬프트 인젝션 시도). "
            "정상적인 문자는 이런 문구를 쓰지 않습니다."
        ] + list(result["reasons"])
        if result["scam_type"] == "normal" and bert_result:
            top_scam = next((d for d in bert_result["distribution"] if d["label"] != "normal"), None)
            if top_scam:
                result["scam_type"] = top_scam["label"]
                result["scam_type_ko"] = top_scam["label_ko"]

    @staticmethod
    def _is_fast_path(bert_result: Dict, options: PipelineOptions) -> bool:
        # 사기 라벨은 확신도가 아무리 높아도 LLM을 거친다 (사기를 안전으로 놓치는 오류만 막으면 된다).
        return (
            options.fast_path_threshold is not None
            and bert_result["label"] == "normal"
            and bert_result["confidence"] >= options.fast_path_threshold
        )

    @staticmethod
    def _fast_path_response(text: str, bert_result: Dict, metrics: Dict) -> Dict:
        return {
            "input_text": text,
            "verdict": "안전",
            "scam_type": "normal",
            "scam_type_ko": ko_name("normal"),
            "summary": (
                f"분석 모델이 정상 문자로 {bert_result['confidence']:.1%} 확신도로 판단했습니다. "
                "스미싱 의심 패턴이 발견되지 않았습니다."
            ),
            "reasons": ["BERT 분류 모델이 정상 문자로 매우 높은 확신도로 판단했습니다."],
            "action_guide": [
                "링크 클릭이나 개인정보 입력 전에는 발신처가 공식 번호·도메인인지 한 번 더 확인하세요."
            ],
            "used_sources": [],
            "bert_prediction": bert_result,
            "reference_chunks": [],
            "fast_path": True,
            "injection_detected": False,
            "metrics": metrics,
        }

    def _retrieve(
        self,
        text: str,
        bert_result: Dict,
        top_k: int,
        use_ambiguity_branch: bool,
    ) -> List[Dict]:
        distribution = bert_result["distribution"]
        ambiguous = use_ambiguity_branch and is_ambiguous(distribution)
        secondary_label = distribution[1]["label"] if len(distribution) > 1 else None

        if bert_result["label"] != "normal":
            if ambiguous and secondary_label is not None:
                primary_k = max(top_k - top_k // 2, 1)
                secondary_k = max(top_k // 2, 1)
                return self.retriever.search(
                    query_text=text, label=bert_result["label"], top_k=primary_k
                ) + self.retriever.search(
                    query_text=text, label=secondary_label, top_k=secondary_k
                )
            return self.retriever.search(
                query_text=text, label=bert_result["label"], top_k=top_k
            )

        if ambiguous and secondary_label is not None and secondary_label != "normal":
            # 1순위는 normal이지만 2순위 사기 유형과 근소한 차이인 경우.
            # normal 라벨은 knowledge base에 참고자료가 없으므로 2순위 라벨만 검색한다.
            return self.retriever.search(
                query_text=text, label=secondary_label, top_k=top_k
            )
        return []

    def _call_llm_with_retry(
        self,
        messages: List[Dict[str, str]],
        metrics: Dict,
        temperature: Optional[float] = None,
    ) -> SmishingVerdict:
        last_error: Optional[str] = None
        temperature = OPENAI_TEMPERATURE if temperature is None else temperature

        for _ in range(MAX_RETRIES):
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=temperature,
                max_tokens=OPENAI_MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=messages,
            )
            metrics["llm_calls"] += 1
            if response.usage is not None:
                metrics["prompt_tokens"] += response.usage.prompt_tokens
                metrics["completion_tokens"] += response.usage.completion_tokens
            raw_content = response.choices[0].message.content

            try:
                return self._parse_and_validate(raw_content)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                # 재시도 원인을 벤치마크 결과에서 볼 수 있게 남긴다 (재시도는 비용·지연을 그대로 늘린다)
                metrics.setdefault("retry_errors", []).append(
                    {"error": last_error[:300], "raw": (raw_content or "")[:300]}
                )
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



