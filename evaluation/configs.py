"""벤치마크에서 비교할 파이프라인 구성.

운영 코드(SmishingAgent.analyze)를 그대로 쓰되 PipelineOptions 스위치만 바꾼다.
Fast Path 임계값을 바꿔 가며 보는 곡선은 D의 결과에서 계산하므로(report.py) 별도 실행이 필요 없다.
F는 선택한 임계값 하나를 실제로 돌려 그 계산이 맞는지 검증하는 용도다.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

# 재현성을 위해 벤치마크는 temperature=0으로 고정한다.
_T = 0.0

# F를 실제로 돌릴 때 쓰는 기본 임계값 (--fast-path-threshold 로 변경)
DEFAULT_FAST_PATH_THRESHOLD = 0.9

# report.py가 torch/openai 없이도 돌도록 가벼운 이름표는 따로 둔다.
CONFIG_TITLES: Dict[str, str] = {
    "A": "LLM only",
    "B": "BERT only",
    "C": "BERT+LLM",
    "D": "BERT+RAG+LLM",
    "E": "D − 애매함 분기",
    "F": "Fast Path(실측)",
}


@dataclass(frozen=True)
class BenchConfig:
    name: str
    title: str
    description: str
    uses_llm: bool
    options: Optional["PipelineOptions"]  # BERT only(B)는 에이전트를 거치지 않으므로 None


def build_configs(fast_path_threshold: float = DEFAULT_FAST_PATH_THRESHOLD) -> Dict[str, BenchConfig]:
    from llm.agent import PipelineOptions  # 무거운 의존성(openai, torch)은 실제로 실행할 때만 로드

    return {
        "A": BenchConfig(
            "A", "LLM only", "BERT·RAG 없이 LLM만 호출 (비용·속도 baseline)", True,
            PipelineOptions(use_bert=False, use_rag=False, use_ambiguity_branch=False,
                            fast_path_threshold=None, temperature=_T),
        ),
        "B": BenchConfig(
            "B", "BERT only", "LLM·RAG 없이 BERT 1순위 라벨만 사용 (판별 baseline, 서비스 불가)", False,
            None,
        ),
        "C": BenchConfig(
            "C", "BERT+LLM", "RAG 없이 BERT 예측을 LLM에 전달", True,
            PipelineOptions(use_bert=True, use_rag=False, use_ambiguity_branch=False,
                            fast_path_threshold=None, temperature=_T),
        ),
        "D": BenchConfig(
            "D", "BERT+RAG+LLM", "현재 서비스 구성 (Fast Path 없음)", True,
            PipelineOptions(use_bert=True, use_rag=True, use_ambiguity_branch=True,
                            fast_path_threshold=None, temperature=_T),
        ),
        "E": BenchConfig(
            "E", "D − 애매함 분기", "D에서 1·2순위 근소 시 2순위 검색 분기를 제거", True,
            PipelineOptions(use_bert=True, use_rag=True, use_ambiguity_branch=False,
                            fast_path_threshold=None, temperature=_T),
        ),
        "F": BenchConfig(
            "F", f"Fast Path(≥{fast_path_threshold})",
            "D + BERT가 정상을 임계값 이상 확신하면 LLM 생략 (실측 검증용)", True,
            PipelineOptions(use_bert=True, use_rag=True, use_ambiguity_branch=True,
                            fast_path_threshold=fast_path_threshold, temperature=_T),
        ),
    }
