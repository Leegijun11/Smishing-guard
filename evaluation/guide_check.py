"""행동 가이드(action_guide)의 환각 검사 - LLM 없이 코드로 확인한다.

가이드에 나온 전화번호·짧은 번호(112, 1332 등)·URL이 근거 문서(data/knowledge)에도, 사용자가
입력한 문자에도 없으면 "근거 없음"으로 센다. 틀린 연락처를 안내하는 것이 이 서비스에서 가장 해로운
환각이라 이 두 종류만 본다. 기관명은 일반 명사와 구분이 어려워 코드로는 신뢰도 있게 잡을 수 없어 제외했다.

한계: 근거 문서에 "있다"는 것이 "이 상황에 맞다"를 뜻하지는 않는다 (엉뚱한 번호를 문서 안의 다른
번호로 안내해도 통과). 즉 "지어낸 번호"만 잡는 보수적인 검사다.
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.settings import KNOWLEDGE_DIR

Entity = Tuple[str, str]  # (종류, 정규화된 값)

# 02-1234-5678, 1577-1000 처럼 하이픈으로 이어진 번호
PHONE_RE = re.compile(r"(?<![\d-])(\d{2,4}-\d{3,4}(?:-\d{4})?)(?![\d-])")
# 112, 118, 1332 같은 짧은 번호. 100%, 30분, 1,200원, 1000만원 같은 숫자는 제외
SHORT_RE = re.compile(
    r"(?<![\d.,\-])(1\d{2,3})(?![\d.,\-]|\s*(?:%|퍼센트|분|시간|일|원|만|천|개|회|건|번째|배|명|차))"
)
URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?((?:[a-z0-9-]+\.)+(?:go\.kr|or\.kr|co\.kr|kr|com|net|org|info|xyz|top|cc|me|io|do))",
    re.IGNORECASE,
)


def extract_entities(text: str) -> Set[Entity]:
    entities: Set[Entity] = set()
    for m in PHONE_RE.finditer(text):
        entities.add(("phone", re.sub(r"\D", "", m.group(1))))
    for m in SHORT_RE.finditer(text):
        entities.add(("phone", m.group(1)))
    for m in URL_RE.finditer(text):
        entities.add(("url", m.group(1).lower()))
    return entities


def load_corpus_entities() -> Set[Entity]:
    """지식베이스 전체(모든 라벨 문서)에 등장하는 번호·URL."""
    entities: Set[Entity] = set()
    for md in sorted(Path(KNOWLEDGE_DIR).glob("*.md")):
        entities |= extract_entities(md.read_text(encoding="utf-8"))
    return entities


def check_guide(
    action_guide: List[str], input_text: str, corpus_entities: Set[Entity]
) -> Dict:
    """action_guide의 번호·URL 중 근거(지식베이스 또는 사용자 입력)에 없는 것을 찾는다.

    사용자 문자에 적힌 번호를 "이 번호로 연락하지 마세요"처럼 인용하는 것은 환각이 아니므로
    입력 문자에 있는 값은 근거 있음으로 본다.
    """
    found = extract_entities("\n".join(action_guide))
    allowed = corpus_entities | extract_entities(input_text)
    unsupported = sorted(e for e in found if e not in allowed)
    return {
        "entities": sorted(found),
        "unsupported": unsupported,
        "total": len(found),
        "n_unsupported": len(unsupported),
    }
