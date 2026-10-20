"""문자 사이의 비슷함(near-duplicate) 검사.

정확히 같은 문자만 막으면 한두 글자만 바꾼 변형이 학습 보강 데이터로 들어가 평가셋을 사실상 "미리 보고 공부"하게 된다.
띄어쓰기·기호를 제거하고 비교해서 글자 변형(예: "택 배" / "택.배")으로 가려진 중복도 잡는다.
"""
import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

_KEEP = re.compile(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]")

# 이 이상이면 사실상 같은 문자로 보고 학습을 막는다 (ml_model/dataset.py, evaluation/check_leakage.py 공통)
NEAR_DUPLICATE_BLOCK = 0.85


def normalize(text: str) -> str:
    """공백·기호를 지우고 소문자로 바꾼다 (글자 변형에 가려진 중복을 찾기 위함)."""
    return _KEEP.sub("", text).lower()


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def near_duplicates(
    candidates: Iterable[str], pool: Dict[str, str], threshold: float
) -> List[Tuple[float, str, str, str]]:
    """candidates 각각과 가장 비슷한 pool 문자를 찾아 유사도가 threshold 이상인 것만 돌려준다.

    pool: {문자: 출처 이름}. 반환: (유사도, 후보 문자, 가장 비슷한 문자, 출처) 목록 (유사도 내림차순).
    """
    found = []
    for cand in candidates:
        best = (0.0, "", "")
        for text, source in pool.items():
            score = similarity(cand, text)
            if score > best[0]:
                best = (score, text, source)
        if best[0] >= threshold:
            found.append((best[0], cand, best[1], best[2]))
    return sorted(found, reverse=True)
