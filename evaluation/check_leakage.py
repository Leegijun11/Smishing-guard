"""학습 보강 데이터(augment.csv)가 평가·검증용 문자와 너무 비슷한지 점검한다.

    python -m evaluation.check_leakage

정확히 같은 문자는 학습 시작 전에 ml_model/dataset.py가 막는다. 이 스크립트는 한두 글자만 다른 '비슷한' 문자를
찾는다 (공백·기호를 지우고 비교하므로 글자 변형으로 가려진 중복도 잡는다).

    유사도 >= 0.85  : 사실상 같은 문자 -> 학습이 막힌다 (dataset.py). 문자를 새로 써야 한다.
    유사도 >= 0.70  : 사람이 확인할 것 (같은 문장 구조를 따라 썼을 가능성)
    그 아래          : 문제 없음

같은 "유형"(예: 카드 승인 알림)은 비슷할 수밖에 없다. 문제는 같은 문장을 조금만 바꾼 것이다.
"""
import csv
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.settings import (
    AUGMENT_DATASET,
    CHALLENGE_SET,
    DATA_DIR,
    HARD_SET,
    INJECTION_HOLDOUT2_SET,
    INJECTION_HOLDOUT_SET,
    INJECTION_SET,
)
from common.similarity import NEAR_DUPLICATE_BLOCK, near_duplicates

BLOCK_THRESHOLD = NEAR_DUPLICATE_BLOCK
REVIEW_THRESHOLD = 0.70


def _read_texts(path: Path) -> list:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r["text"] for r in csv.DictReader(f) if r.get("text")]


def build_pool() -> Dict[str, str]:
    """비교 대상: 모든 평가셋 + test 분할 + (가능하면) validation 분할."""
    pool: Dict[str, str] = {}
    for name, path in [
        ("hard", HARD_SET),
        ("challenge", CHALLENGE_SET),
        ("injection", INJECTION_SET),
        ("injection_holdout", INJECTION_HOLDOUT_SET),
        ("injection_holdout2", INJECTION_HOLDOUT2_SET),
        ("test", DATA_DIR / "eval" / "test_split.csv"),
    ]:
        for text in _read_texts(path):
            pool.setdefault(text, name)
    try:  # validation은 CSV로 저장돼 있지 않아 분할을 다시 계산한다 (pandas 필요)
        from ml_model.dataset import split_dataframes

        _, valid_df, test_df = split_dataframes()
        for text in valid_df["text"]:
            pool.setdefault(text, "validation")
        for text in test_df["text"]:
            pool.setdefault(text, "test")
    except ImportError:
        print("(pandas 등이 없어 validation 비교는 건너뜁니다)")
    return pool


def main() -> int:
    candidates = _read_texts(AUGMENT_DATASET)
    if not candidates:
        print(f"{AUGMENT_DATASET} 가 없거나 비어 있습니다.")
        return 0
    pool = build_pool()
    found = near_duplicates(candidates, pool, REVIEW_THRESHOLD)

    print(f"보강 데이터 {len(candidates)}건 vs 평가·검증 문자 {len(pool)}건")
    blocked = [f for f in found if f[0] >= BLOCK_THRESHOLD]
    review = [f for f in found if f[0] < BLOCK_THRESHOLD]
    for title, rows in (("차단 대상 (>= 0.85)", blocked), ("확인 필요 (0.70 ~ 0.85)", review)):
        print(f"\n[{title}] {len(rows)}건")
        for score, cand, match, source in rows:
            print(f"  {score:.2f} [{source}]\n     보강: {cand[:60]}\n     평가: {match[:60]}")
    if not found:
        print("\n비슷한 문자 없음 (최대 유사도 0.70 미만).")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
