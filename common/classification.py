from typing import Dict, List

AMBIGUITY_CONFIDENCE_THRESHOLD = 0.45
AMBIGUITY_GAP_THRESHOLD = 0.15


def is_ambiguous(
    distribution: List[Dict],
    confidence_threshold: float = AMBIGUITY_CONFIDENCE_THRESHOLD,
    gap_threshold: float = AMBIGUITY_GAP_THRESHOLD,
) -> bool:
    """BERT 1순위 예측이 애매한지 판단.

    1순위 confidence가 threshold보다 낮거나, 1순위-2순위 score 차이가
    gap_threshold보다 작으면 애매한 것으로 본다. distribution은 score 내림차순으로
    정렬되어 있다고 가정한다 (SmishingClassifier.predict 참고).
    """
    if len(distribution) < 2:
        return False

    top_score = distribution[0]["score"]
    second_score = distribution[1]["score"]

    return top_score < confidence_threshold or (top_score - second_score) < gap_threshold
