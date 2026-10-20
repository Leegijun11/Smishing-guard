"""벤치마크 지표 계산 (표준 라이브러리만 사용)."""
import math
from typing import Dict, List, Sequence, Tuple


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def per_class_f1(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> float:
    per_class = per_class_f1(gold, pred, labels)
    return sum(v["f1"] for v in per_class.values()) / len(labels)


def confusion_matrix(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> List[List[int]]:
    """행=정답, 열=예측. labels 밖의 예측(LLM이 엉뚱한 키를 낸 경우)은 마지막 열 'other'로 모은다."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for g, p in zip(gold, pred):
        matrix[index[g]][index.get(p, len(labels))] += 1
    return matrix


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """이항 비율의 95% Wilson 신뢰구간. 표본이 작을 때(수십 건) 정규근사보다 안전하다."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile(values: Sequence[float], q: float) -> float:
    """선형 보간 백분위수 (q: 0~100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
