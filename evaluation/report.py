"""run_benchmark 결과를 읽어 비교 리포트(evaluation/results/report.md)를 만든다.

    python -m evaluation.report

API 호출 없이 저장된 결과만 사용한다. Fast Path 임계값 곡선은 D의 샘플별 결과에서
"BERT가 normal을 t 이상 확신하면 LLM을 생략했을 때"를 계산한 값이다.
(Fast Path는 BERT 결과만 보고 분기하므로 시뮬레이션이 실제 동작과 같다. F 실측으로 검증 가능.)
"""
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.labels import LABEL_KEYS, ko_name
from common.settings import DATA_DIR, HARD_SET
from evaluation.configs import CONFIG_TITLES
from evaluation.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    mean,
    percentile,
    wilson_interval,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
REPORT_PATH = RESULTS_DIR / "report.md"
TEST_SPLIT_CSV = DATA_DIR / "eval" / "test_split.csv"

# gpt-3.5-turbo 기준 단가(USD / 100만 토큰). 다른 모델을 쓰면 OpenAI 요금표를 확인해 수정할 것.
PRICE_PER_1M = {"input": 0.50, "output": 1.50}

SET_TITLES = {"test": "Test set (학습에 쓰이지 않은 hold-out)", "hard": "Hard set (직접 작성한 어려운 케이스)"}
SWEEP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


# ---------------------------------------------------------------- 로딩
def _read_texts(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r["text"] for r in csv.DictReader(f)]


def load_results(set_name: str, config: str, valid_texts: set) -> Dict[str, Dict]:
    path = RAW_DIR / f"{set_name}__{config}.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["text"]: r for r in records if not r.get("error") and r["text"] in valid_texts}


# ---------------------------------------------------------------- 계산
def cost_usd(metrics: Dict) -> float:
    return (
        metrics["prompt_tokens"] * PRICE_PER_1M["input"]
        + metrics["completion_tokens"] * PRICE_PER_1M["output"]
    ) / 1_000_000


def as_fast_path(record: Dict, threshold: float) -> Dict:
    """D의 샘플 결과에 Fast Path를 적용한 결과. BERT가 normal을 threshold 이상 확신하면 LLM 생략."""
    if record["bert_label"] == "normal" and record["bert_confidence"] >= threshold:
        bert_ms = record["metrics"]["bert_ms"]
        return {
            **record,
            "pred_label": "normal",
            "flagged": False,
            "fast_path": True,
            "metrics": {"bert_ms": bert_ms, "retrieval_ms": 0.0, "llm_ms": 0.0, "total_ms": bert_ms,
                        "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        }
    return record


def summarize(records: List[Dict]) -> Dict:
    gold = [r["gold"] for r in records]
    pred = [r["pred_label"] for r in records]
    n = len(records)
    correct = sum(g == p for g, p in zip(gold, pred))
    scams = [r for r in records if r["gold"] != "normal"]
    normals = [r for r in records if r["gold"] == "normal"]
    latencies = [r["metrics"]["total_ms"] for r in records]
    llm_calls = [r["metrics"]["llm_calls"] for r in records]
    costs = [cost_usd(r["metrics"]) for r in records]
    tokens = [r["metrics"]["prompt_tokens"] + r["metrics"]["completion_tokens"] for r in records]

    return {
        "n": n,
        "accuracy": accuracy(gold, pred),
        "acc_ci": wilson_interval(correct, n),
        "macro_f1": macro_f1(gold, pred, LABEL_KEYS),
        "missed_scams": sum(not r["flagged"] for r in scams),
        "n_scams": len(scams),
        "false_alarms": sum(r["flagged"] for r in normals),
        "n_normals": len(normals),
        "llm_call_rate": sum(c > 0 for c in llm_calls) / n if n else 0.0,
        "lat_mean": mean(latencies),
        "lat_p50": percentile(latencies, 50),
        "lat_p95": percentile(latencies, 95),
        "tokens_mean": mean(tokens),
        "cost_per_1k": mean(costs) * 1000,
    }


# ---------------------------------------------------------------- 출력
def md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{x:.1%}"


def main_table(by_config: Dict[str, List[Dict]]) -> str:
    rows = []
    for name, records in by_config.items():
        s = summarize(records)
        lo, hi = s["acc_ci"]
        rows.append([
            f"**{name}** {CONFIG_TITLES[name]}",
            f"{pct(s['accuracy'])} ({pct(lo)}~{pct(hi)})",
            f"{s['macro_f1']:.3f}",
            f"{s['missed_scams']}/{s['n_scams']}",
            f"{s['false_alarms']}/{s['n_normals']}",
            pct(s["llm_call_rate"]),
            f"{s['lat_p50']:.0f} / {s['lat_p95']:.0f}",
            f"{s['tokens_mean']:.0f}",
            f"${s['cost_per_1k']:.3f}",
        ])
    return md_table(
        ["구성", "정확도 (95% CI)", "macro-F1", "사기 놓침", "정상 오탐", "LLM 호출률",
         "지연 p50 / p95 (ms)", "평균 토큰", "1,000건당 비용"],
        rows,
    )


def sweep_table(d_records: List[Dict], a_records: Optional[List[Dict]]) -> str:
    a_summary = summarize(a_records) if a_records else None
    rows = []
    for threshold in [None] + SWEEP_THRESHOLDS:
        records = d_records if threshold is None else [as_fast_path(r, threshold) for r in d_records]
        s = summarize(records)
        row = [
            "없음 (= D)" if threshold is None else f"≥ {threshold}",
            pct(1 - s["llm_call_rate"]),
            pct(s["accuracy"]),
            f"{s['missed_scams']}/{s['n_scams']}",
            f"{s['false_alarms']}/{s['n_normals']}",
            f"{s['lat_mean']:.0f}",
            f"${s['cost_per_1k']:.3f}",
        ]
        if a_summary:
            row.append(f"{s['lat_mean'] / a_summary['lat_mean']:.2f}x / "
                       f"{s['cost_per_1k'] / a_summary['cost_per_1k']:.2f}x")
        rows.append(row)
    header = ["Fast Path 임계값", "LLM 호출 절감률", "정확도", "사기 놓침", "정상 오탐",
              "평균 지연 (ms)", "1,000건당 비용"]
    if a_summary:
        header.append("A 대비 지연 / 비용")
    return md_table(header, rows)


def override_table(by_config: Dict[str, List[Dict]]) -> str:
    """LLM이 BERT 1순위와 다른 최종 유형을 냈을 때, 누가 맞았는가."""
    rows = []
    for name in ["C", "D", "E"]:
        records = by_config.get(name)
        if not records:
            continue
        changed = [r for r in records if r["pred_label"] != r["bert_label"]]
        llm_right = sum(r["pred_label"] == r["gold"] for r in changed)
        bert_right = sum(r["bert_label"] == r["gold"] for r in changed)
        both_wrong = len(changed) - llm_right - bert_right
        rows.append([f"**{name}** {CONFIG_TITLES[name]}", f"{len(changed)}/{len(records)}",
                     str(llm_right), str(bert_right), str(both_wrong)])
    if not rows:
        return ""
    return md_table(["구성", "LLM이 BERT를 뒤집은 건수", "LLM이 맞음", "BERT가 맞았음 (LLM이 틀림)", "둘 다 틀림"], rows)


def category_table(by_config: Dict[str, List[Dict]]) -> str:
    categories = sorted({r["category"] for rs in by_config.values() for r in rs if r.get("category")})
    if not categories:
        return ""
    rows = []
    for cat in categories:
        row = [cat]
        for name, records in by_config.items():
            subset = [r for r in records if r.get("category") == cat]
            correct = sum(r["gold"] == r["pred_label"] for r in subset)
            row.append(f"{correct}/{len(subset)}")
        rows.append(row)
    return md_table(["카테고리"] + [f"{n}" for n in by_config], rows)


def confusion_text(records: List[Dict]) -> str:
    labels = LABEL_KEYS
    matrix = confusion_matrix([r["gold"] for r in records], [r["pred_label"] for r in records], labels)
    header = ["정답 \\ 예측"] + [ko_name(l) for l in labels] + ["기타"]
    rows = [[ko_name(l)] + [str(v) for v in matrix[i]] for i, l in enumerate(labels)]
    return md_table(header, rows)


def error_list(records: List[Dict], limit: int = 15) -> str:
    wrong = [r for r in records if r["pred_label"] != r["gold"]]
    if not wrong:
        return "오분류 없음."
    lines = []
    for r in wrong[:limit]:
        text = r["text"].replace("\n", " ")
        text = text if len(text) <= 70 else text[:70] + "…"
        lines.append(
            f"- 정답 `{r['gold']}` → 예측 `{r['pred_label']}` (BERT `{r['bert_label']}` "
            f"{r['bert_confidence']:.0%}): {text}"
        )
    if len(wrong) > limit:
        lines.append(f"- … 외 {len(wrong) - limit}건")
    return "\n".join(lines)


def build_set_section(set_name: str, texts: List[str]) -> Optional[str]:
    valid = set(texts)
    loaded = {c: load_results(set_name, c, valid) for c in "ABCDEF"}
    loaded = {c: r for c, r in loaded.items() if r}
    if not loaded:
        return None

    common = set.intersection(*(set(r) for r in loaded.values()))
    by_config = {c: [r[t] for t in texts if t in common] for c, r in loaded.items()}

    parts = [f"## {SET_TITLES[set_name]}", ""]
    note = f"비교 대상 {len(common)}건 (세트 전체 {len(texts)}건)"
    if len(common) < len(texts):
        note += " — 일부 구성에서 결과가 없거나 실패한 샘플은 공정한 비교를 위해 모든 구성에서 제외했습니다."
    parts += [note, "", "### 구성별 비교", "", main_table(by_config), ""]

    if "D" in by_config:
        parts += ["### Fast Path 임계값별 trade-off (D 결과에서 계산)", "",
                  sweep_table(by_config["D"], by_config.get("A")), ""]
    override = override_table(by_config)
    if override:
        parts += ["### LLM이 BERT 예측을 뒤집은 경우", "", override, ""]
    category = category_table(by_config)
    if category:
        parts += ["### 카테고리별 정답 수", "", category, ""]
    if "D" in by_config:
        parts += ["### D 혼동행렬", "", confusion_text(by_config["D"]), "",
                  "### D 오분류 사례", "", error_list(by_config["D"]), ""]
    return "\n".join(parts)


def main() -> None:
    sections = []
    for set_name, path in [("test", TEST_SPLIT_CSV), ("hard", HARD_SET)]:
        section = build_set_section(set_name, _read_texts(path))
        if section:
            sections.append(section)

    if not sections:
        print("결과가 없습니다. 먼저 `python -m evaluation.run_benchmark` 를 실행하세요.")
        return

    header = (
        "# 파이프라인 비교 벤치마크 리포트\n\n"
        "- 정확도·macro-F1: 7개 유형 분류 기준 / 사기 놓침: 사기 문자를 \"안전\"으로 판정한 건수 / "
        "정상 오탐: 정상 문자를 \"위험·주의\"로 판정한 건수\n"
        f"- 비용은 gpt-3.5-turbo 단가(입력 ${PRICE_PER_1M['input']}, 출력 ${PRICE_PER_1M['output']} / 100만 토큰)로 환산한 추정치입니다.\n"
        "- 표본이 수십 건이라 정확도 차이가 신뢰구간 안이면 우열을 단정할 수 없습니다.\n\n"
    )
    report = header + "\n".join(sections)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n-> {REPORT_PATH}")


if __name__ == "__main__":
    main()
