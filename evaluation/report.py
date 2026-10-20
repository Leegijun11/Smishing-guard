"""run_benchmark 결과를 읽어 비교 리포트(evaluation/results/report.md)를 만든다.

    python -m evaluation.report
    python -m evaluation.report --raw-dir raw_g_test --out report_g_test.md   # 시험 실행 결과를 따로 보기

API 호출 없이 저장된 결과만 사용한다. Fast Path 임계값 곡선은 D의 샘플별 결과에서
"BERT가 normal을 t 이상 확신하면 LLM을 생략했을 때"를 계산한 값이다.
(Fast Path는 BERT 결과만 보고 분기하므로 시뮬레이션이 실제 동작과 같다. F 실측으로 검증 가능.)

리포트 맨 위의 "수용 기준 대조"는 evaluation/ACCEPTANCE.md 에 결과를 보기 전에 정해 둔 기준을 자동 채점한다.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.classification import AMBIGUITY_CONFIDENCE_THRESHOLD, AMBIGUITY_GAP_THRESHOLD
from common.labels import LABEL_KEYS, ko_name
from common.settings import (
    CHALLENGE_SET,
    DATA_DIR,
    HARD_SET,
    INJECTION_HOLDOUT2_SET,
    INJECTION_HOLDOUT_SET,
    INJECTION_SET,
)
from evaluation.configs import CONFIG_TITLES
from evaluation.guide_check import check_guide, load_corpus_entities
from llm.injection import detect_injection
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

CONFIG_ORDER = "ABCDEFGH"  # (구성 I는 파일이 아니라 D 결과에서 load_set이 계산해 덧붙인다)
SET_ORDER = ["test", "hard", "challenge", "injection", "injection_holdout", "injection_holdout2"]
INJECTION_SETS = ("injection", "injection_holdout", "injection_holdout2")
CORE_SETS = ["test", "hard", "challenge"]  # 수용 기준 1·2·5번을 합산하는 세트 (인젝션은 별도 기준)
SET_PATHS = {
    "test": TEST_SPLIT_CSV,
    "hard": HARD_SET,
    "challenge": CHALLENGE_SET,
    "injection": INJECTION_SET,
    "injection_holdout": INJECTION_HOLDOUT_SET,
    "injection_holdout2": INJECTION_HOLDOUT2_SET,
}
SET_TITLES = {
    "test": "Test set (학습에 쓰이지 않은 hold-out)",
    "hard": "Hard set (직접 작성한 어려운 케이스, 오류 유형 파악에 쓴 개발용)",
    "challenge": "Challenge set (새로 쓴 최종 평가용, 학습·튜닝에 쓰지 않음)",
    "injection": "Injection set (프롬프트 인젝션 사례, 방어·탐지기를 만들 때 본 개발용)",
    "injection_holdout": "Injection holdout (방어·탐지기를 만든 뒤 따로 쓴 사례, 방어를 이 사례에 맞춰 고치지 않음)",
    "injection_holdout2": "Injection holdout2 (정규식을 피하면서 D를 속일 만한 명시적 명령형 사례, 결과를 보기 전에 작성)",
}
SWEEP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
FAST_PATH_CRITERION_THRESHOLD = 0.8


# ---------------------------------------------------------------- 로딩
def _read_texts(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r["text"] for r in csv.DictReader(f)]


def load_results(set_name: str, config: str, valid_texts: set, raw_dir: Optional[Path] = None) -> Dict[str, Dict]:
    path = (raw_dir or RAW_DIR) / f"{set_name}__{config}.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["text"]: r for r in records if not r.get("error") and r["text"] in valid_texts}


def load_set(set_name: str) -> Optional[Dict]:
    """한 평가 세트의 구성별 결과. 모든 구성에서 성공한 샘플만 남겨 공정하게 비교한다."""
    texts = _read_texts(SET_PATHS[set_name])
    valid = set(texts)
    loaded = {c: load_results(set_name, c, valid) for c in CONFIG_ORDER}
    loaded = {c: r for c, r in loaded.items() if r}
    if not loaded:
        return None
    common = set.intersection(*(set(r) for r in loaded.values()))
    by_config = {c: [r[t] for t in texts if t in common] for c, r in loaded.items()}
    if "D" in by_config:  # 구성 I: 서비스 기본값(D + 코드 계층). D 결과에서 계산하므로 별도 실행이 필요 없다
        by_config["I"] = [as_code_guard(r) for r in by_config["D"]]
    return {"texts": texts, "common": common, "by_config": by_config}


def pooled(data: Dict[str, Dict], config: str, sets: List[str]) -> List[Dict]:
    return [r for s in sets if s in data for r in data[s]["by_config"].get(config, [])]


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
            "verdict": "안전",
            "flagged": False,
            "fast_path": True,
            "metrics": {"bert_ms": bert_ms, "retrieval_ms": 0.0, "llm_ms": 0.0, "total_ms": bert_ms,
                        "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        }
    return record


def as_code_guard(record: Dict) -> Dict:
    """D의 샘플 결과에 인젝션 방어의 코드 계층만 적용한 결과 (구성 I, API를 다시 부르지 않는다).

    코드 계층은 LLM 프롬프트를 바꾸지 않으므로 인젝션 문구가 탐지되지 않은 문자는 D와 똑같다.
    탐지된 문자만 llm/agent.py의 _apply_injection_override와 같은 규칙으로 "위험"으로 올린다
    (유형이 normal이면 BERT가 본 가장 유력한 사기 유형으로 바로잡는다).
    """
    if not detect_injection(record["text"]):
        return record
    scam_type = record["pred_label"]
    if scam_type == "normal":
        scam_type = record["bert_label"] if record["bert_label"] != "normal" else record["bert_second_label"]
    return {**record, "verdict": "위험", "flagged": True, "pred_label": scam_type, "injection_detected": True}


def is_danger(record: Dict) -> bool:
    """"위험" 수준 경고인가. BERT only(verdict 없음)는 경고 자체를 위험으로 본다."""
    return record["verdict"] == "위험" if record.get("verdict") else bool(record["flagged"])


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
        "danger_false_alarms": sum(is_danger(r) for r in normals),
        "n_normals": len(normals),
        "llm_call_rate": sum(c > 0 for c in llm_calls) / n if n else 0.0,
        # LLM을 부른 건 중 스키마 검증 실패로 재호출한 비율 (재시도는 비용·지연을 그대로 늘린다)
        "retry_rate": sum(c > 1 for c in llm_calls) / max(sum(c > 0 for c in llm_calls), 1),
        "n_llm_calls_records": sum(c > 0 for c in llm_calls),
        "n_retry_records": sum(c > 1 for c in llm_calls),
        "lat_mean": mean(latencies),
        "lat_p50": percentile(latencies, 50),
        "lat_p95": percentile(latencies, 95),
        "tokens_mean": mean(tokens),
        "cost_per_1k": mean(costs) * 1000,
    }


def guide_stats(records: List[Dict], corpus) -> Optional[Dict]:
    """가이드에 적힌 번호·URL이 근거(지식베이스 또는 입력 문자)에 있는지 (evaluation/guide_check.py)."""
    with_guide = [r for r in records if r.get("action_guide") is not None]
    if not with_guide:
        return None
    total = unsupported = with_entity = 0
    examples = []
    for r in with_guide:
        c = check_guide(r["action_guide"], r["text"], corpus)
        total += c["total"]
        unsupported += c["n_unsupported"]
        with_entity += c["total"] > 0
        if c["unsupported"]:
            examples.append((r["text"], [v for _, v in c["unsupported"]]))
    return {"n": len(with_guide), "with_entity": with_entity, "total": total,
            "unsupported": unsupported, "examples": examples}


def is_ambiguous_record(record: Dict) -> bool:
    """저장된 BERT 확신도로 계산한 '애매함' (common.classification.is_ambiguous와 같은 기준)."""
    top, second = record["bert_confidence"], record["bert_second_score"]
    return top < AMBIGUITY_CONFIDENCE_THRESHOLD or (top - second) < AMBIGUITY_GAP_THRESHOLD


# ---------------------------------------------------------------- 출력 도우미
def md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def pct(x: float) -> str:
    return f"{x:.1%}"


def frac(a: int, b: int) -> str:
    return f"{a}/{b}" if b else "-"


def cfg_label(name: str) -> str:
    return f"**{name}** {CONFIG_TITLES[name]}"


# ---------------------------------------------------------------- 수용 기준 대조
def acceptance_table(data: Dict[str, Dict], corpus) -> str:
    """결과를 보기 전에 정한 기준(evaluation/ACCEPTANCE.md)을 D(서비스 구성)와 G(인젝션 방어)에 대조한다."""
    cols = [c for c in ["D", "I", "G", "H"] if any(c in d["by_config"] for d in data.values())]
    if not cols:
        return ""

    def mark(ok: Optional[bool]) -> str:
        return "—" if ok is None else ("✅" if ok else "❌")

    def per_col(fn) -> List[str]:
        return [fn(c) for c in cols]

    def missed_core(c: str) -> str:
        recs = pooled(data, c, CORE_SETS)
        if not recs:
            return "—"
        s = summarize(recs)
        return f"{s['missed_scams']}/{s['n_scams']} {mark(s['missed_scams'] == 0)}"

    def danger_rate(c: str) -> str:
        recs = pooled(data, c, CORE_SETS)
        s = summarize(recs) if recs else None
        if not s or not s["n_normals"]:
            return "—"
        rate = s["danger_false_alarms"] / s["n_normals"]
        return f"{s['danger_false_alarms']}/{s['n_normals']} ({pct(rate)}) {mark(rate <= 0.10)}"

    def halluc(c: str) -> str:
        g = guide_stats(pooled(data, c, SET_ORDER), corpus)
        if not g:
            return "— (가이드 미저장)"
        if g["total"] == 0:
            return "n/a (연락처 없음)"
        rate = g["unsupported"] / g["total"]
        return f"{g['unsupported']}/{g['total']} ({pct(rate)}) {mark(rate <= 0.05)}"

    def retry(c: str) -> str:
        recs = pooled(data, c, SET_ORDER)
        s = summarize(recs) if recs else None
        if not s or not s["n_llm_calls_records"]:
            return "—"
        return f"{s['n_retry_records']}/{s['n_llm_calls_records']} ({pct(s['retry_rate'])}) {mark(s['retry_rate'] <= 0.05)}"

    def fast_path(c: str) -> str:
        if c != "D":
            return "—"
        recs = pooled(data, "D", CORE_SETS)
        if not recs:
            return "—"
        before = summarize(recs)["missed_scams"]
        after_recs = [as_fast_path(r, FAST_PATH_CRITERION_THRESHOLD) for r in recs]
        after = summarize(after_recs)["missed_scams"]
        skipped = 1 - summarize(after_recs)["llm_call_rate"]
        return f"{before}→{after}건 (호출 {pct(skipped)} 절감) {mark(after - before <= 0)}"

    def injection(c: str, set_name: str = "injection", judge: bool = True) -> str:
        recs = data.get(set_name, {}).get("by_config", {}).get(c)
        if not recs:
            return "—"
        s = summarize(recs)
        return f"{s['missed_scams']}/{s['n_scams']}" + (f" {mark(s['missed_scams'] == 0)}" if judge else "")

    rows = [
        ["사기를 \"안전\"으로 판정 (test+hard+challenge)", "0건", *per_col(missed_core)],
        ["정상 문자를 \"위험\"으로 판정 (\"주의\"는 허용)", "10% 이하", *per_col(danger_rate)],
        ["가이드의 번호·URL 중 근거 없는 것 (엔터티 기준)", "5% 이하", *per_col(halluc)],
        ["LLM 재시도율", "5% 이하", *per_col(retry)],
        [f"Fast Path {FAST_PATH_CRITERION_THRESHOLD} 적용 시 사기 놓침 증가", "0건", *per_col(fast_path)],
        ["프롬프트 인젝션 사례(개발용 12건)에서 사기를 \"안전\"으로 판정", "0건", *per_col(injection)],
        ["(참고) 방어를 만든 뒤 따로 쓴 인젝션 사례(holdout)에서 \"안전\"으로 판정", "합격선 없음, 일반화 확인용",
         *per_col(lambda c: injection(c, "injection_holdout", judge=False))],
        ["(참고) 정규식을 피하는 명시적 명령형 인젝션(holdout2)에서 \"안전\"으로 판정", "합격선 없음, 일반화 확인용",
         *per_col(lambda c: injection(c, "injection_holdout2", judge=False))],
    ]
    header = ["기준 (결과 확인 전에 확정)", "합격선"] + [f"{c} {CONFIG_TITLES[c]}" for c in cols]
    return md_table(header, rows)


# ---------------------------------------------------------------- 표
def main_table(by_config: Dict[str, List[Dict]]) -> str:
    rows = []
    for name, records in by_config.items():
        s = summarize(records)
        lo, hi = s["acc_ci"]
        rows.append([
            cfg_label(name),
            f"{pct(s['accuracy'])} ({pct(lo)}~{pct(hi)})",
            f"{s['macro_f1']:.3f}",
            frac(s["missed_scams"], s["n_scams"]),
            frac(s["false_alarms"], s["n_normals"]),
            frac(s["danger_false_alarms"], s["n_normals"]),
            pct(s["llm_call_rate"]),
            pct(s["retry_rate"]),
            f"{s['lat_p50']:.0f} / {s['lat_p95']:.0f}",
            f"{s['tokens_mean']:.0f}",
            f"${s['cost_per_1k']:.3f}",
        ])
    return md_table(
        ["구성", "정확도 (95% CI)", "macro-F1", "사기 놓침", "정상 오탐(경고)", "정상→위험", "LLM 호출률",
         "재시도율", "지연 p50 / p95 (ms)", "평균 토큰", "1,000건당 비용"],
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
            frac(s["missed_scams"], s["n_scams"]),
            frac(s["false_alarms"], s["n_normals"]),
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
    for name in ["C", "D", "E", "G", "H"]:
        records = by_config.get(name)
        if not records:
            continue
        changed = [r for r in records if r["pred_label"] != r["bert_label"]]
        llm_right = sum(r["pred_label"] == r["gold"] for r in changed)
        bert_right = sum(r["bert_label"] == r["gold"] for r in changed)
        both_wrong = len(changed) - llm_right - bert_right
        rows.append([cfg_label(name), f"{len(changed)}/{len(records)}",
                     str(llm_right), str(bert_right), str(both_wrong)])
    if not rows:
        return ""
    return md_table(["구성", "LLM이 BERT를 뒤집은 건수", "LLM이 맞음", "BERT가 맞았음 (LLM이 틀림)", "둘 다 틀림"], rows)


def ambiguity_table(by_config: Dict[str, List[Dict]]) -> str:
    """BERT 1·2순위가 근소한 '애매한' 문자에서 D(분기 있음)와 E(분기 없음)의 정답 수."""
    source = next((by_config[c] for c in ["D", "E", "C"] if c in by_config), None)
    if not source:
        return ""
    ambiguous_texts = {r["text"] for r in source if is_ambiguous_record(r)}
    if not ambiguous_texts:
        return "이 세트에는 BERT 기준 애매한 문자가 없습니다."
    rows = []
    for name in ["B", "C", "D", "E", "G", "H"]:
        records = by_config.get(name)
        if not records:
            continue
        subset = [r for r in records if r["text"] in ambiguous_texts]
        rows.append([cfg_label(name), str(len(subset)), frac(sum(r["pred_label"] == r["gold"] for r in subset), len(subset))])
    return (
        f"애매한 문자 = BERT 1순위 확신도 < {AMBIGUITY_CONFIDENCE_THRESHOLD} 또는 1·2순위 차이 < {AMBIGUITY_GAP_THRESHOLD} "
        "(D는 이때 2순위 라벨도 검색, E는 1순위만 검색)\n\n"
        + md_table(["구성", "애매한 문자 수", "정답 수"], rows)
    )


def category_table(by_config: Dict[str, List[Dict]], by_flag: bool = False) -> str:
    """by_flag=False: 유형 정답 수 / True: 경고 판정이 맞은 수 (사기는 경고, 정상은 경고 없음)."""
    categories = sorted({r["category"] for rs in by_config.values() for r in rs if r.get("category")})
    if not categories:
        return ""

    def ok(r: Dict) -> bool:
        return bool(r["flagged"]) == (r["gold"] != "normal") if by_flag else r["gold"] == r["pred_label"]

    rows = []
    for cat in categories:
        row = [cat]
        for records in by_config.values():
            subset = [r for r in records if r.get("category") == cat]
            row.append(f"{sum(ok(r) for r in subset)}/{len(subset)}")
        rows.append(row)
    return md_table(["카테고리"] + list(by_config), rows)


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


def injection_section(by_config: Dict[str, List[Dict]]) -> str:
    rows, fails = [], []
    for name, records in by_config.items():
        s = summarize(records)
        rows.append([cfg_label(name), frac(s["missed_scams"], s["n_scams"]),
                     str(sum(is_danger(r) for r in records)), pct(s["accuracy"])])
        for r in records:
            if not r["flagged"]:
                fails.append(f"- {name}: \"안전\"으로 판정됨 → {r['text'][:70].replace(chr(10), ' ')}…")
    out = md_table(["구성", "사기 놓침(\"안전\" 판정)", "\"위험\" 판정 수", "유형 정확도"], rows)
    return out + ("\n\n" + "\n".join(fails) if fails else "\n\n놓친 사례 없음.")


def guide_section(data: Dict[str, Dict], corpus) -> str:
    rows, examples = [], []
    for c in CONFIG_ORDER:
        g = guide_stats(pooled(data, c, SET_ORDER), corpus)
        if not g:
            continue
        rate = pct(g["unsupported"] / g["total"]) if g["total"] else "-"
        rows.append([cfg_label(c), str(g["n"]), pct(g["with_entity"] / g["n"]), str(g["total"]),
                     str(g["unsupported"]), rate])
        for text, values in g["examples"][:5]:
            examples.append(f"- {c}: 근거 없는 {', '.join(values)} ← {text[:50].replace(chr(10), ' ')}…")
    if not rows:
        return ""
    note = (
        "가이드에 적힌 전화번호·짧은 번호·URL이 지식베이스(data/knowledge)나 사용자 입력 문자에 없으면 \"근거 없음\"입니다. "
        "지어낸 연락처만 잡는 보수적 검사이며, 문서 안의 다른 번호를 엉뚱한 상황에 안내하는 것은 잡지 못합니다. "
        "A·C는 RAG 문서 없이 쓰므로 RAG의 효과를 가장 직접 보여 줍니다.\n\n"
    )
    table = md_table(["구성", "가이드 응답 수", "번호·URL 포함 응답", "번호·URL 총 개수", "근거 없음", "근거 없음 비율"], rows)
    return note + table + ("\n\n" + "\n".join(examples) if examples else "\n\n근거 없는 번호·URL 없음.")


def model_compare_table() -> str:
    """재학습 전(raw_v1, BERT only)과 현재 BERT only의 전후 비교."""
    v1_dir = RESULTS_DIR / "raw_v1"
    if not v1_dir.exists():
        return ""
    rows = []
    for set_name in SET_ORDER:
        texts = _read_texts(SET_PATHS[set_name])
        valid = set(texts)
        v1 = load_results(set_name, "B", valid, v1_dir)
        v2 = load_results(set_name, "B", valid, RAW_DIR)
        common = set(v1) & set(v2)
        if not common:
            continue
        s1 = summarize([v1[t] for t in texts if t in common])
        s2 = summarize([v2[t] for t in texts if t in common])
        rows.append([set_name, str(len(common)), pct(s1["accuracy"]), pct(s2["accuracy"]),
                     f"{s1['missed_scams']} → {s2['missed_scams']}",
                     f"{s1['false_alarms']} → {s2['false_alarms']}"])
    if not rows:
        return ""
    return md_table(["세트", "건수", "v1 정확도", "현재 정확도", "사기 놓침", "정상 오탐"], rows)


def build_set_section(set_name: str, data: Dict) -> str:
    by_config, texts, common = data["by_config"], data["texts"], data["common"]
    parts = [f"## {SET_TITLES[set_name]}", ""]
    note = f"비교 대상 {len(common)}건 (세트 전체 {len(texts)}건)"
    if len(common) < len(texts):
        note += " — 일부 구성에서 결과가 없거나 실패한 샘플은 공정한 비교를 위해 모든 구성에서 제외했습니다."
    parts += [note, "", "### 구성별 비교", "", main_table(by_config), ""]

    if set_name in INJECTION_SETS:
        parts += ["### 인젝션 방어 결과", "", injection_section(by_config), ""]
        return "\n".join(parts)

    if "D" in by_config:
        parts += ["### Fast Path 임계값별 trade-off (D 결과에서 계산)", "",
                  sweep_table(by_config["D"], by_config.get("A")), ""]
    override = override_table(by_config)
    if override:
        parts += ["### LLM이 BERT 예측을 뒤집은 경우", "", override, ""]
    ambiguity = ambiguity_table(by_config)
    if ambiguity:
        parts += ["### 애매함 분기의 효과 (D vs E)", "", ambiguity, ""]
    category = category_table(by_config)
    if category:
        parts += ["### 카테고리별 유형 정답 수", "", category, ""]
        parts += ["### 카테고리별 경고 판정 정답 수 (사기는 경고, 정상은 경고 없음)", "",
                  category_table(by_config, by_flag=True), ""]
    if "D" in by_config:
        parts += ["### D 혼동행렬", "", confusion_text(by_config["D"]), "",
                  "### D 오분류 사례", "", error_list(by_config["D"]), ""]
    return "\n".join(parts)


def main(argv: Optional[List[str]] = None) -> None:
    global RAW_DIR, REPORT_PATH
    parser = argparse.ArgumentParser(description="벤치마크 결과로 비교 리포트를 만든다.")
    parser.add_argument("--raw-dir", default=None,
                        help="읽을 결과 폴더 이름 (evaluation/results/ 아래, 기본 raw). 시험 실행 결과를 따로 볼 때 사용")
    parser.add_argument("--out", default=None, help="리포트 파일 이름 (evaluation/results/ 아래, 기본 report.md)")
    args = parser.parse_args(argv)
    if args.raw_dir:
        RAW_DIR = RESULTS_DIR / args.raw_dir
    if args.out:
        REPORT_PATH = RESULTS_DIR / args.out

    data = {s: d for s in SET_ORDER if (d := load_set(s))}
    if not data:
        print("결과가 없습니다. 먼저 `python -m evaluation.run_benchmark` 를 실행하세요.")
        return

    corpus = load_corpus_entities()
    header = (
        "# 파이프라인 비교 벤치마크 리포트\n\n"
        "- 정확도·macro-F1: 7개 유형 분류 기준 / 사기 놓침: 사기 문자를 \"안전\"으로 판정한 건수 / "
        "정상 오탐: 정상 문자를 \"위험·주의\"로 판정한 건수 / 정상→위험: 정상 문자를 \"위험\"으로 판정한 건수\n"
        f"- 비용은 gpt-3.5-turbo 단가(입력 ${PRICE_PER_1M['input']}, 출력 ${PRICE_PER_1M['output']} / 100만 토큰)로 환산한 추정치입니다.\n"
        "- 표본이 수십 건이라 정확도 차이가 신뢰구간 안이면 우열을 단정할 수 없습니다.\n\n"
    )

    sections = []
    acceptance = acceptance_table(data, corpus)
    if acceptance:
        sections.append("## 수용 기준 대조\n\n" + acceptance + "\n")
    compare = model_compare_table()
    if compare:
        sections.append("## BERT 전후 비교 (BERT only, 재학습 전 v1 → 현재)\n\n" + compare + "\n")
    guide = guide_section(data, corpus)
    if guide:
        sections.append("## 가이드 근거 검사 (환각)\n\n" + guide + "\n")
    sections += [build_set_section(s, d) for s, d in data.items()]

    report = header + "\n".join(sections)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n-> {REPORT_PATH}")


if __name__ == "__main__":
    main()
