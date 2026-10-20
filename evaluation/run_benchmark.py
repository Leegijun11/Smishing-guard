"""파이프라인 구성 A~G를 평가 세트에 돌려 샘플별 결과를 저장한다.

    python -m evaluation.run_benchmark                       # test+hard+challenge+injection, A~E 전체
    python -m evaluation.run_benchmark --configs B           # BERT only (API 키 불필요)
    python -m evaluation.run_benchmark --configs D --limit 5 # 스모크 테스트
    python -m evaluation.run_benchmark --configs F --fast-path-threshold 0.8
    python -m evaluation.run_benchmark --configs B --out-dir raw_v1   # 옛 모델 결과를 따로 보관

결과는 evaluation/results/<out-dir>/<세트>__<구성>.json 에 샘플마다 저장되며(기본 raw),
다시 실행하면 이미 성공한 샘플은 건너뛰므로 API 비용이 중복으로 들지 않는다.
리포트는 `python -m evaluation.report` 로 만든다.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.labels import LABEL_KEYS
from common.settings import CHALLENGE_SET, DATA_DIR, HARD_SET, INJECTION_SET
from evaluation.configs import DEFAULT_FAST_PATH_THRESHOLD, BenchConfig, build_configs

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "raw"
TEST_SPLIT_CSV = DATA_DIR / "eval" / "test_split.csv"
DEFAULT_CONFIGS = ["A", "B", "C", "D", "E"]
EVAL_SETS = ["test", "hard", "challenge", "injection"]
CSV_SETS = {"hard": HARD_SET, "challenge": CHALLENGE_SET, "injection": INJECTION_SET}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_eval_set(name: str) -> List[Dict[str, str]]:
    if name in CSV_SETS:
        rows = _read_csv(CSV_SETS[name])
    elif name == "test":
        # 분할은 seed 고정이라 항상 같다. 어떤 문자가 test인지 확인할 수 있게 CSV로도 남긴다.
        from ml_model.dataset import load_test_dataframe

        test_df = load_test_dataframe()
        TEST_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        test_df[["text", "label"]].to_csv(TEST_SPLIT_CSV, index=False, encoding="utf-8-sig")
        rows = test_df[["text", "label"]].to_dict("records")
    else:
        raise ValueError(f"알 수 없는 평가 세트: {name}")

    unknown = {r["label"] for r in rows} - set(LABEL_KEYS)
    if unknown:
        raise ValueError(f"{name} 세트에 정의되지 않은 라벨이 있습니다: {unknown}")
    return rows


def _bert_fields(bert: Dict) -> Dict:
    top, second = bert["distribution"][0], bert["distribution"][1]
    return {
        "bert_label": top["label"],
        "bert_confidence": top["score"],
        "bert_second_label": second["label"],
        "bert_second_score": second["score"],
    }


# 기록 형식 버전. 올리면 이전 형식의 캐시는 건너뛰지 않고 다시 실행한다
# (v2: action_guide / used_sources / reference_sections 저장 추가 - 환각 검사에 필요)
RECORD_SCHEMA = 2


def run_one(cfg: BenchConfig, item: Dict[str, str], agent, classifier, top_k: int) -> Dict:
    text = item["text"]
    record: Dict = {
        "schema": RECORD_SCHEMA,
        "text": text,
        "gold": item["label"],
        "category": item.get("category"),
        "error": None,
    }

    if cfg.options is None:  # B: BERT only
        t0 = time.perf_counter()
        bert = classifier.predict(text)
        ms = (time.perf_counter() - t0) * 1000
        record.update(_bert_fields(bert))
        record.update(
            pred_label=bert["label"],
            verdict=None,
            flagged=bert["label"] != "normal",
            fast_path=False,
            summary="",
            metrics={"bert_ms": ms, "retrieval_ms": 0.0, "llm_ms": 0.0, "total_ms": ms,
                     "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        )
        return record

    result = agent.analyze(text, top_k=top_k, options=cfg.options)
    if result.get("bert_prediction"):
        record.update(_bert_fields(result["bert_prediction"]))
    record.update(
        pred_label=result["scam_type"],
        verdict=result["verdict"],
        flagged=result["verdict"] != "안전",
        fast_path=result["fast_path"],
        summary=result["summary"],
        reasons=result["reasons"],
        action_guide=result["action_guide"],
        used_sources=result["used_sources"],
        reference_sections=[f"{c['label']}:{c['section']}" for c in result["reference_chunks"]],
        metrics=result["metrics"],
    )
    return record


def _load_cache(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    return {r["text"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def _save(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=EVAL_SETS, choices=EVAL_SETS)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list("ABCDEFG"))
    parser.add_argument("--limit", type=int, default=None, help="세트당 앞에서부터 N건만 (스모크 테스트용)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--fast-path-threshold", type=float, default=DEFAULT_FAST_PATH_THRESHOLD)
    parser.add_argument("--out-dir", default=None,
                        help="결과 저장 폴더 이름 (evaluation/results/ 아래, 기본 raw). 모델을 바꿔 다시 돌릴 때 이전 결과를 보관하는 용도")
    args = parser.parse_args()

    results_dir = RESULTS_DIR.parent / args.out_dir if args.out_dir else RESULTS_DIR

    configs = build_configs(args.fast_path_threshold)
    selected = [configs[c] for c in args.configs]

    from ml_model.classifier import get_classifier

    classifier = get_classifier()
    agent = None
    if any(c.uses_llm for c in selected):
        from llm.agent import get_agent

        agent = get_agent()
        agent.retriever.search("워밍업", label="delivery", top_k=1)  # 첫 호출 지연이 측정에 섞이지 않게
    classifier.predict("워밍업")

    for set_name in args.sets:
        items = load_eval_set(set_name)
        if args.limit:
            items = items[: args.limit]

        for cfg in selected:
            path = results_dir / f"{set_name}__{cfg.name}.json"
            merged = _load_cache(path)  # text -> record. --limit 실행이 이전 결과를 지우지 않게 병합한다
            failed = 0

            print(f"\n[{set_name}] {cfg.name}. {cfg.title} - {len(items)}건")
            for i, item in enumerate(items, 1):
                cached = merged.get(item["text"])
                if cached and not cached.get("error") and cached.get("schema") == RECORD_SCHEMA:
                    continue
                try:
                    record = run_one(cfg, item, agent, classifier, args.top_k)
                except Exception as e:  # 한 건 실패로 전체 실행(과 이미 쓴 API 비용)을 잃지 않는다
                    failed += 1
                    record = {"text": item["text"], "gold": item["label"],
                              "category": item.get("category"), "error": f"{type(e).__name__}: {e}"}
                    print(f"  ! {i}/{len(items)} 실패: {record['error']}")
                merged[item["text"]] = record
                _save(path, list(merged.values()))
                if i % 10 == 0:
                    print(f"  {i}/{len(items)}")

            print(f"  완료: 성공 {len(items) - failed}, 실패 {failed} -> {path}")
            if failed:
                print("  (실패한 샘플은 다시 실행하면 그 건만 재시도합니다)")


if __name__ == "__main__":
    main()
