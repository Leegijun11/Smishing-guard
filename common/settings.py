import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# ---- 데이터 ----
DATA_DIR = ROOT_DIR / "data"
RAW_DATASET = DATA_DIR / "raw" / "Smishing_dataset.csv"
# 학습 보강 데이터(선택). 원본 CSV에 직접 추가하면 train/validation/test 분할이 바뀌어 test가 오염되므로
# 별도 파일에 두고, 분할 이후 train에만 합친다 (ml_model/dataset.py).
AUGMENT_DATASET = DATA_DIR / "raw" / "augment.csv"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
HARD_SET = DATA_DIR / "eval" / "hard_set.csv"
# 학습 보강 때 절대 보지 않는 최종 평가용 (hard set은 오류 유형 파악에 쓴 개발용)
CHALLENGE_SET = DATA_DIR / "eval" / "challenge_set.csv"
INJECTION_SET = DATA_DIR / "eval" / "injection_set.csv"
# 인젝션 탐지기·방어를 만든 뒤에 따로 쓴 사례. 탐지기를 이 파일에 맞춰 고치지 않는다 (일반화 확인용).
INJECTION_HOLDOUT_SET = DATA_DIR / "eval" / "injection_holdout.csv"
# holdout은 방어 없이도 안 뚫려서 변별력이 없었다. 정규식을 피하면서 방어 없는 구성(D)을 실제로 속일 만한
# 명시적 명령 형태로 새로 쓴 사례. 결과를 보기 전에 작성했고, 이 파일에 맞춰 방어를 고치지 않는다.
INJECTION_HOLDOUT2_SET = DATA_DIR / "eval" / "injection_holdout2.csv"

# ---- 산출물 ----
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def _path(default: str) -> Path:
    p = Path(default)
    return p if p.is_absolute() else (ROOT_DIR / p)


def _optional_float(name: str):
    value = os.getenv(name, "").strip()
    return float(value) if value else None


# ---- ML model ----
BASE_BERT_MODEL = os.getenv("BASE_BERT_MODEL", "klue/bert-base")
# 재학습 시 이전 모델을 덮어쓰지 않도록 BERT_MODEL_DIR로 저장·로드 위치를 바꿀 수 있다.
BERT_MODEL_DIR = _path(os.getenv("BERT_MODEL_DIR", "artifacts/bert-smishing"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "128"))

# ---- Vector DB ----
CHROMA_DIR = _path("artifacts/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "smishing_playbook")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# ---- LLM (OpenAI) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "900"))

# ---- Fast Path ----
# BERT가 "정상 문자"를 이 확신도 이상으로 판단하면 LLM 호출 없이 바로 "안전"을 반환한다.
# 비어 있으면(기본값) 비활성화. 벤치마크(evaluation/)로 임계값을 정한 뒤 .env에 설정한다.
FAST_PATH_NORMAL_THRESHOLD = _optional_float("FAST_PATH_NORMAL_THRESHOLD")

# ---- 프롬프트 인젝션 방어 ----
# 켜면 문자 격리 프롬프트 + 인젝션 문구 탐지 시 "위험" 강제 + Fast Path 차단 (llm/injection.py).
# 기본은 꺼져 있고, 벤치마크(구성 G)로 효과와 부작용을 확인한 뒤 .env에 INJECTION_GUARD=true 로 켠다.
INJECTION_GUARD = os.getenv("INJECTION_GUARD", "").strip().lower() in ("1", "true", "yes", "on")

# ---- Backend ----
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))