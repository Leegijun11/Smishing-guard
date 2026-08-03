import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# ---- 데이터 ----
DATA_DIR = ROOT_DIR / "data"
RAW_DATASET = DATA_DIR / "raw" / "smishing_dataset.csv"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"

# ---- 산출물 ----
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else (ROOT_DIR / p)


# ---- ML model ----
BASE_BERT_MODEL = os.getenv("BASE_BERT_MODEL", "klue/bert-base")
BERT_MODEL_DIR = _path("BERT_MODEL_DIR", "artifacts/bert-smishing")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "128"))

# ---- Vector DB ----
CHROMA_DIR = _path("CHROMA_DIR", "artifacts/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "smishing_playbook")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# ---- LLM (OpenAI) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "900"))

# ---- Backend ----
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))