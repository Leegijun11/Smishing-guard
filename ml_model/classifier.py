import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.labels import ID2LABEL, ko_name
from common.settings import BASE_BERT_MODEL, BERT_MODEL_DIR, MAX_SEQ_LEN


class SmishingClassifier:
    """스미싱 문자 -> 유형 라벨 + 확률 분포."""

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        model_path = model_dir or str(BERT_MODEL_DIR)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"파인튜닝된 모델을 찾을 수 없습니다: {model_path}\n"
                f"먼저 `python -m ml_model.train` 을 실행해 학습하세요. "
                f"(베이스 모델: {BASE_BERT_MODEL})"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1).cpu().tolist()

        ranked = sorted(
            (
                {"label": ID2LABEL[i], "label_ko": ko_name(ID2LABEL[i]), "score": p}
                for i, p in enumerate(probs)
            ),
            key=lambda x: x["score"],
            reverse=True,
        )
        top = ranked[0]
        return {
            "label": top["label"],
            "label_ko": top["label_ko"],
            "confidence": top["score"],
            "distribution": ranked,
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]


_singleton: Optional[SmishingClassifier] = None


def get_classifier() -> SmishingClassifier:
    """FastAPI 등에서 모델을 한 번만 로드해 재사용하기 위한 헬퍼."""
    global _singleton
    if _singleton is None:
        _singleton = SmishingClassifier()
    return _singleton


if __name__ == "__main__":
    clf = get_classifier()
    sample = "고객님의 카드론 연체건이 검찰 송치 대상으로 분류되었습니다. 즉시 상환 확인이 없을 경우 법적 조치가 진행됩니다. 상담원 연결 02-XXX-XXXX"
    print(clf.predict(sample))
