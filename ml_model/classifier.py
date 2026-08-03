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


if __name__ == "__main__":
    clf =  SmishingClassifier()
    sample = "[Web발신] CJ대한통운 고객님의 택배가 주소지 불명으로 보관중입니다. 주소 재확인 http://cj-parcel.info/kr"
    print(clf.predict(sample))