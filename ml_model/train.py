import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common.labels import ID2LABEL, LABEL2ID, LABEL_KEYS
from common.settings import BASE_BERT_MODEL, BERT_MODEL_DIR, MAX_SEQ_LEN
from ml_model.dataset import load_dataset_dict


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_BERT_MODEL)
    dataset = load_dataset_dict()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label_id", "labels")
    keep_cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized["train"].column_names:
        keep_cols.append("token_type_ids")
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep_cols]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_BERT_MODEL,
        num_labels=len(LABEL_KEYS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = BERT_MODEL_DIR.parent / "bert-checkpoints"

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("최종 검증 결과:", metrics)

    trainer.save_model(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))
    print(f"모델 저장 완료 -> {BERT_MODEL_DIR}")


if __name__ == "__main__":
    main()