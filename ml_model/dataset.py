import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from common.labels import LABEL2ID
from common.settings import RAW_DATASET


def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATASET)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    unknown = set(df["label"]) - set(LABEL2ID)
    if unknown:
        raise ValueError(
            f"CSV에 common/labels.py 에 정의되지 않은 라벨이 있습니다: {unknown}"
        )

    df["label_id"] = df["label"].map(LABEL2ID)
    # ml 모델 학습용 숫자
    return df


def load_dataset_dict(test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    df = load_dataframe()
    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_id"],
    )
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(valid_df.reset_index(drop=True)),
        }
    )


if __name__ == "__main__":
    dd = load_dataset_dict()
    print(dd)
    print(dd["train"][0])