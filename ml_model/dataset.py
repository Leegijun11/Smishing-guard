import sys
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from common.labels import LABEL2ID
from common.settings import RAW_DATASET

SPLIT_SEED = 42


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


def split_dataframes(
    valid_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train / validation / test 3분할 (라벨 비율 유지, seed 고정).

    validation은 학습 중 best model 선택(early stopping)에 쓰이므로, 성능 보고는
    한 번도 학습 과정에 쓰이지 않은 test로만 한다.
    """
    df = load_dataframe()
    rest_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_id"],
    )
    train_df, valid_df = train_test_split(
        rest_df,
        test_size=valid_size / (1 - test_size),
        random_state=seed,
        stratify=rest_df["label_id"],
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_dataset_dict(seed: int = SPLIT_SEED) -> DatasetDict:
    train_df, valid_df, test_df = split_dataframes(seed=seed)
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(valid_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def load_test_dataframe() -> pd.DataFrame:
    """벤치마크(evaluation/)가 쓰는 고정 test set."""
    return split_dataframes()[2]


if __name__ == "__main__":
    dd = load_dataset_dict()
    print(dd)
    print(dd["train"][0])
