import sys
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from common.labels import LABEL2ID
from common.settings import AUGMENT_DATASET, CHALLENGE_SET, HARD_SET, INJECTION_SET, RAW_DATASET

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


def _protected_texts(valid_df: pd.DataFrame, test_df: pd.DataFrame) -> set:
    """학습 보강 데이터에 들어오면 안 되는 문자: validation/test와 모든 평가셋(hard, challenge, injection)."""
    texts = set(valid_df["text"]) | set(test_df["text"])
    for path in (HARD_SET, CHALLENGE_SET, INJECTION_SET):
        if Path(path).exists():
            texts |= set(pd.read_csv(path)["text"].dropna())
    return texts


def load_augmentation(protected: set) -> pd.DataFrame:
    """data/raw/augment.csv(text,label)가 있으면 읽는다. 평가용 문자와 겹치면 오류로 막는다.

    원본 CSV와 분리해 두는 이유: 원본에 행을 추가하면 train/validation/test 분할 자체가 바뀌어,
    기존 test 문자가 train으로 들어가거나 새 문자가 test로 들어가 평가가 오염된다.
    """
    columns = ["text", "label", "label_id"]
    if not Path(AUGMENT_DATASET).exists():
        return pd.DataFrame(columns=columns)

    aug = pd.read_csv(AUGMENT_DATASET).dropna(subset=["text", "label"]).drop_duplicates("text")
    unknown = set(aug["label"]) - set(LABEL2ID)
    if unknown:
        raise ValueError(f"augment.csv에 common/labels.py 에 정의되지 않은 라벨이 있습니다: {unknown}")
    overlap = protected & set(aug["text"])
    if overlap:
        raise ValueError(
            f"augment.csv에 평가·검증용 문자와 똑같은 문자가 {len(overlap)}건 있습니다 "
            f"(학습에 쓰면 평가가 오염됩니다): {sorted(overlap)[:3]}"
        )
    aug["label_id"] = aug["label"].map(LABEL2ID)
    return aug[columns].reset_index(drop=True)


def load_dataset_dict(seed: int = SPLIT_SEED) -> DatasetDict:
    train_df, valid_df, test_df = split_dataframes(seed=seed)
    aug_df = load_augmentation(_protected_texts(valid_df, test_df))
    if len(aug_df):
        # 분할 '이후' train에만 합친다. validation/test는 보강 여부와 무관하게 항상 같다.
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        print(f"학습 보강 데이터 {len(aug_df)}건을 train에 추가 (train {len(train_df)}건)")
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
