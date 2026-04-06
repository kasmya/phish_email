from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import (
    CLASS_NAMES,
    DATASET_CANDIDATES,
    LABEL_COLUMN_CANDIDATES,
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    PROCESSED_DIR,
    RANDOM_SEED,
    TARGET_DATASET_SIZE,
    TEST_RATIO,
    TEXT_COLUMN_CANDIDATES,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.preprocessing import clean_text
from src.utils import ensure_dir


def resolve_dataset_path(dataset_path: Optional[str] = None) -> Path:
    if dataset_path:
        path = Path(dataset_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Dataset not found at {path}")

    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(path) for path in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Could not locate phishing_email.csv. Place it in one of: "
        f"{checked}"
    )


def infer_text_column(columns) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        "Unable to infer the text column. Expected one of: "
        f"{', '.join(TEXT_COLUMN_CANDIDATES)}"
    )


def infer_label_column(columns) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        "Unable to infer the label column. Expected one of: "
        f"{', '.join(LABEL_COLUMN_CANDIDATES)}"
    )


def normalize_label(value) -> int:
    if pd.isna(value):
        raise ValueError("Encountered missing label value.")

    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)

    normalized = str(value).strip().lower()
    positive_values = {"1", "true", "yes", "phishing", "phish", "spam", "malicious"}
    negative_values = {"0", "false", "no", "legitimate", "ham", "safe", "benign"}

    if normalized in positive_values:
        return 1
    if normalized in negative_values:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def load_and_prepare_dataframe(
    dataset_path: Optional[str] = None,
    target_size: int = TARGET_DATASET_SIZE,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    path = resolve_dataset_path(dataset_path)
    dataframe = pd.read_csv(path)

    text_column = infer_text_column(dataframe.columns)
    label_column = infer_label_column(dataframe.columns)

    dataframe = dataframe[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe["label"] = dataframe["label"].apply(normalize_label)
    dataframe["clean_text"] = dataframe["text"].astype(str).apply(clean_text)
    dataframe = dataframe[dataframe["clean_text"].str.len() > 0]
    dataframe = dataframe.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    if len(dataframe) > target_size:
        dataframe, _ = train_test_split(
            dataframe,
            train_size=target_size,
            stratify=dataframe["label"],
            random_state=random_state,
        )
        dataframe = dataframe.reset_index(drop=True)

    return dataframe


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict:
        return {
            "train": len(self.train),
            "validation": len(self.val),
            "test": len(self.test),
        }


def split_dataframe(
    dataframe: pd.DataFrame,
    random_state: int = RANDOM_SEED,
) -> DatasetSplits:
    train_df, temp_df = train_test_split(
        dataframe,
        train_size=TRAIN_RATIO,
        stratify=dataframe["label"],
        random_state=random_state,
    )

    val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction_of_temp,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    ensure_dir(PROCESSED_DIR)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    return DatasetSplits(train=train_df, val=val_df, test=test_df)


class LSTMEmailDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


class BERTEmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def class_distribution(dataframe: pd.DataFrame) -> dict:
    counts = dataframe["label"].value_counts().to_dict()
    return {
        NEGATIVE_CLASS_NAME: int(counts.get(0, 0)),
        POSITIVE_CLASS_NAME: int(counts.get(1, 0)),
        "class_names": CLASS_NAMES,
    }
