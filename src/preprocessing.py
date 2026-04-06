import re
from collections import Counter
from typing import Iterable, List

import torch

from src.config import MAX_SEQUENCE_LENGTH

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    return clean_text(text).split()


def build_vocabulary(
    texts: Iterable[str],
    max_vocab_size: int,
    min_frequency: int = 1,
) -> dict:
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    # Reserve stable ids for padding and unknown words so saved models remain reusable.
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_frequency:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text_for_lstm(text: str, vocab: dict, max_length: int = MAX_SEQUENCE_LENGTH) -> List[int]:
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize_text(text)]
    if len(token_ids) >= max_length:
        return token_ids[:max_length]
    return token_ids + [vocab[PAD_TOKEN]] * (max_length - len(token_ids))


def batch_encode_for_lstm(texts: Iterable[str], vocab: dict, max_length: int = MAX_SEQUENCE_LENGTH) -> torch.Tensor:
    encoded = [encode_text_for_lstm(text, vocab, max_length=max_length) for text in texts]
    return torch.tensor(encoded, dtype=torch.long)
