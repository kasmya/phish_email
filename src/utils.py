import json
import random
from pathlib import Path

import numpy as np
import torch

from src.config import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def phishing_probability_to_label(probability: float) -> str:
    return "Phishing" if probability >= 0.5 else "Legitimate"


def confidence_from_probability(probability: float) -> float:
    return probability if probability >= 0.5 else 1.0 - probability


def risk_from_probability(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= LOW_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def percent(value: float) -> float:
    return round(value * 100.0, 2)
