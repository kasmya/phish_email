import time

import torch
from transformers import AutoTokenizer

from src.config import (
    BERT_ARTIFACT_DIR,
    CLASS_NAMES,
    LSTM_ARTIFACT_DIR,
    LSTM_DROPOUT,
    LSTM_EMBEDDING_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    MAX_SEQUENCE_LENGTH,
)
from src.models.bert_model import create_bert_classifier
from src.models.lstm_model import BiLSTMClassifier
from src.preprocessing import encode_text_for_lstm
from src.utils import (
    confidence_from_probability,
    get_device,
    load_json,
    percent,
    phishing_probability_to_label,
    risk_from_probability,
)


class PhishingInferenceService:
    def __init__(self):
        self.device = get_device()
        self.lstm_model = None
        self.bert_model = None
        self.vocab = None
        self.bert_tokenizer = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        vocab_path = LSTM_ARTIFACT_DIR / "vocab.json"
        lstm_weights_path = LSTM_ARTIFACT_DIR / "model.pt"
        bert_config_path = BERT_ARTIFACT_DIR / "config.json"

        if not vocab_path.exists() or not lstm_weights_path.exists():
            raise FileNotFoundError(
                "LSTM artifacts were not found. Train the project first with `python train.py`."
            )
        if not bert_config_path.exists():
            raise FileNotFoundError(
                "BERT artifacts were not found. Train the project first with `python train.py`."
            )

        self.vocab = load_json(vocab_path)
        self.lstm_model = BiLSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=LSTM_EMBEDDING_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            num_classes=len(CLASS_NAMES),
        )
        state_dict = torch.load(lstm_weights_path, map_location=self.device)
        self.lstm_model.load_state_dict(state_dict)
        self.lstm_model.to(self.device)
        self.lstm_model.eval()

        # The saved BERT directory contains both the fine-tuned weights and tokenizer files.
        self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_ARTIFACT_DIR)
        self.bert_model = create_bert_classifier(str(BERT_ARTIFACT_DIR), num_labels=len(CLASS_NAMES))
        self.bert_model.to(self.device)
        self.bert_model.eval()

    def _predict_lstm(self, text: str) -> dict:
        encoded = encode_text_for_lstm(text, self.vocab, max_length=MAX_SEQUENCE_LENGTH)
        tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.lstm_model(tensor)
            phishing_probability = torch.softmax(logits, dim=1)[:, 1].item()
        return self._format_prediction("LSTM", phishing_probability)

    def _predict_bert(self, text: str) -> dict:
        encoded = self.bert_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.bert_model(**encoded).logits
            phishing_probability = torch.softmax(logits, dim=1)[:, 1].item()
        return self._format_prediction("BERT", phishing_probability)

    def _format_prediction(self, model_name: str, phishing_probability: float) -> dict:
        label = phishing_probability_to_label(phishing_probability)
        confidence = confidence_from_probability(phishing_probability)
        return {
            "model_name": model_name,
            "prediction": label,
            "phishing_probability": percent(phishing_probability),
            "confidence": percent(confidence),
            "risk_level": risk_from_probability(phishing_probability),
            "verdict_class": label.lower(),
        }

    def predict(self, text: str) -> dict:
        started_at = time.perf_counter()
        bert_result = self._predict_bert(text)
        lstm_result = self._predict_lstm(text)

        average_probability = (
            (bert_result["phishing_probability"] + lstm_result["phishing_probability"]) / 2.0
        ) / 100.0
        average_confidence = (
            bert_result["confidence"] + lstm_result["confidence"]
        ) / 2.0
        final_verdict = phishing_probability_to_label(average_probability)

        return {
            "bert": bert_result,
            "lstm": lstm_result,
            "summary": {
                "models_agree": bert_result["prediction"] == lstm_result["prediction"],
                "average_confidence": round(average_confidence, 2),
                "final_verdict": final_verdict,
                "analysis_time_seconds": round(time.perf_counter() - started_at, 3),
            },
        }
