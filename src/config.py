from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"
METRICS_DIR = REPORTS_DIR / "metrics"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
LSTM_ARTIFACT_DIR = ARTIFACTS_DIR / "lstm"
BERT_ARTIFACT_DIR = ARTIFACTS_DIR / "bert"

DATASET_CANDIDATES = (
    BASE_DIR / "phishing_email.csv",
    DATA_DIR / "phishing_email.csv",
)

RANDOM_SEED = 42
TARGET_DATASET_SIZE = 15_000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

MAX_SEQUENCE_LENGTH = 256
MAX_VOCAB_SIZE = 20_000
MIN_TOKEN_FREQUENCY = 1

LSTM_EMBEDDING_DIM = 64
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.60
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 15
LSTM_LEARNING_RATE = 1e-3
LSTM_WEIGHT_DECAY = 1e-5

BERT_MODEL_NAME = "bert-base-uncased"
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 4
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.10

EARLY_STOPPING_PATIENCE = 2
GRADIENT_CLIP_NORM = 1.0

POSITIVE_CLASS_NAME = "Phishing"
NEGATIVE_CLASS_NAME = "Legitimate"
CLASS_NAMES = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]

TEXT_COLUMN_CANDIDATES = (
    "text",
    "email_text",
    "email",
    "body",
    "message",
    "content",
    "mail_text",
    "mail_body",
)

LABEL_COLUMN_CANDIDATES = (
    "label",
    "target",
    "class",
    "is_phishing",
    "phishing",
    "spam",
    "category",
)

LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.70
