import argparse

import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import (
    BERT_ARTIFACT_DIR,
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LEARNING_RATE,
    BERT_MODEL_NAME,
    BERT_WEIGHT_DECAY,
    CLASS_NAMES,
    LSTM_ARTIFACT_DIR,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EMBEDDING_DIM,
    LSTM_EPOCHS,
    LSTM_HIDDEN_DIM,
    LSTM_LEARNING_RATE,
    LSTM_NUM_LAYERS,
    LSTM_WEIGHT_DECAY,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    METRICS_DIR,
    MIN_TOKEN_FREQUENCY,
    PLOTS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.data import (
    BERTEmailDataset,
    LSTMEmailDataset,
    class_distribution,
    load_and_prepare_dataframe,
    split_dataframe,
)
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    plot_training_history,
    save_metrics_bundle,
    write_comparison_report,
)
from src.models.bert_model import create_bert_classifier
from src.models.lstm_model import BiLSTMClassifier
from src.preprocessing import batch_encode_for_lstm, build_vocabulary
from src.trainer import (
    evaluate_bert_model,
    evaluate_lstm_model,
    train_bert_model,
    train_lstm_model,
)
from src.utils import ensure_dir, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM and BERT phishing detectors.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to phishing_email.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    device = get_device()

    for directory in (LSTM_ARTIFACT_DIR, BERT_ARTIFACT_DIR, PLOTS_DIR, METRICS_DIR, PROCESSED_DIR):
        ensure_dir(directory)

    dataframe = load_and_prepare_dataframe(args.dataset)
    splits = split_dataframe(dataframe)

    dataset_summary = {
        "total_samples": len(dataframe),
        "split_sizes": splits.sizes,
        "train_distribution": class_distribution(splits.train),
        "validation_distribution": class_distribution(splits.val),
        "test_distribution": class_distribution(splits.test),
    }
    save_json(REPORTS_DIR / "dataset_summary.json", dataset_summary)
    save_json(
        LSTM_ARTIFACT_DIR / "run_config.json",
        {
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "batch_size": LSTM_BATCH_SIZE,
            "epochs": LSTM_EPOCHS,
        },
    )

    train_texts = splits.train["clean_text"].tolist()
    val_texts = splits.val["clean_text"].tolist()
    test_texts = splits.test["clean_text"].tolist()

    train_labels = splits.train["label"].tolist()
    val_labels = splits.val["label"].tolist()
    test_labels = splits.test["label"].tolist()

    vocab = build_vocabulary(
        train_texts,
        max_vocab_size=MAX_VOCAB_SIZE,
        min_frequency=MIN_TOKEN_FREQUENCY,
    )
    save_json(LSTM_ARTIFACT_DIR / "vocab.json", vocab)

    train_sequences = batch_encode_for_lstm(train_texts, vocab, max_length=MAX_SEQUENCE_LENGTH)
    val_sequences = batch_encode_for_lstm(val_texts, vocab, max_length=MAX_SEQUENCE_LENGTH)
    test_sequences = batch_encode_for_lstm(test_texts, vocab, max_length=MAX_SEQUENCE_LENGTH)

    lstm_train_loader = DataLoader(
        LSTMEmailDataset(train_sequences, train_labels),
        batch_size=LSTM_BATCH_SIZE,
        shuffle=True,
    )
    lstm_val_loader = DataLoader(
        LSTMEmailDataset(val_sequences, val_labels),
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False,
    )
    lstm_test_loader = DataLoader(
        LSTMEmailDataset(test_sequences, test_labels),
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False,
    )

    lstm_model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=LSTM_EMBEDDING_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        num_classes=len(CLASS_NAMES),
    ).to(device)
    lstm_optimizer = Adam(
        lstm_model.parameters(),
        lr=LSTM_LEARNING_RATE,
        weight_decay=LSTM_WEIGHT_DECAY,
    )
    lstm_model, lstm_history = train_lstm_model(
        lstm_model,
        lstm_train_loader,
        lstm_val_loader,
        device,
        lstm_optimizer,
        epochs=LSTM_EPOCHS,
    )
    _, _, lstm_test_y, lstm_test_probabilities = evaluate_lstm_model(
        lstm_model,
        lstm_test_loader,
        device,
    )
    lstm_metrics = compute_metrics(lstm_test_y, lstm_test_probabilities)
    torch.save(lstm_model.state_dict(), LSTM_ARTIFACT_DIR / "model.pt")
    save_metrics_bundle(
        lstm_metrics,
        lstm_history,
        LSTM_ARTIFACT_DIR / "metrics.json",
        LSTM_ARTIFACT_DIR / "history.json",
    )
    plot_training_history(lstm_history, "LSTM", PLOTS_DIR / "lstm_training_curves.png")
    plot_confusion_matrix(lstm_metrics, "LSTM", PLOTS_DIR / "lstm_confusion_matrix.png")
    plot_roc_curve(lstm_metrics, "LSTM", PLOTS_DIR / "lstm_roc_curve.png")

    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_train_loader = DataLoader(
        BERTEmailDataset(train_texts, train_labels, bert_tokenizer, MAX_SEQUENCE_LENGTH),
        batch_size=BERT_BATCH_SIZE,
        shuffle=True,
    )
    bert_val_loader = DataLoader(
        BERTEmailDataset(val_texts, val_labels, bert_tokenizer, MAX_SEQUENCE_LENGTH),
        batch_size=BERT_BATCH_SIZE,
        shuffle=False,
    )
    bert_test_loader = DataLoader(
        BERTEmailDataset(test_texts, test_labels, bert_tokenizer, MAX_SEQUENCE_LENGTH),
        batch_size=BERT_BATCH_SIZE,
        shuffle=False,
    )

    bert_model = create_bert_classifier(BERT_MODEL_NAME, num_labels=len(CLASS_NAMES)).to(device)
    bert_optimizer = AdamW(
        bert_model.parameters(),
        lr=BERT_LEARNING_RATE,
        weight_decay=BERT_WEIGHT_DECAY,
    )
    bert_model, bert_history = train_bert_model(
        bert_model,
        bert_train_loader,
        bert_val_loader,
        device,
        bert_optimizer,
        epochs=BERT_EPOCHS,
    )
    _, _, bert_test_y, bert_test_probabilities = evaluate_bert_model(
        bert_model,
        bert_test_loader,
        device,
    )
    bert_metrics = compute_metrics(bert_test_y, bert_test_probabilities)
    bert_model.save_pretrained(BERT_ARTIFACT_DIR)
    bert_tokenizer.save_pretrained(BERT_ARTIFACT_DIR)
    save_metrics_bundle(
        bert_metrics,
        bert_history,
        BERT_ARTIFACT_DIR / "metrics.json",
        BERT_ARTIFACT_DIR / "history.json",
    )
    save_json(
        BERT_ARTIFACT_DIR / "run_config.json",
        {
            "model_name": BERT_MODEL_NAME,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "batch_size": BERT_BATCH_SIZE,
            "epochs": BERT_EPOCHS,
        },
    )
    plot_training_history(bert_history, "BERT", PLOTS_DIR / "bert_training_curves.png")
    plot_confusion_matrix(bert_metrics, "BERT", PLOTS_DIR / "bert_confusion_matrix.png")
    plot_roc_curve(bert_metrics, "BERT", PLOTS_DIR / "bert_roc_curve.png")

    better_model = "BERT" if bert_metrics["auc_score"] >= lstm_metrics["auc_score"] else "LSTM"
    comparison = {
        "bert": bert_metrics,
        "lstm": lstm_metrics,
        "better_model": better_model,
    }
    save_json(METRICS_DIR / "comparison.json", comparison)
    write_comparison_report(comparison, REPORTS_DIR / "model_comparison.md")
    plot_model_comparison(comparison, PLOTS_DIR / "model_comparison.png")

    print("Training complete.")
    print(f"Dataset size after cleaning: {len(dataframe)}")
    print(f"Split sizes: {splits.sizes}")
    print(f"BERT metrics: {bert_metrics}")
    print(f"LSTM metrics: {lstm_metrics}")
    print(f"Better overall model: {better_model}")


if __name__ == "__main__":
    main()
