from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CLASS_NAMES
from src.utils import ensure_dir, save_json


def compute_metrics(y_true, y_probabilities, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_probabilities = np.asarray(y_probabilities)
    y_predictions = (y_probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_predictions)
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_probabilities)
    auc_score = roc_auc_score(y_true, y_probabilities)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_predictions)),
        "precision": float(precision_score(y_true, y_predictions, zero_division=0)),
        "recall": float(recall_score(y_true, y_predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_predictions, zero_division=0)),
        "auc_score": float(auc_score),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr_curve.tolist(),
            "tpr": tpr_curve.tolist(),
            "thresholds": roc_thresholds.tolist(),
            "auc": float(auc(fpr_curve, tpr_curve)),
        },
    }


def plot_training_history(history: dict, model_name: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(metrics: dict, model_name: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    matrix = np.array(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_title(f"{model_name} Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(metrics: dict, model_name: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    roc_data = metrics["roc_curve"]
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        roc_data["fpr"],
        roc_data["tpr"],
        label=f"{model_name} (AUC = {metrics['auc_score']:.3f})",
        linewidth=2,
        color="#e08a00",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#243b71", label="Random classifier")
    axis.set_title(f"{model_name} ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    bert_values = [comparison["bert"][metric] * 100 for metric in metric_names]
    lstm_values = [comparison["lstm"][metric] * 100 for metric in metric_names]

    indices = np.arange(len(metric_names))
    width = 0.35

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.bar(indices - width / 2, bert_values, width, label="BERT", color="#477db3")
    axis.bar(indices + width / 2, lstm_values, width, label="LSTM", color="#53a03f")
    axis.set_xticks(indices)
    axis.set_xticklabels(["Accuracy", "Precision", "Recall", "F1 Score"])
    axis.set_ylabel("Percentage (%)")
    axis.set_ylim(90, 100)
    axis.set_title("BERT vs LSTM - Performance Comparison")
    axis.legend()

    for idx, value in enumerate(bert_values):
        axis.text(idx - width / 2, value + 0.15, f"{value:.2f}%", ha="center", fontsize=9)
    for idx, value in enumerate(lstm_values):
        axis.text(idx + width / 2, value + 0.15, f"{value:.2f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bundle(metrics: dict, history: dict, metrics_path: Path, history_path: Path) -> None:
    save_json(metrics_path, metrics)
    save_json(history_path, history)


def write_comparison_report(comparison: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    bert = comparison["bert"]
    lstm = comparison["lstm"]
    better_model = comparison["better_model"]

    markdown = f"""# Model Comparison

| Metric | BERT | LSTM |
| --- | ---: | ---: |
| Accuracy | {bert['accuracy']:.4f} | {lstm['accuracy']:.4f} |
| Precision | {bert['precision']:.4f} | {lstm['precision']:.4f} |
| Recall | {bert['recall']:.4f} | {lstm['recall']:.4f} |
| F1 Score | {bert['f1_score']:.4f} | {lstm['f1_score']:.4f} |
| AUC Score | {bert['auc_score']:.4f} | {lstm['auc_score']:.4f} |
| False Positive Rate | {bert['false_positive_rate']:.4f} | {lstm['false_positive_rate']:.4f} |

**Better overall model:** {better_model}
"""
    output_path.write_text(markdown, encoding="utf-8")
