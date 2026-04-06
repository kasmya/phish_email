from copy import deepcopy

import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from src.config import BERT_WARMUP_RATIO, EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_NORM


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean().item()


def train_lstm_model(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    epochs: int,
):
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_state = None
    best_val_loss = float("inf")
    patience = 0

    for _ in range(epochs):
        model.train()
        train_losses = []
        train_accuracies = []

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # The LSTM receives padded token ids and predicts the binary class logits.
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            train_losses.append(loss.item())
            train_accuracies.append(_accuracy_from_logits(logits, labels))

        val_loss, val_accuracy, _, _ = evaluate_lstm_model(model, val_loader, device, criterion)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(val_loss))
        history["train_accuracy"].append(float(np.mean(train_accuracies)))
        history["val_accuracy"].append(float(val_accuracy))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_lstm_model(model, data_loader, device, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    losses = []
    accuracies = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            losses.append(loss.item())
            accuracies.append(_accuracy_from_logits(logits, labels))
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    return (
        float(np.mean(losses)),
        float(np.mean(accuracies)),
        all_labels,
        all_probabilities,
    )


def train_bert_model(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    epochs: int,
):
    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(total_training_steps * BERT_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_state = None
    best_val_loss = float("inf")
    patience = 0

    for _ in range(epochs):
        model.train()
        train_losses = []
        train_accuracies = []

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            # Hugging Face returns loss and logits directly when labels are supplied.
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            train_accuracies.append(_accuracy_from_logits(logits, batch["labels"]))

        val_loss, val_accuracy, _, _ = evaluate_bert_model(model, val_loader, device)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(val_loss))
        history["train_accuracy"].append(float(np.mean(train_accuracies)))
        history["val_accuracy"].append(float(val_accuracy))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_bert_model(model, data_loader, device):
    model.eval()
    losses = []
    accuracies = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            losses.append(outputs.loss.item())
            accuracies.append(_accuracy_from_logits(logits, batch["labels"]))
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    return (
        float(np.mean(losses)),
        float(np.mean(accuracies)),
        all_labels,
        all_probabilities,
    )
