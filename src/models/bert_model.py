from transformers import AutoModelForSequenceClassification


def create_bert_classifier(model_name: str, num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
