# Phishing Email Detection System

This project implements the report-specified end-to-end phishing email detection system with two trained deep learning models:

- `BERT (bert-base-uncased)` fine-tuned for binary classification
- `Bidirectional LSTM` with `64`-dimensional embeddings, `2` recurrent layers, `64` hidden units, and `0.6` dropout

The pipeline includes:

- dataset cleaning and deduplication
- stratified `70/10/20` train/validation/test split
- preprocessing for both models
- training and validation curves
- full evaluation metrics for both models
- a Flask application that shows both model predictions side by side

## Project Structure

```text
phish_email/
├── app.py
├── train.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── data/
│   └── phishing_email.csv            # place dataset here
├── artifacts/
│   ├── bert/
│   ├── lstm/
│   └── processed/
├── reports/
│   ├── plots/
│   ├── metrics/
│   └── model_comparison.md
└── src/
    ├── config.py
    ├── data.py
    ├── evaluate.py
    ├── inference.py
    ├── preprocessing.py
    ├── trainer.py
    ├── utils.py
    └── models/
        ├── bert_model.py
        └── lstm_model.py
```

## Dataset Expectations

The code expects `phishing_email.csv` with:

- one text column such as `text`, `email_text`, `body`, `message`, or `content`
- one binary label column such as `label`, `target`, `class`, `is_phishing`, or `phishing`

Accepted positive labels include values such as `phishing`, `phish`, `spam`, `1`, `true`, `yes`, and `malicious`.

The report states:

- original dataset size: `82,486`
- cleaned and deduplicated working size: approximately `15,000`
- split: `10,500` train, `1,500` validation, `3,000` test

This implementation enforces those rules where possible by deduplicating and then taking a stratified sample of up to `15,000` records.

## Setup

Use Python `3.10+`. The current machine is running Python `3.13`, so install the latest compatible wheels for your platform.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place the dataset at:

```bash
data/phishing_email.csv
```

## Train Both Models

```bash
python train.py --dataset data/phishing_email.csv
```

This will:

- clean and sample the dataset
- create train/validation/test splits
- train the LSTM model
- fine-tune `bert-base-uncased`
- save model artifacts
- generate training curves, confusion matrices, ROC curves, and metrics JSON files
- create a Markdown comparison summary

## Run the Flask App

After training completes:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Expected Outputs

After training, the project generates:

- `artifacts/lstm/model.pt`
- `artifacts/lstm/vocab.json`
- `artifacts/lstm/history.json`
- `artifacts/lstm/metrics.json`
- `artifacts/bert/` saved Hugging Face model + tokenizer
- `artifacts/bert/history.json`
- `artifacts/bert/metrics.json`
- `reports/plots/` training curves, confusion matrices, ROC curves
- `reports/model_comparison.md`

The report's expected benchmark values are:

- BERT: `95.5%` accuracy, `95.6%` precision, `95.8%` recall, `95.7%` F1, AUC about `0.99`
- LSTM: `95.25%` accuracy, `95.6%` precision, `95.2%` recall, `95.4%` F1, AUC about `0.985`
- false positive rate below `5%`

Actual values will depend on the exact contents of `phishing_email.csv`.

## Engineering Notes

- The report has one internal inconsistency in Chapter 4 claiming LSTM slightly outperformed BERT, but the abstract, summary, and embedded comparison chart all show BERT ahead overall. The generated comparison report uses the actual evaluated metrics from your run.
- The report also describes a bidirectional LSTM with `64` hidden units per direction. In PyTorch this produces a `128`-feature concatenated representation before the final binary classifier, which is the implementation used here.
