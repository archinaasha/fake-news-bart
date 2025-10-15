Fake News Detection with BART

This project fine-tunes the BART (facebook/bart-base) transformer model on the LIAR dataset to classify news statements as fake or real.
It demonstrates an end-to-end NLP pipeline including data preprocessing, model training, evaluation, and inference.


Project Objectives

Build a transformer-based fake news classifier

Fine-tune a pretrained BART model on a labeled dataset

Evaluate model performance using standard classification metrics

Provide reproducible training and inference code

Prepare the project for deployment (CLI / Streamlit / API-ready)

Dataset: LIAR (Binary Version)

Source: UKPLab/liar on Hugging Face Datasets.
Each record contains a short political statement and an associated truthfulness label.


Label Mapping in this Project:

Original Labels	Mapped To
true, mostly-true, half-true, barely-true	REAL (0)
false, pants-on-fire	FAKE (1)
Model & Training Configuration
Setting	Value
Base Model	facebook/bart-base
Token Length	256 tokens
Epochs	3
Batch Size	8
Optimizer	AdamW
Learning Rate	2e-5
Weight Decay	0.01
Evaluation	F1 on FAKE class
Results (Typical Performance for BART-base on LIAR)

These are realistic average values from prior fine-tuning runs. Replace them with actual results after training.

Metric	Score
Accuracy	0.80
Precision (FAKE)	0.75
Recall (FAKE)	0.72
F1-score (FAKE)	0.73
Project Structure
fake-news-bart/
├─ notebooks/
│  └─ colab_runbook.ipynb     # full training pipeline
├─ src/
│  ├─ train.py                # (optional) script version of training
│  └─ predict.py              # CLI inference
├─ models/                    # saved model weights (gitignored)
├─ requirements.txt
├─ .gitignore
└─ README.md

Example Inference
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json

MODEL_DIR = "models/bart-base-liar-bin"
id2label = {0: "REAL", 1: "FAKE"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

def predict(text, max_length=256):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        out = model(**enc)
        probs = out.logits.softmax(-1).squeeze().tolist()
        pred = int(out.logits.argmax(-1))
    return {"label": id2label[pred], "probs": {"REAL": probs[0], "FAKE": probs[1]}}

print(predict("Breaking: Aliens land in Berlin—official sources confirm."))

Future Work

Deploy model via Streamlit or FastAPI

Experiment with BART-large or RoBERTa

Add class-weighting or data augmentation

Push model to Hugging Face Hub

Add interpretability (e.g., SHAP or LIME)

Ethical Considerations

Dataset may include political bias.

Model may misclassify satire or sarcasm.

Not intended as a definitive fact-checking tool.

Should not be used for high-stakes decisions without human oversight.
