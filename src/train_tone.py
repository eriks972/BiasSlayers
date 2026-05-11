# from preprocess import load_data
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np

# -----------------------
# 0. Device
# -----------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

def load_tone_data(path):
    df = pd.read_csv(path, sep="\t")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df

# -----------------------
# 1. Load Data
# -----------------------
train_df = load_tone_data("data/tone_train.tsv")
test_df = load_tone_data("data/tone_test.tsv")
valid_df = load_tone_data("data/tone_valid.tsv")

print("\nTraining Distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation Distribution:")
print(valid_df["label"].value_counts().sort_index())

print("\nTest Distribution:")
print(test_df["label"].value_counts().sort_index())

# Shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------
# 2. Class Weights
# -----------------------
label_counts = train_df["label"].value_counts().sort_index()
total = len(train_df)

weights = []

for i in range(3):
    count = label_counts[i]
    weight = total / (3 * count)
    weights.append(weight)

# Force float32 for MPS
class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print("\nClass Weights:")
print("Negative:", class_weights[0].item())
print("Neutral:", class_weights[1].item())
print("Positive:", class_weights[2].item())

# -----------------------
# 3. Convert to HF datasets
# -----------------------
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
valid_dataset = Dataset.from_pandas(valid_df)

# -----------------------
# 4. Tokenizer
# -----------------------
MODEL_NAME = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
valid_dataset = valid_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------
# 5. Model
# -----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    problem_type="single_label_classification"
)

model.to(device)

# -----------------------
# 6. Weighted Loss
# -----------------------
def weighted_loss(model, inputs, return_outputs=False, **kwargs):
    labels = inputs.get("labels")

    outputs = model(**inputs)
    logits = outputs.get("logits")

    # Match logits dtype dynamically
    weights = class_weights.to(logits.dtype)

    loss_fct = CrossEntropyLoss(weight=weights)

    loss = loss_fct(logits, labels)

    return (loss, outputs) if return_outputs else loss

# -----------------------
# 7. Metrics
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    print("\nClassification Report:")
    print(classification_report(
        labels,
        preds,
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    ))

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "negative_f1": f1_score(labels, preds, labels=[0], average="macro", zero_division=0),
        "neutral_f1": f1_score(labels, preds, labels=[1], average="macro", zero_division=0),
        "positive_f1": f1_score(labels, preds, labels=[2], average="macro", zero_division=0),
    }

# -----------------------
# 8. Training Config
# -----------------------
training_args = TrainingArguments(
    output_dir="models/tone_roberta_checkpoints",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,

    num_train_epochs=3,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    logging_steps=100,
    save_total_limit=2,
    report_to="none",

    use_cpu=(device == "cpu")
)

# -----------------------
# 9. Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

trainer.compute_loss = weighted_loss

# -----------------------
# 10. Train
# -----------------------
trainer.train()

# -----------------------
# 11. Evaluate Validation
# -----------------------
validation_results = trainer.evaluate(valid_dataset)
print("\nFinal Validation Results:")
print(validation_results)

# -----------------------
# 12. Evaluate Test
# -----------------------
test_results = trainer.evaluate(test_dataset)
print("\nFinal Test Results:")
print(test_results)

# -----------------------
# 13. Save Model
# -----------------------
trainer.save_model("models/tone_deberta")
tokenizer.save_pretrained("models/tone_deberta")

print("\nModel saved to models/tone_deberta")