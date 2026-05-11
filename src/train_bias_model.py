import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "models/bias_roberta3"  # changed to avoid overwriting previous model
DATA_PATH = "data/bias_dataset_v2.csv"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# LOAD DATA
# -----------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Keep only needed columns
df = df[["text", "label"]]

# Drop bad rows
df = df.dropna()
df = df[df["text"].str.len() > 200]

# -----------------------
# BALANCE DATA
# -----------------------
counts = df["label"].value_counts()
min_count = counts.min()

df = df.groupby("label").sample(n=min_count, random_state=42)

print("\nBalanced dataset:")
print(df["label"].value_counts())

# -----------------------
# SPLIT DATA
# -----------------------
dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(test_size=0.1)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# -----------------------
# TOKENIZER
# -----------------------
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

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------
# MODEL
# -----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

model.to(device)

# -----------------------
# METRICS
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# -----------------------
# TRAINING CONFIG
# -----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # slightly higher for small dataset
    learning_rate=2e-5,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# -----------------------
# TRAINER
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# -----------------------
# TRAIN
# -----------------------
trainer.train()

# -----------------------
# SAVE
# -----------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Bias model trained and saved!")