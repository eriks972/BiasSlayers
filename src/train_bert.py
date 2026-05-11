from preprocess import load_data
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
import pandas as pd
import numpy as np

# -----------------------
# 0. Device
# -----------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# 1. Load Data
# -----------------------
train_df = load_data("data/train.tsv")
test_df = load_data("data/test.tsv")
validation_df = load_data("data/valid.tsv")

print("\nOriginal Training Distribution:")
print(train_df["label"].value_counts())

# -----------------------
# 2. Upsample Fake Class
# -----------------------
# fake_df = train_df[train_df["label"] == 0]
# real_df = train_df[train_df["label"] == 1]

# fake_df = fake_df.sample(len(real_df), replace=True, random_state=42)

# train_df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced Training Distribution:")
print(train_df["label"].value_counts())

print("\nValidation Distribution:")
print(validation_df["label"].value_counts())

print("\nTest Distribution:")
print(test_df["label"].value_counts())

# -----------------------
# 3. Convert to HF Dataset
# -----------------------
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
validation_dataset = Dataset.from_pandas(validation_df)

# -----------------------
# 4. Tokenizer
# -----------------------
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
validation_dataset = validation_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
validation_dataset = validation_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------
# 5. Model
# -----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="single_label_classification"
)

model.to(device)

# -----------------------
# 6. Metrics
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    fake_threshold = 0.45

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    fake_probs = probs[:, 0]

    preds = np.where(
        fake_probs >= fake_threshold,
        0,
        1
    )

    print("\nClassification Report:")
    print(classification_report(
        labels,
        preds,
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "fake_f1": f1_score(labels, preds, pos_label=0, zero_division=0),
        "real_f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "fake_recall": recall_score(labels, preds, pos_label=0, zero_division=0),
        "real_recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "fake_precision": precision_score(labels, preds, pos_label=0, zero_division=0),
        "real_precision": precision_score(labels, preds, pos_label=1, zero_division=0),
    }

# -----------------------
# 7. Training Config
# -----------------------
training_args = TrainingArguments(
    output_dir="models/roberta_checkpoints",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=8,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    save_total_limit=2,
    logging_steps=100,

    use_cpu=(device == "cpu"),
    report_to="none"
)

# -----------------------
# 8. Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# -----------------------
# 9. Train
# -----------------------
trainer.train()

# -----------------------
# 10. Evaluate on Validation
# -----------------------
validation_results = trainer.evaluate(validation_dataset)
print("\nFinal Validation Results:")
print(validation_results)

# -----------------------
# 11. Evaluate on Test
# -----------------------
test_results = trainer.evaluate(test_dataset)
print("\nFinal Test Results:")
print(test_results)

# -----------------------
# 12. Save Model
# -----------------------
trainer.save_model("models/roberta")
tokenizer.save_pretrained("models/roberta")

print("\nModel saved to models/roberta")