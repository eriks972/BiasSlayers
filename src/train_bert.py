from preprocess import load_data
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# 1. Load Data
# -----------------------
train_df = load_data("data/train.tsv")
test_df = load_data("data/test.tsv")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# -----------------------
# 2. Load Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Rename label column
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------
# 3. Load Model
# -----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    problem_type="single_label_classification"
)

model.to(device)

# -----------------------
# 4. Metrics
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# -----------------------
# 5. Training Config
# -----------------------
training_args = TrainingArguments(
    output_dir="../models/bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="../logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,   # keep small for now
    weight_decay=0.01,
    load_best_model_at_end=True,
    use_cpu=(device == "cpu")
)

# -----------------------
# 6. Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# -----------------------
# 7. Train
# -----------------------
trainer.train()

# -----------------------
# 8. Evaluate
# -----------------------
results = trainer.evaluate()
print(results)