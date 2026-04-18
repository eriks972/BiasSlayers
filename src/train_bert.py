from preprocess import load_data
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

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
validation_da = load_data("data/valid.tsv")

fake_df = train_df[train_df["label"] == 0]
real_df = train_df[train_df["label"] == 1]

# Upsample fake
fake_df = fake_df.sample(len(real_df), replace=True)

train_df = pd.concat([fake_df, real_df]).sample(frac=1)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
validation_dataset = Dataset.from_pandas(validation_da)

# -----------------------
# 2. Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256   # ⚡ faster on Mac
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
validation_dataset = validation_dataset.map(tokenize, batched=True)
# Rename label column
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
validation_dataset = validation_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
validation_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -----------------------
# 3. Model
# -----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2,
    problem_type="single_label_classification"
)

model.to(device)

# -----------------------
# 4. Class Weights (FIXED)
# -----------------------
from torch.nn import CrossEntropyLoss

class_weights = torch.tensor([2.0, 1.0]).to(device)

def weighted_loss(model, inputs, return_outputs=False, **kwargs):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")

    loss_fct = CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits, labels)

    return (loss, outputs) if return_outputs else loss

# -----------------------
# 5. Metrics
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1)

    print("\nClassification Report:")
    print(classification_report(labels, preds))

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# -----------------------
# 6. Training Config
# -----------------------
training_args = TrainingArguments(
    output_dir="../models/bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,   # ⚡ better for MPS
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    use_cpu=(device == "cpu"),
    logging_steps=100
)

# -----------------------
# 7. Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)

# 🔥 Inject custom loss
trainer.compute_loss = weighted_loss

# -----------------------
# 8. Train
# -----------------------
trainer.train()

# -----------------------
# 9. Evaluate
# -----------------------
results = trainer.evaluate()
print("\nFinal Results:", results)
trainer.save_model("models/roberta")
tokenizer.save_pretrained("models/roberta")