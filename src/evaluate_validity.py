import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import load_data

from preprocess import load_data
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------
# 0. Config
# -----------------------
MODEL_PATH = "models/roberta"
TEST_PATH = "data/test.tsv"
RESULTS_DIR = "results/validity"

os.makedirs(RESULTS_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

LABEL_MAP = {
    0: "Fake",
    1: "Real"
}

# -----------------------
# 1. Load Test Data
# -----------------------
test_df = load_data(TEST_PATH)

print("Test set size:", len(test_df))
print(test_df["label"].value_counts())
print(test_df.head())

def map_liar_label(label):
    label = str(label).strip().lower()

    real_labels = ["true", "mostly-true"]
    fake_labels = ["false", "barely-true", "pants-fire"]

    if label in real_labels:
        return 1
    elif label in fake_labels:
        return 0
    else:
        return None

# test_df["label"] = test_df["label_text"].apply(map_liar_label)

# test_df = test_df.dropna(subset=["text", "label"])
# test_df["label"] = test_df["label"].astype(int)

print("Test set size:", len(test_df))
print(test_df["label"].value_counts())
# print(test_df[["label_text", "label"]].head())
test_dataset = Dataset.from_pandas(test_df)

# -----------------------
# 2. Load Model + Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2
)

model.to(device)

# -----------------------
# 3. Tokenize
# -----------------------
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

test_dataset = test_dataset.map(tokenize, batched=True)

test_dataset = test_dataset.rename_column("label", "labels")

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -----------------------
# 4. Run Predictions
# -----------------------
trainer = Trainer(model=model)

pred_output = trainer.predict(test_dataset)

logits = pred_output.predictions
true_labels = pred_output.label_ids

probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
pred_labels = np.argmax(probs, axis=1)

fake_probs = probs[:, 0]
real_probs = probs[:, 1]
confidence = np.max(probs, axis=1)

# -----------------------
# 5. Compute Metrics
# -----------------------
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

report = classification_report(
    true_labels,
    pred_labels,
    target_names=["Fake", "Real"],
    output_dict=True
)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "classification_report": report
}

print("\nValidity Model Test Results")
print("----------------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=["Fake", "Real"]))

with open(os.path.join(RESULTS_DIR, "validity_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# -----------------------
# 6. Save Predictions CSV
# -----------------------
results_df = test_df.copy()

results_df["true_label"] = [LABEL_MAP[x] for x in true_labels]
results_df["predicted_label"] = [LABEL_MAP[x] for x in pred_labels]
results_df["fake_probability"] = fake_probs
results_df["real_probability"] = real_probs
results_df["confidence"] = confidence
results_df["correct"] = true_labels == pred_labels

predictions_path = os.path.join(RESULTS_DIR, "validity_predictions.csv")
results_df.to_csv(predictions_path, index=False)

print("\nSaved predictions to:", predictions_path)

# -----------------------
# 7. Save Error Examples
# -----------------------
errors_df = results_df[results_df["correct"] == False].copy()

errors_path = os.path.join(RESULTS_DIR, "validity_errors.csv")
errors_df.to_csv(errors_path, index=False)

print("Saved error examples to:", errors_path)
print("Number of errors:", len(errors_df))

# -----------------------
# 8. Confusion Matrix
# -----------------------
cm = confusion_matrix(true_labels, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake", "Real"]
)

disp.plot(values_format="d")
plt.title("Validity Model Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "validity_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()

print("Saved confusion matrix to:", cm_path)

# -----------------------
# 9. Confidence Bin Analysis
# -----------------------
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]

results_df["confidence_bin"] = pd.cut(
    results_df["confidence"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

confidence_summary = results_df.groupby("confidence_bin").agg(
    total=("correct", "count"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean"),
    avg_confidence=("confidence", "mean")
).reset_index()

confidence_path = os.path.join(RESULTS_DIR, "validity_confidence_bins.csv")
confidence_summary.to_csv(confidence_path, index=False)

print("Saved confidence bin analysis to:", confidence_path)

plt.figure(figsize=(8, 5))
plt.bar(
    confidence_summary["confidence_bin"].astype(str),
    confidence_summary["accuracy"]
)

plt.ylim(0, 1)
plt.xlabel("Confidence Range")
plt.ylabel("Accuracy")
plt.title("Validity Accuracy by Confidence Range")
plt.tight_layout()

confidence_plot_path = os.path.join(RESULTS_DIR, "validity_confidence_bins.png")
plt.savefig(confidence_plot_path, dpi=300)
plt.close()

print("Saved confidence plot to:", confidence_plot_path)

# -----------------------
# 10. Summary
# -----------------------
print("\nEvaluation complete.")
print("Files created:")
print("-", os.path.join(RESULTS_DIR, "validity_metrics.json"))
print("-", os.path.join(RESULTS_DIR, "validity_predictions.csv"))
print("-", os.path.join(RESULTS_DIR, "validity_errors.csv"))
print("-", os.path.join(RESULTS_DIR, "validity_confusion_matrix.png"))
print("-", os.path.join(RESULTS_DIR, "validity_confidence_bins.csv"))
print("-", os.path.join(RESULTS_DIR, "validity_confidence_bins.png"))