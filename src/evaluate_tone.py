import os
import json
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------
# 0. Config
# -----------------------
MODEL_PATH = "models/tone_roberta"
TEST_PATH = "data/tone_test.tsv"
RESULTS_DIR = "results/tone"

os.makedirs(RESULTS_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# -----------------------
# 1. Text Cleaning
# -----------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "[URL]", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------
# 2. Load Test Data
# -----------------------
test_df = pd.read_csv(TEST_PATH, sep="\t")

test_df = test_df[["text", "label"]].dropna()
test_df["text"] = test_df["text"].apply(clean_text)
test_df["label"] = test_df["label"].astype(int)

print("\nTest set size:", len(test_df))
print("\nLabel distribution:")
print(test_df["label"].value_counts().sort_index())

test_dataset = Dataset.from_pandas(test_df)

# -----------------------
# 3. Load Model + Tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=3
)

model.to(device)

# -----------------------
# 4. Tokenize
# -----------------------
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.rename_column("label", "labels")

test_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# -----------------------
# 5. Predict
# -----------------------
trainer = Trainer(model=model)

pred_output = trainer.predict(test_dataset)

logits = pred_output.predictions
true_labels = pred_output.label_ids

probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
pred_labels = np.argmax(probs, axis=1)
confidence = np.max(probs, axis=1)

# -----------------------
# 6. Metrics
# -----------------------
accuracy = accuracy_score(true_labels, pred_labels)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    true_labels,
    pred_labels,
    average="macro",
    zero_division=0
)

precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    true_labels,
    pred_labels,
    average="weighted",
    zero_division=0
)

report = classification_report(
    true_labels,
    pred_labels,
    target_names=["Negative", "Neutral", "Positive"],
    output_dict=True,
    zero_division=0
)

print("\nTone Model Test Results")
print("----------------------------")
print("Accuracy:", accuracy)
print("Macro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
print("Macro F1:", f1_macro)
print("Weighted F1:", f1_weighted)

print("\nClassification Report:")
print(classification_report(
    true_labels,
    pred_labels,
    target_names=["Negative", "Neutral", "Positive"],
    zero_division=0
))

metrics = {
    "accuracy": accuracy,
    "macro_precision": precision_macro,
    "macro_recall": recall_macro,
    "macro_f1": f1_macro,
    "weighted_precision": precision_weighted,
    "weighted_recall": recall_weighted,
    "weighted_f1": f1_weighted,
    "classification_report": report
}

metrics_path = os.path.join(RESULTS_DIR, "tone_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# -----------------------
# 7. Save Predictions
# -----------------------
results_df = test_df.copy()

results_df["true_label"] = [LABEL_MAP[x] for x in true_labels]
results_df["predicted_label"] = [LABEL_MAP[x] for x in pred_labels]

results_df["negative_probability"] = probs[:, 0]
results_df["neutral_probability"] = probs[:, 1]
results_df["positive_probability"] = probs[:, 2]

results_df["confidence"] = confidence
results_df["correct"] = true_labels == pred_labels

predictions_path = os.path.join(RESULTS_DIR, "tone_predictions.csv")
results_df.to_csv(predictions_path, index=False)

print("\nSaved predictions to:", predictions_path)

# -----------------------
# 8. Save Errors
# -----------------------
errors_df = results_df[results_df["correct"] == False].copy()

errors_path = os.path.join(RESULTS_DIR, "tone_errors.csv")
errors_df.to_csv(errors_path, index=False)

print("Saved errors to:", errors_path)
print("Number of errors:", len(errors_df))

# -----------------------
# 9. Confusion Matrix
# -----------------------
cm = confusion_matrix(true_labels, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

disp.plot(values_format="d")
plt.title("Tone Model Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "tone_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()

print("Saved confusion matrix to:", cm_path)

# -----------------------
# 10. Error Type Summary
# -----------------------
error_summary = errors_df.groupby(
    ["true_label", "predicted_label"]
).size().reset_index(name="count").sort_values("count", ascending=False)

error_summary_path = os.path.join(RESULTS_DIR, "tone_error_summary.csv")
error_summary.to_csv(error_summary_path, index=False)

print("Saved error summary to:", error_summary_path)

# -----------------------
# 11. Confidence Bin Analysis
# -----------------------
bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ["0-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]

results_df["confidence_bin"] = pd.cut(
    results_df["confidence"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

confidence_summary = results_df.groupby("confidence_bin", observed=False).agg(
    total=("correct", "count"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean"),
    avg_confidence=("confidence", "mean")
).reset_index()

confidence_path = os.path.join(RESULTS_DIR, "tone_confidence_bins.csv")
confidence_summary.to_csv(confidence_path, index=False)

print("Saved confidence analysis to:", confidence_path)

plt.figure(figsize=(8, 5))
plt.bar(
    confidence_summary["confidence_bin"].astype(str),
    confidence_summary["accuracy"]
)

plt.ylim(0, 1)
plt.xlabel("Confidence Range")
plt.ylabel("Accuracy")
plt.title("Tone Accuracy by Confidence Range")
plt.tight_layout()

confidence_plot_path = os.path.join(RESULTS_DIR, "tone_confidence_bins.png")
plt.savefig(confidence_plot_path, dpi=300)
plt.close()

print("Saved confidence plot to:", confidence_plot_path)

# -----------------------
# 12. Summary
# -----------------------
print("\nEvaluation complete.")
print("Files created:")
print("-", metrics_path)
print("-", predictions_path)
print("-", errors_path)
print("-", cm_path)
print("-", error_summary_path)
print("-", confidence_path)
print("-", confidence_plot_path)