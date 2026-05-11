from datasets import load_dataset
import pandas as pd
import os

# Load dataset
dataset = load_dataset("tweet_eval", "sentiment")

# Label mapping (already matches what we want)
label_map = {
    0: 0,  # Negative
    1: 1,  # Neutral
    2: 2   # Positive
}

def convert(split):
    texts = dataset[split]["text"]
    labels = dataset[split]["label"]

    df = pd.DataFrame({
        "text": texts,
        "label": [label_map[l] for l in labels]
    })

    return df

# Convert splits
train_df = convert("train")
test_df = convert("test")
valid_df = convert("validation")

# Create folder if needed
os.makedirs("data", exist_ok=True)

# Save as TSV (matches your pipeline)
train_df.to_csv("data/tone_train.tsv", sep="\t", index=False)
test_df.to_csv("data/tone_test.tsv", sep="\t", index=False)
valid_df.to_csv("data/tone_valid.tsv", sep="\t", index=False)

print("✅ Tone dataset ready!")
print(train_df.head())