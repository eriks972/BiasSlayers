import pandas as pd
from preprocess import load_data
from textblob import TextBlob

# Load data
df = load_data("data/train.tsv")

def get_bias_label(text):
    blob = TextBlob(text)
    
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Rule-based labeling
    if subjectivity > 0.5 or abs(polarity) > 0.3:
        return 1  # BIASED
    else:
        return 0  # NEUTRAL

df["bias_label"] = df["text"].apply(get_bias_label)

# Save new dataset
df.to_csv("data/bias_train.csv", index=False)

print("Bias dataset created!")