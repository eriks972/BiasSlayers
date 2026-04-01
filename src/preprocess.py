import pandas as pd

def load_data(path):
    df = pd.read_csv(path, sep="\t", header=None)

    df = df[[2, 1]]  # text, label
    df.columns = ["text", "label"]

    # Map labels to binary
    real_labels = ["true", "mostly-true", "half-true"]
    fake_labels = ["false", "pants-fire"]

    def map_label(label):
        if label in real_labels:
            return 1  # REAL
        elif label in fake_labels:
            return 0  # FAKE
        else:
            return None

    df["label"] = df["label"].apply(map_label)
    df = df.dropna()

    return df