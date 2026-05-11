import pandas as pd

def load_data(path):
    # LIAR format has no header:
    # 0 = id
    # 1 = truth label
    # 2 = statement text
    df = pd.read_csv(path, sep="\t", header=None)

    df = df[[2, 1]]
    df.columns = ["text", "label_text"]

    def map_label(label):
        label = str(label).strip().lower()

        real_labels = ["true", "mostly-true"]
        fake_labels = ["false", "barely-true", "pants-fire"]

        if label in real_labels:
            return 1
        elif label in fake_labels:
            return 0
        else:
            return None

    df["label"] = df["label_text"].apply(map_label)

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]