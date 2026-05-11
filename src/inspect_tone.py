import pandas as pd

TRAIN_PATH = "data/tone_train.tsv"
VALID_PATH = "data/tone_valid.tsv"
TEST_PATH = "data/tone_test.tsv"

def inspect_file(path):
    print(f"\n===== {path} =====")

    df = pd.read_csv(path, sep="\t")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

inspect_file(TRAIN_PATH)
inspect_file(VALID_PATH)
inspect_file(TEST_PATH)