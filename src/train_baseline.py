from preprocess import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
train_df = load_data("data/train.tsv")
test_df = load_data("data/test.tsv")

# Vectorize text
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    stop_words="english",
    min_df=2
)

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_test = test_df["label"]

# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1.5, 1: 1}  # give fake more importance
)
model.fit(X_train, y_train)

# Save for evaluation
import pickle
pickle.dump(model, open("data/baseline_model.pkl", "wb"))
pickle.dump(vectorizer, open("data/vectorizer.pkl", "wb"))

print("Baseline training complete!")