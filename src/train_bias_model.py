import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data/bias_train.csv")

# Features
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(df["text"])
y = df["bias_label"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
print(classification_report(y, y_pred))