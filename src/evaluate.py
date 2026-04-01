import pickle
from preprocess import load_data
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

# Load model + vectorizer
model = pickle.load(open("data/baseline_model.pkl", "rb"))
vectorizer = pickle.load(open("data/vectorizer.pkl", "rb"))

# Load test data
test_df = load_data("data/test.tsv")

X_test = vectorizer.transform(test_df["text"])
y_test = test_df["label"]

# Predict
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))