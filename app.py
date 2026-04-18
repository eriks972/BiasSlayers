from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model + tokenizer
model_path = "models/roberta"

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.to(device)
model.eval()

def get_explanation(text):
    words = text.split()
    return "Key phrases: " + ", ".join(words[:8])

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return pred, confidence, probs

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    text = data.get("text", "")

    if not text or len(text.strip()) < 5:
        return jsonify({"error": "Text too short"}), 400

    pred, confidence, probs = predict(text)

    return jsonify({
    "label": "Real" if pred == 1 else "Fake",
    "confidence": round(confidence, 3),
    "fake_prob": round(probs[0][0].item(), 3),
    "real_prob": round(probs[0][1].item(), 3),
    "explanation": get_explanation(text)
    })

if __name__ == "__main__":
    app.run(debug=True)