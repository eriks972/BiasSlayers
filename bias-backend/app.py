from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS
from newspaper import Article
from transformers import pipeline
import re
from collections import Counter

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "http://localhost:3000"}},
    supports_credentials=True
)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        headers = response.headers

        headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        headers["Access-Control-Allow-Headers"] = "Content-Type"
        headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"

        return response

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# LOAD MODELS
# -----------------------
bert_path = "models/bert"
roberta_path = "models/roberta"
tone_path = "models/tone_roberta"
bias_path = "models/bias_roberta2"

tone_tokenizer= AutoTokenizer.from_pretrained(tone_path)
tone_model = AutoModelForSequenceClassification.from_pretrained(
    tone_path, local_files_only=True
)

# BERT
bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(
    bert_path, local_files_only=True
)

# RoBERTa
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path)
roberta_model = AutoModelForSequenceClassification.from_pretrained(
    roberta_path, local_files_only=True
)

# Bias
bias_tokenizer = AutoTokenizer.from_pretrained(bias_path)
bias_model = AutoModelForSequenceClassification.from_pretrained(
    bias_path, local_files_only=True
)

bert_model.to(device).eval()
roberta_model.to(device).eval()
tone_model.to(device).eval()
bias_model.to(device).eval()

# -----------------------
# HELPERS
# -----------------------
def get_explanation(text):
    words = text.split()
    return "Key phrases: " + ", ".join(words[:8])


def run_model(text, model, tokenizer):
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
        probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return {
        "label": "Real" if pred == 1 else "Fake",
        "confidence": round(confidence, 3),
        "fake_prob": round(probs[0][0].item(), 3),
        "real_prob": round(probs[0][1].item(), 3)
    }


def combine_results(bert_result, roberta_result):
    if bert_result["label"] == roberta_result["label"]:
        return {
            "label": bert_result["label"],
            "confidence": round(
                (bert_result["confidence"] + roberta_result["confidence"]) / 2, 3
            ),
            "agreement": True
        }
    else:
        better = (
            bert_result
            if bert_result["confidence"] > roberta_result["confidence"]
            else roberta_result
        )

        return {
            "label": better["label"],
            "confidence": better["confidence"],
            "agreement": False
        }

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None

def get_tone(text):
    inputs = tone_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = tone_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    return {
        "tone": label_map[pred],
        "confidence": round(confidence, 3),
        "negative": round(probs[0][0].item(), 3),
        "neutral": round(probs[0][1].item(), 3),
        "positive": round(probs[0][2].item(), 3)
    }

def split_sentences(text):
    # Simple but effective sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def get_sentence_tone(text):
    sentences = split_sentences(text)

    results = []

    for sentence in sentences:
        inputs = tone_tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = tone_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

        label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        results.append({
            "sentence": sentence,
            "tone": label_map[pred],
            "confidence": round(confidence, 3)
        })

    return results


def get_bias(text):
    inputs = bias_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bias_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()

    label_map = {
        0: "Left",
        1: "Neutral",
        2: "Right"
    }

    return {
        "label": label_map[pred],
        "confidence": round(probs[0][pred].item(), 3),
        "left": round(probs[0][0].item(), 3),
        "neutral": round(probs[0][1].item(), 3),
        "right": round(probs[0][2].item(), 3)
    }

def extract_keywords(text, top_k=10):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    common = Counter(words).most_common(top_k)
    return [w for w, _ in common]


def detect_loaded_words(text):
    loaded = [
        "revenge", "attack", "pressure", "demand", "crisis",
        "threat", "radical", "extreme", "collapse", "failure"
    ]
    found = [w for w in loaded if w in text.lower()]
    return found


def detect_entities(text):
    # Simple heuristic (you can upgrade later with spaCy)
    entities = []
    candidates = ["trump", "biden", "republican", "democrat", "gop"]

    for c in candidates:
        if c in text.lower():
            entities.append(c.capitalize())

    return entities


def generate_bias_explanation(text, bias_result, tone_result):
    keywords = extract_keywords(text)
    loaded_words = detect_loaded_words(text)
    entities = detect_entities(text)

    tone = tone_result["tone"]

    # Simple reasoning logic
    if tone == "Negative":
        tone_skew = "Negative framing detected"
    elif tone == "Positive":
        tone_skew = "Positive framing detected"
    else:
        tone_skew = "Neutral tone"

    summary = f"Text contains emotionally loaded language and focuses on {', '.join(entities) if entities else 'key political actors'}."

    return {
        "keywords": keywords,
        "loaded_words": loaded_words,
        "entities": entities,
        "tone_skew": tone_skew,
        "summary": summary
    }

# -----------------------
# ROUTES
# -----------------------

# 🔹 Single model (kept for compatibility)
@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json

    text = data.get("text", "")
    url = data.get("url", "")

    # Extract from URL if provided
    if url:
        extracted = extract_text_from_url(url)
        if not extracted:
            return jsonify({"error": "Failed to extract article"}), 400
        text = extracted

    if not text or len(text.strip()) < 5:
        return jsonify({"error": "Text too short"}), 400

    # 🔥 Use BERT as default single model
    result = run_model(text, bert_model, bert_tokenizer)

    return jsonify({
        **result,
        "explanation": get_explanation(text),
        "preview": text[:500]
    })


# 🔥 NEW: Multi-model endpoint
@app.route("/predict_all", methods=["POST", "OPTIONS"])
def predict_all():
    data = request.json

    text = data.get("text", "")
    url = data.get("url", "")

    # 🔥 If URL provided → extract article text
    if url:
        extracted = extract_text_from_url(url)

        if not extracted:
            return jsonify({"error": "Failed to extract article"}), 400

        text = extracted

    # Validate text
    if not text or len(text.strip()) < 5:
        return jsonify({"error": "Text too short"}), 400

    # 🔥 Run both models
    bert_result = run_model(text, bert_model, bert_tokenizer)
    roberta_result = run_model(text, roberta_model, roberta_tokenizer)
    tone_result = get_tone(text)
    sentence_tone = get_sentence_tone(text)
    bias_result = get_bias(text)

    # 🔥 Combine results
    combined = combine_results(bert_result, roberta_result)
    bias_explanation = generate_bias_explanation(text, bias_result, tone_result)

    return jsonify({
        "bert": bert_result,
        "roberta": roberta_result,
        "combined": combined,
        "explanation": get_explanation(text),
        "preview": text[:800],  # 👈 helps UI show extracted content
        "source": url if url else "manual",
        "tone": tone_result,
        "sentence_tone": sentence_tone,
        "bias": bias_result,
        "bias_explanation": bias_explanation,  # 👈 ADD THIS
    })

@app.route("/", methods=["GET"])
def home():
    return "Backend is running"

# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
    print("🔥 APP STARTED")
    print(app.url_map)