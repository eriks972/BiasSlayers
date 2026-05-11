from flask import Flask, request, jsonify
from flask_cors import CORS
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import torch
import re
import os

app = Flask(__name__)

# Allow Vercel frontend + local development
CORS(app, resources={r"/*": {"origins": "*"}})

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# MODEL PATHS
# -----------------------
bert_path = "models/bert"
roberta_path = "models/roberta"
tone_path = "models/tone_roberta"
bias_path = "models/bias_roberta2"


def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True
    )
    model.to(device)
    model.eval()
    return tokenizer, model


print("Loading models...")

tone_tokenizer, tone_model = load_model(tone_path)
bert_tokenizer, bert_model = load_model(bert_path)
roberta_tokenizer, roberta_model = load_model(roberta_path)
bias_tokenizer, bias_model = load_model(bias_path)

print("All models loaded successfully.")


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

    inputs = {key: value.to(device) for key, value in inputs.items()}

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
                (bert_result["confidence"] + roberta_result["confidence"]) / 2,
                3
            ),
            "agreement": True
        }

    better = bert_result if bert_result["confidence"] > roberta_result["confidence"] else roberta_result

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
    except Exception as error:
        print("Article extraction error:", error)
        return None


def get_tone(text):
    inputs = tone_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

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
        "tone": label_map.get(pred, "Unknown"),
        "confidence": round(confidence, 3),
        "negative": round(probs[0][0].item(), 3),
        "neutral": round(probs[0][1].item(), 3),
        "positive": round(probs[0][2].item(), 3)
    }


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]


def get_sentence_tone(text):
    sentences = split_sentences(text)
    results = []

    for sentence in sentences[:20]:
        inputs = tone_tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

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
            "tone": label_map.get(pred, "Unknown"),
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

    inputs = {key: value.to(device) for key, value in inputs.items()}

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
        "label": label_map.get(pred, "Unknown"),
        "confidence": round(probs[0][pred].item(), 3),
        "left": round(probs[0][0].item(), 3),
        "neutral": round(probs[0][1].item(), 3),
        "right": round(probs[0][2].item(), 3)
    }


def extract_keywords(text, top_k=10):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    common = Counter(words).most_common(top_k)
    return [word for word, _ in common]


def detect_loaded_words(text):
    loaded_words = [
        "revenge", "attack", "pressure", "demand", "crisis",
        "threat", "radical", "extreme", "collapse", "failure"
    ]

    text_lower = text.lower()
    return [word for word in loaded_words if word in text_lower]


def detect_entities(text):
    entities = []
    candidates = ["trump", "biden", "republican", "democrat", "gop"]

    text_lower = text.lower()

    for candidate in candidates:
        if candidate in text_lower:
            entities.append(candidate.capitalize())

    return entities


def generate_bias_explanation(text, bias_result, tone_result):
    keywords = extract_keywords(text)
    loaded_words = detect_loaded_words(text)
    entities = detect_entities(text)

    tone = tone_result["tone"]

    if tone == "Negative":
        tone_skew = "Negative framing detected"
    elif tone == "Positive":
        tone_skew = "Positive framing detected"
    else:
        tone_skew = "Neutral tone"

    summary = (
        "Text contains emotionally loaded language and focuses on "
        f"{', '.join(entities) if entities else 'key political actors'}."
    )

    return {
        "keywords": keywords,
        "loaded_words": loaded_words,
        "entities": entities,
        "tone_skew": tone_skew,
        "summary": summary
    }


def get_text_from_request(data):
    text = data.get("text", "")
    url = data.get("url", "")

    if url:
        extracted = extract_text_from_url(url)

        if not extracted:
            return None, url, "Failed to extract article"

        text = extracted

    if not text or len(text.strip()) < 5:
        return None, url, "Text too short"

    return text, url, None


# -----------------------
# ROUTES
# -----------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Backend is running",
        "service": "BiasSlayers API",
        "endpoints": ["/predict", "/predict_all"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "device": device
    })


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_route():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True) or {}
    text, url, error = get_text_from_request(data)

    if error:
        return jsonify({"error": error}), 400

    result = run_model(text, bert_model, bert_tokenizer)

    return jsonify({
        **result,
        "explanation": get_explanation(text),
        "preview": text[:500],
        "source": url if url else "manual"
    })


@app.route("/predict_all", methods=["POST", "OPTIONS"])
def predict_all():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True) or {}
    text, url, error = get_text_from_request(data)

    if error:
        return jsonify({"error": error}), 400

    bert_result = run_model(text, bert_model, bert_tokenizer)
    roberta_result = run_model(text, roberta_model, roberta_tokenizer)
    tone_result = get_tone(text)
    sentence_tone = get_sentence_tone(text)
    bias_result = get_bias(text)

    combined = combine_results(bert_result, roberta_result)
    bias_explanation = generate_bias_explanation(text, bias_result, tone_result)

    return jsonify({
        "bert": bert_result,
        "roberta": roberta_result,
        "combined": combined,
        "explanation": get_explanation(text),
        "preview": text[:800],
        "source": url if url else "manual",
        "tone": tone_result,
        "sentence_tone": sentence_tone,
        "bias": bias_result,
        "bias_explanation": bias_explanation
    })


# -----------------------
# RUN LOCALLY ONLY
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)