def analyze_text(text):
    # Fake news model prediction
    fake_pred = fake_model.predict(text)

    # Bias model prediction
    bias_pred = bias_model.predict(text)

    return {
        "veracity": "Real" if fake_pred == 1 else "Fake",
        "bias": "Biased" if bias_pred == 1 else "Neutral"
    }