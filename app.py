import os
import pickle
import requests
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ----------------- CONFIG -----------------
MODEL_URL = (
    "https://github.com/Sajia2217/sentiment-analyzer/"
    "releases/download/v1.0/sentiment_model.h5"
)
MODEL_PATH = "sentiment_model.h5"

# VERY IMPORTANT: keep order same as training labels
CLASS_LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
# ------------------------------------------


def ensure_model_file():
    """Download the model from GitHub Release if it's not present."""
    if os.path.exists(MODEL_PATH):
        return

    with st.spinner("Downloading sentiment model... (first run only)"):
        resp = requests.get(MODEL_URL, stream=True)
        resp.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


@st.cache_resource
def load_sentiment_model():
    ensure_model_file()
    model = load_model(MODEL_PATH)
    return model


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def get_maxlen(model):
    """Try to infer max sequence length from model input shape."""
    try:
        return model.input_shape[1]
    except Exception:
        # fallback – change if you know the exact value used in training
        return 100


# Load resources
model = load_sentiment_model()
tokenizer = load_tokenizer()
MAXLEN = get_maxlen(model)


def predict_sentiment(text: str):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_LABELS[idx]
    return label, probs


# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊")

st.title("📊Text Sentiment Analyzer")
st.write("Model: GRU sentiment model loaded from GitHub Release asset.")

with st.form("sentiment_form"):
    user_text = st.text_area(
        "Enter a message / signal text:",
        height=150,
        placeholder="Example: Telegram messages write to need here....",
    )
    submitted = st.form_submit_button("Analyze Sentiment")

if submitted:
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        label, probs = predict_sentiment(user_text)

        st.subheader("Prediction")
        st.markdown(f"**Predicted sentiment:** `{label}`")

        st.subheader("Class probabilities")
        prob_dict = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
        st.json(prob_dict)

        st.bar_chart(prob_dict)

st.markdown("---")
st.caption(
    "Note: If texts contain characters like 'Ã°', they come from encoding issues in "
    "the source data, but the model will still handle them if it was trained that way."
)
