import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------- Load model & tokenizer (cached) ----------
@st.cache_resource
def load_sentiment_model():
    model = load_model("sentiment_model.h5")
    return model


@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


# Try to infer max sequence length from model input
def get_maxlen(model):
    try:
        # usually (None, maxlen)
        return model.input_shape[1]
    except Exception:
        # fallback – change if you know the exact value
        return 100


model = load_sentiment_model()
tokenizer = load_tokenizer()
MAXLEN = get_maxlen(model)

# ⚠️ VERY IMPORTANT:
# Set this list in the SAME ORDER you used during training
# e.g. if y_train was [NEG, NEU, POS] one-hot encoded:
CLASS_LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


# ---------- Prediction helper ----------
def predict_sentiment(text: str):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]  # shape: (num_classes,)
    idx = int(np.argmax(probs))
    label = CLASS_LABELS[idx]
    return label, probs


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Crypto Sentiment Analyzer", page_icon="📊")

st.title("📊 Sentiment Analysis")
st.write("GRU model using your **binancesignals** dataset.")

with st.form("sentiment_form"):
    user_text = st.text_area(
        "Enter a message / signal text:",
        height=150,
        placeholder="Example: bitcoin etf growing, gold etf losing fund, crypto pumping..."
    )
    submitted = st.form_submit_button("Analyze Sentiment")

if submitted:
    if not user_text.strip():
        st.warning("Please type some text first.")
    else:
        label, probs = predict_sentiment(user_text)

        st.subheader("Prediction")
        st.markdown(f"**Predicted sentiment:** `{label}`")

        # Show probabilities nicely
        st.subheader("Class probabilities")
        prob_dict = {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
        st.json(prob_dict)

        # Optional bar chart
        st.bar_chart(prob_dict)


st.markdown("---")
st.caption(
    "Note: Weird characters like 'Ã°' appear from encoding issues in the original data. "
    "If your model was trained with them, it's okay to leave them as-is."
)
