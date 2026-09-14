import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page setup
st.set_page_config(page_title="Amazon Review Sentiment Analyzer", page_icon="🛍️", layout="centered")

MODEL_NAME = "jitheshpoojary/amazon-roberta-sentiment"
MAX_LENGTH = 256
LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Custom CSS
st.markdown(
    """
<style>
.main-title {text-align:center;font-size:2.5rem;font-weight:700;margin-bottom:0.2rem;}
.subtitle {text-align:center;color:#777;font-size:1.05rem;margin-bottom:1.5rem;}
.prediction-card {padding:1.25rem;border-radius:15px;text-align:center;border:1px solid rgba(128,128,128,0.25);margin:0.8rem 0 1rem;}
.prediction-label {font-size:2rem;font-weight:700;margin-bottom:0.25rem;}
.confidence {font-size:1.05rem;color:#777;}
.info-card {padding:1rem 1.2rem;border-radius:12px;border:1px solid rgba(128,128,128,0.2);margin-top:1rem;}
.warning-card {padding:1.15rem 1.3rem;border-radius:12px;border:1px solid rgba(255,165,0,0.35);background-color:rgba(255,165,0,0.08);margin-top:0.5rem;}
.warning-title {font-size:1.05rem;font-weight:700;margin-bottom:0.7rem;}
.warning-card p {margin:0.55rem 0;line-height:1.55;}
.warning-card ul {margin:0.35rem 0.6rem;padding-left:1.4rem;}
.warning-card li {margin-bottom:0.25rem;line-height:1.45;}
.footer {text-align:center;color:#777;margin:1.5rem 0 0.5rem;}
</style>
""",
    unsafe_allow_html=True
)

@st.cache_resource
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model

def predict_sentiment(review, tokenizer, model):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_id = torch.argmax(probabilities).item()
    prediction = LABELS[predicted_id]
    confidence = probabilities[predicted_id].item()
    probability_dict = {lbl: probabilities[i].item() for i, lbl in LABELS.items()}
    return prediction, confidence, probability_dict

# Header
st.markdown('<div class="main-title">🛍️ Amazon Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze customer reviews using a fine-tuned RoBERTa-base model</div>', unsafe_allow_html=True)

# Load model
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()

# Examples
examples = {
    "Positive example": "I absolutely love this product! The quality is excellent and it works perfectly.",
    "Neutral example": "The product is okay. It works as expected but there is nothing special about it.",
    "Negative example": "Very disappointed with this purchase. The product stopped working after only a few days.",
    "Mixed / informal": "good product but delivery was late and packaging was not great"
}
selected_example = st.selectbox("Choose a sample review", ["None"] + list(examples.keys()))

if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if selected_example != "None":
    st.session_state.review_text = examples[selected_example]

review = st.text_area("📝 Enter an Amazon review", value=st.session_state.review_text, height=180, max_chars=5000,
                      placeholder="Example: The product quality is amazing and I would definitely recommend it...")

if review.strip():
    col1, col2 = st.columns(2)
    col1.caption(f"📝 Words: {len(review.split())}")
    col2.caption(f"🔤 Characters: {len(review)}")

col1, col2 = st.columns(2)
analyze = col1.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
clear = col2.button("🗑️ Clear", use_container_width=True)

if clear:
    st.session_state.review_text = ""
    st.rerun()

if analyze:
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        with st.spinner("Analyzing review..."):
            prediction, confidence, probabilities = predict_sentiment(review, tokenizer, model)

        icon = "🟢" if prediction == "Positive" else "🔴" if prediction == "Negative" else "🟡"
        st.markdown(
            f"""
<div class="prediction-card">
    <div class="prediction-label">{icon} {prediction}</div>
    <div class="confidence">Model confidence: <strong>{confidence:.2%}</strong></div>
</div>
""",
            unsafe_allow_html=True
        )

        st.subheader("📊 Sentiment Probability")
        probability_df = pd.DataFrame({"Sentiment": list(probabilities.keys()),
                                       "Probability": [f"{v:.2%}" for v in probabilities.values()]})
        st.dataframe(probability_df, hide_index=True, use_container_width=True)
        for sentiment, probability in probabilities.items():
            st.write(f"**{sentiment}** — {probability:.2%}")
            st.progress(probability)

        st.subheader("💬 Interpretation")
        if prediction == "Positive":
            st.success("The model identifies this review as predominantly positive.")
        elif prediction == "Negative":
            st.error("The model identifies this review as predominantly negative.")
        else:
            st.warning("The model identifies this review as predominantly neutral.")

st.divider()
st.subheader("🤖 Model Information")
col1, col2, col3 = st.columns(3)
col1.metric("Model", "RoBERTa-base")
col2.metric("Accuracy", "91%")
col3.metric("Macro F1", "80%")
st.caption("Three-class sentiment classification: Negative • Neutral • Positive")

st.markdown(
    """
<div class="warning-card">
    <div class="warning-title">⚠️ Model Limitation</div>
    <p>This prototype uses a fine-tuned RoBERTa-base model. Training was limited, so performance may be lower on difficult or ambiguous reviews.</p>
    <p>The model may be less reliable when:</p>
    <ul>
        <li>Reviews contain spelling mistakes or informal text</li>
        <li>Sentiment is mixed or ambiguous</li>
        <li>Reviews are very short or lack context</li>
        <li>Neutral sentiment is difficult to distinguish</li>
        <li>Text contains unusual expressions or slang</li>
    </ul>
    <p><strong>Note:</strong> Neutral sentiment was the most challenging class during evaluation. Reported test performance: <strong>91% Accuracy</strong>, <strong>80% Macro F1</strong>.</p>
</div>
""",
    unsafe_allow_html=True
)

st.markdown('<div class="footer"><small>Built with 🤗 Transformers + PyTorch + Streamlit</small></div>', unsafe_allow_html=True)
