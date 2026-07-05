"""AI-Generated Text Detector — Streamlit demo (BERT vs Claude, side by side).

Runs locally (`streamlit run app.py`) and on HuggingFace Spaces. The BERT weights load from
the local `models/bert/` save when present (dev machine), else from the HF Hub repo (Spaces).
The Claude side reuses the canonical prompt/parsing in `src/claude_detector.py` — the same
classifier evaluated in Notebook 4, so the demo behaves exactly like the reported model.
"""
import os
import time
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:  # local dev: pick up ANTHROPIC_API_KEY from the gitignored .env; harmless on Spaces
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.claude_detector import DEFAULT_MODEL as CLAUDE_MODEL, classify_text, cost_usd

# --- Page config -----------------------------------------------------------
st.set_page_config(page_title="AI Text Detector — BERT vs Claude", page_icon="🤖", layout="wide")

BERT_REPO = "lilhuang15/bert-ai-text-detector-reddit"
BERT_SOURCE = "models/bert" if Path("models/bert/model.safetensors").exists() else BERT_REPO
MAX_LEN = 256                                    # locked from EDA (Notebook 1)


# --- Model loading (cached across reruns) ----------------------------------
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained(BERT_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_SOURCE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_claude():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None                              # demo degrades to BERT-only, no crash
    from anthropic import Anthropic
    return Anthropic()


# --- Inference --------------------------------------------------------------
def predict_bert(text, tokenizer, model):
    t0 = time.time()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label = int(probs.argmax())
    return {
        "label": "AI" if label == 1 else "Human",
        "confidence": float(probs[label]),
        "latency_ms": (time.time() - t0) * 1000,
    }


def predict_claude(text, client):
    t0 = time.time()
    label, raw, usage = classify_text(client, text)   # canonical prompt — same as Notebook 4
    return {
        "label": {0: "Human", 1: "AI"}.get(label, f"Unparsed ({raw!r})"),
        "latency_ms": (time.time() - t0) * 1000,
        "cost": cost_usd(usage),
    }


# --- UI ----------------------------------------------------------------------
st.title("🤖 AI-Generated Text Detector")
st.markdown(
    "Compare two approaches on the same text: a **fine-tuned BERT** (110M params, trained on "
    "HC3 reddit_eli5) vs **Claude Haiku 4.5 zero-shot**."
)

text = st.text_area(
    "Paste any text to classify:", height=200,
    placeholder="Paste an essay, article, or chatbot response (a few sentences or more works best)...",
)

if st.button("🔍 Detect", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        tokenizer, model = load_bert()
        client = load_claude()

        with st.spinner("Running both models..."):
            bert_result = predict_bert(text, tokenizer, model)
            claude_result = predict_claude(text, client) if client else None

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Fine-tuned BERT")
            st.metric("Prediction", bert_result["label"])
            st.progress(bert_result["confidence"],
                        text=f"Confidence: {bert_result['confidence']:.0%}")
            st.metric("Latency", f"{bert_result['latency_ms']:.0f} ms")
            st.metric("Cost", "$0 (runs locally)")
        with col2:
            st.subheader("🤖 Claude Haiku 4.5 (zero-shot)")
            if claude_result is None:
                st.info("Claude comparison is disabled — no `ANTHROPIC_API_KEY` configured.")
            else:
                st.metric("Prediction", claude_result["label"])
                st.write("Confidence: n/a (zero-shot — no calibrated probability)")
                st.metric("Latency", f"{claude_result['latency_ms']:.0f} ms")
                st.metric("Cost", f"${claude_result['cost']:.4f}")

st.divider()
with st.expander("About this project"):
    st.markdown(
        """
        A 5-model comparison of AI-text detectors on **HC3** (human vs ChatGPT answers), with a
        cross-domain generalization study and a full error analysis.

        - **This BERT**: macro-F1 **0.9935** on the held-out reddit test set; cross-domain it stays
          robust (finance **0.9812**, medicine **0.9821**) because it keys on the generator's
          *style*, not the topic.
        - **Known limits**: trained on 2023 GPT-3.5 text — a different generator or an unusually
          *short* AI answer can evade it; long, well-structured *human* writing is what it
          most often false-flags.
        - [GitHub repo](https://github.com/lilhuang15/ai-generated-text-detector) ·
          [BERT weights on HF Hub](https://huggingface.co/lilhuang15/bert-ai-text-detector-reddit)
        """
    )
