"""AI-Generated Text Detector — Streamlit demo (BERT vs Claude, side by side).

Runs locally (`streamlit run app.py`) and on HuggingFace Spaces. The BERT weights load from
the local `models/bert/` save when present (dev machine), else from the HF Hub repo (Spaces).
The Claude side reuses the canonical prompt/parsing in `src/claude_detector.py` — the same
classifier evaluated in Notebook 4, so the demo behaves exactly like the reported model.
"""
import os
os.environ.setdefault("USE_TF", "0")   # transformers: PyTorch only — hosts that install the full
                                       # root requirements.txt (Streamlit Cloud) have TF present,
                                       # and importing it wastes ~1GB RAM / can crash the free tier
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
st.set_page_config(page_title="AI Text Detector — BERT vs Claude", page_icon="🤖", layout="centered")

BERT_REPO = "lilhuang15/bert-ai-text-detector-reddit"
BERT_SOURCE = "models/bert" if Path("models/bert/model.safetensors").exists() else BERT_REPO
MAX_LEN = 256                                    # locked from EDA (Notebook 1)

# Verdict color = identity (AI violet / Human teal); banner color = agreement (green/red).
CUSTOM_CSS = """<style>
.verdict-ai, .verdict-human {
  display: inline-block; padding: 6px 14px; border-radius: 6px;
  font-size: 1.35rem; font-weight: 800; line-height: 1.3; margin: 2px 0 10px;
}
.verdict-ai    { background: #F1EBFC; color: #6D28D9; border-left: 3px solid #6D28D9; }
.verdict-human { background: #E5F5F2; color: #0F766E; border-left: 3px solid #0F766E; }
.banner-agree, .banner-disagree {
  border-radius: 6px; padding: 10px 14px; margin: 4px 0 14px; font-size: 0.95rem;
}
.banner-agree    { background: #EAF6EE; color: #1E7E34; border-left: 3px solid #1E7E34; font-weight: 600; }
.banner-disagree { background: #FDEEEE; color: #B42318; border-left: 3px solid #B42318; }
.stat-strip {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.72rem; letter-spacing: .08em; color: #2563EB; margin: -8px 0 12px;
}
.conf-note { color: #64748B; font-style: italic; font-size: 0.9rem; margin-bottom: 8px; }
/* captions (latency/cost, example hint, panel identity lines): default is too faint & small */
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p {
  color: #475569 !important; opacity: 1 !important; font-size: 0.95rem !important;
}
/* the two result panels: white cards floating on the blueprint background */
div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; }
</style>"""


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


# --- Result presentation (pure helpers; colors/copy from the 2026-07-16 UI spec) ---
def fmt_latency(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def verdict_badge(label: str) -> str:
    cls = "verdict-ai" if label == "AI" else "verdict-human"
    return f'<span class="{cls}">{label}</span>'


def agreement_banner(bert_label: str, claude_label: str | None) -> str | None:
    """Banner color encodes agreement, not the verdict. None → nothing to compare
    (Claude disabled or unparsed), so no banner is rendered at all."""
    if claude_label not in ("AI", "Human"):
        return None
    if bert_label == claude_label:
        verdict = "AI-generated" if bert_label == "AI" else "Human-written"
        return f'<div class="banner-agree">✓ Both models agree: {verdict}</div>'
    note = (" Short AI text is BERT's known blind spot (length prior — see the error analysis)."
            if (bert_label, claude_label) == ("Human", "AI") else "")
    return ('<div class="banner-disagree"><b>✕ Models disagree</b> — '
            f'BERT says <b>{bert_label}</b>, Claude says <b>{claude_label}</b>.{note}</div>')


# --- UI ----------------------------------------------------------------------
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("AI-Generated Text Detector")
st.markdown("Fine-tuned **BERT** vs **Claude Haiku 4.5** zero-shot — same text, side by side.")
st.markdown(
    '<div class="stat-strip">MACRO-F1 0.9935 · CROSS-DOMAIN −1.2 PP · HC3 REDDIT_ELI5</div>',
    unsafe_allow_html=True,
)

# --- Example texts (from the HC3 test set, so predictions match the reported eval) ---
EXAMPLES = {
    "🧑 Human answer": (
        "Squatting puts the heavy lifting duties on the massive powerful muscles and thick "
        "bones of the legs . Bending puts the lifting duties on the smaller stabilizing "
        "muscles and fragile vertebra and cartilage in the spine . The spine can carry a "
        "massive load as long as it 's compressed , it 's easily injured when trying to "
        "support much weight while bent ."
    ),
    "🤖 AI answer (ChatGPT)": (
        "BBC Three is not coming off the air. BBC Three is a television channel operated by "
        "the British Broadcasting Corporation (BBC). It was originally a television channel, "
        "but it has since transitioned to an online-only service, which means that it is no "
        "longer available as a traditional television channel that you can watch on your TV. "
        "Instead, you can watch BBC Three content online, through the BBC iPlayer app or on "
        "the BBC Three website. The decision to make BBC Three an online-only service was "
        "made by the BBC in 2016 as part of a wider strategy to adapt to changing viewing "
        "habits and to make more efficient use of its resources. Despite this change, BBC "
        "Three remains a popular service, and you can still watch a wide range of "
        "high-quality programming on the channel online."
    ),
    "🕳️ Short AI — BERT's blind spot": (
        "Canada and the United States are two separate countries with their own governments "
        "and histories. They have a close relationship and share a long border, but they "
        "have never been absorbed into each other. They have always remained independent "
        "and sovereign nations."
    ),
}

def _fill_example(name: str) -> None:
    st.session_state.input_text = EXAMPLES[name]

EXAMPLE_HELP = {
    "🧑 Human answer": "A casual Reddit explanation — both models should say Human.",
    "🤖 AI answer (ChatGPT)": "A typical GPT-3.5 answer — both models should say AI.",
    "🕳️ Short AI — BERT's blind spot": "A 42-word AI answer at the ≤2nd percentile of AI training "
    "length. Per the error analysis, BERT misses it (length prior) while Claude catches it.",
}

st.caption("No text handy? Try one of these real samples:")
for col, name in zip(st.columns([0.8, 1, 1.2]), EXAMPLES):  # widest label gets the widest column
    col.button(name, on_click=_fill_example, args=(name,),
               help=EXAMPLE_HELP[name], use_container_width=True)

text = st.text_area(
    "Paste any text to classify:", height=200, key="input_text",
    placeholder="Paste an essay, article, or chatbot response (a few sentences or more works best)...",
)

_, _mid, _ = st.columns([1, 2, 1])
if _mid.button("Detect", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        tokenizer, model = load_bert()
        client = load_claude()

        claude_result, claude_error = None, None
        with st.spinner("Running both models..."):
            bert_result = predict_bert(text, tokenizer, model)
            if client:
                try:
                    claude_result = predict_claude(text, client)
                except Exception as exc:   # usage limits, network, outages — never crash the page
                    claude_error = exc

        banner = agreement_banner(bert_result["label"],
                                  claude_result["label"] if claude_result else None)
        if banner:
            st.markdown(banner, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("**Fine-tuned BERT**")
                st.caption("110M params · runs locally")
                st.markdown(verdict_badge(bert_result["label"]), unsafe_allow_html=True)
                st.progress(bert_result["confidence"],
                            text=f"Confidence: {bert_result['confidence']:.0%}")
                st.caption(f"⏱ `{fmt_latency(bert_result['latency_ms'])}` · `$0 (local)`")
        with col2:
            with st.container(border=True):
                st.markdown("**Claude Haiku 4.5**")
                st.caption("zero-shot · Anthropic API")
                if claude_result is None:
                    if claude_error is not None:
                        st.warning("Claude comparison is temporarily unavailable — the API "
                                   "returned an error. The BERT verdict on the left is unaffected.")
                        st.caption(f"`{type(claude_error).__name__}`")
                    else:
                        st.info("Claude comparison is disabled — no `ANTHROPIC_API_KEY` configured.")
                else:
                    if claude_result["label"] in ("AI", "Human"):
                        st.markdown(verdict_badge(claude_result["label"]), unsafe_allow_html=True)
                        st.markdown('<div class="conf-note">confidence n/a — zero-shot</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.write(claude_result["label"])   # Unparsed(raw) — rare, no badge
                    st.caption(f"⏱ `{fmt_latency(claude_result['latency_ms'])}` "
                               f"· `${claude_result['cost']:.4f} per call`")

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
