---
title: AI Text Detector
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.58.0"
app_file: app.py
pinned: false
license: cc-by-sa-4.0
---

# AI-Generated Text Detector — BERT vs Claude

Paste any text and compare a **fine-tuned BERT** (110M params, trained on HC3 reddit_eli5,
macro-F1 0.9935) against **Claude Haiku 4.5 zero-shot**, side by side.

> First load after idling takes ~30 s (free CPU Space cold start).

Full project — 5-model comparison, cross-domain generalization study, error analysis:
[GitHub repo](https://github.com/lilhuang15/ai-generated-text-detector) ·
[model weights](https://huggingface.co/lilhuang15/bert-ai-text-detector-reddit)
