# AI-Generated Text Detection — BERT vs Claude

> 🚧 **Work in progress.** This README is a placeholder. The full write-up — results
> tables, cross-domain analysis, error analysis, limitations, and the live demo + model
> links — lands on Day 7. Until then, see the notebooks for current progress.

End-to-end NLP project comparing five approaches to detecting AI-generated text —
TF-IDF + Logistic Regression, a PyTorch LSTM, a fine-tuned BERT, a LoRA-adapted BERT, and
Claude Haiku 4.5 zero-shot — with a cross-domain experiment that tests whether a detector
trained on Reddit text transfers to finance and medicine.

## Data

- **Source:** [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
  — License **CC-BY-SA-4.0**
- English subsets only: `reddit_eli5` (training + in-domain test), `finance` + `medicine`
  (held out for the cross-domain experiment)

## Status

| Day | Deliverable | State |
|-----|-------------|-------|
| 1   | EDA + train/val/test splits (Notebook 1) | done |
| 2–3 | TF-IDF + LogReg, PyTorch LSTM baselines (Notebook 2) | done |
| 4   | BERT full fine-tune + LoRA (Notebook 3) | in progress |
| 5   | Claude zero-shot + 5-model comparison (Notebook 4) | todo |
| 6   | Streamlit demo on HuggingFace Spaces | todo |
| 7   | Cross-domain + error analysis + final README | todo |

## Reproducing (so far)

**Python 3.11 is required** (the pinned `numpy<2.0` / TensorFlow stack has no wheels for 3.12+):

```bash
conda create -n aidetect python=3.11 -y && conda activate aidetect
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_and_data_prep.ipynb
```
