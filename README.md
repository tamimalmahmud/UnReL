# UnReL: Unlearning via Relearning

This repository contains the reproducibility code for **UnReL**, an experimental framework exploring efficient data unlearning in large language models (LLMs).  
The project focuses on estimating how likely each data instance is to be requested for removal and preparing models for targeted retraining with minimal cost.

---

## 🚀 Overview

UnReL works in two broad steps:
1. **Risk Scoring:** Each file or dataset item is analyzed to estimate its likelihood of future unlearning requests.  
2. **Shard Assignment:** Data are ordered and distributed into shards for training and evaluation under different strategies.

The system uses a Sentence-Transformer model to assess semantic similarity to sensitive categories (e.g., personal, medical, financial) and combines that with temporal indicators.

---

## ⚙️ Running the AutoScoring Tool

Run the scoring module on your dataset directory:

```bash
python autoscoring.py
