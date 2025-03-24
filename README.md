# Next Token Prediction: Baseline & Transformer

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](#)
[![Hugging%20Face%20Transformers](https://img.shields.io/badge/🤗%20Transformers-4.x-green.svg)](#)

This **Jupyter Notebook** demonstrates two approaches for **Next Token Prediction** on the WikiText-2 dataset:
1. **Baseline Model**: A feed-forward neural network that uses token embeddings.
2. **Transformer-Based Model**: A simplified multi-head (4-head) attention mechanism, inspired by modern NLP architectures.

You’ll find sections for:
- **Tokenization & Preprocessing** (including vocabulary creation and batching)
- **Baseline Model** (embeddings + MLP)
- **Transformer Model** (multi-head self-attention + positional encoding)
- **Evaluation** (perplexity, accuracy, inference time, and more)

---

## Table of Contents
1. [Prerequisites](#prerequisites)  
2. [Notebook Overview](#notebook-overview)  
3. [Data Preparation](#data-preparation)  
4. [Running the Notebook](#running-the-notebook)  
5. [Key Sections](#key-sections)  
6. [Results](#results)  
7. [References & Additional Resources](#references--additional-resources)

---

## Prerequisites
1. **Python 3.9+** (or a compatible version).
2. A working **Jupyter** or **JupyterLab** setup to open and run the `.ipynb` file.
3. The following libraries (listed in `requirements.txt`):
   - `torch`
   - `transformers`
   - `tqdm`
   - `numpy`
   - (Optional) `spacy` or other tokenizer libraries
   - (Optional) `matplotlib` (if you plan to visualize training curves)

Install all dependencies via:
```bash
pip install -r requirements.txt
