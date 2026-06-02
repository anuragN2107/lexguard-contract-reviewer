# LexGuard AI: Enterprise Contract Review & Risk Assessment Engine

LexGuard AI is an optimized, high-throughput machine learning pipeline engineered to review long-form, dense corporate legal agreements. The system automatically classifies individual contract clauses into specific domains and flags overlapping legal risks simultaneously.

🚀 **Live Interactive Demo:** [Hugging Face Space Live](https://anuragn2107-lexguard-contract-reviewer.hf.space/)

---

## 🏷️ Project Core Metadata
* **Domain:** `legal-tech`
* **Core Technology:** `nlp`, `transformers`, `pytorch`, `text-cnn`
* **Task Type:** `multi-label-classification`
* **Model Backbone:** `bert` (`nlpaueb/legal-bert-base-uncased`)
* **Deployment Stack:** `fastapi`, `gradio`, `docker`

---

## 🧠 System Architecture & Methodology

Instead of relying on basic text embeddings or slow, computationally expensive large language models (LLMs), LexGuard utilizes a high-performance hybrid deep learning topology:

1. **Transformer Context Layer:** Extracted token matrices travel through a fine-tuned `Legal-BERT` backbone to capture domain-specific legal vocabularies.
2. **Structural Feature Extractor (TextCNN):** Utilizes multi-kernel 1D Convolutional Neural Networks (kernel sizes: 3, 4, 5) running in parallel to capture local multi-gram phrase combinations (e.g., *"shall defend, indemnify"*, *"shall not compete"*).
3. **Multi-Label Head:** Employs an independent Sigmoid layer optimized by `BCEWithLogitsLoss` to calculate multiple independent risk probabilities for a single paragraph.

---

## 🛠️ Tools & Ecosystem

* **Deep Learning Framework:** PyTorch
* **Model Hub & Pipeline Utilities:** Hugging Face Transformers & Datasets
* **Web UI Component:** Gradio (for rapid interface testing)
* **Production API Backend:** FastAPI (supporting asynchronous multi-thread handling)
* **Development Environment:** Google Colab (T4 GPU Compute Infrastructure)

---

## 📂 Repository File Structure

Based on our verified operational environment, your repository contains the following core files:

* **`.gitignore`** — Instructs Git to ignore local python cache files, virtual environments, and heavy neural network weights parameters (`legal_bert_cnn.pt`).
* **`README.md`** — This documentation sheet containing architecture overviews and installation blueprints.
* **`app.py`** — Main interactive web application execution framework built using Gradio.
* **`app_model.py`** — Pure PyTorch implementation defining the structural class layout of the `LegalBertCNNClassifier`.
* **`lexguard_contract_reviewer.ipynb`** — The complete Google Colab notebook tracking model dataset synthesis, training configurations, and F1 validation loss loops.
* **`requirements.txt`** — Package manifest file listing exact dependencies for rapid server instantiation.

---

## ⚙️ Local Installation & Execution

### Prerequisites
* Python 3.10 or higher
* PyTorch (CPU or CUDA equivalent)
