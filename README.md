# LexGuard AI: Enterprise Contract Review & Risk Assessment Engine

LexGuard AI is an optimized, high-throughput Natural Language Processing (NLP) pipeline engineered to review long-form, dense corporate legal agreements. The system automatically classifies legal clauses into specific operational domains and identifies overlapping compliance risks concurrently.

## 🔗 Quick Links & Live Deployments

| Resource | Access Link | Description |
| :--- | :--- | :--- |
| **Interactive UI Demo** | [Hugging Face Space Live](https://anuragn2107-lexguard-contract-reviewer.hf.space/) | Test raw text chunks against our production engine live in-browser. |
| **API Endpoint Documentation** | [HF Space API Settings](https://anuragn2107-lexguard-contract-reviewer.hf.space/?view=api) | Reference for integrating programmatic cURL / Python API calls. |
| **Training Notebook** | [Google Colab Blueprint](https://colab.research.google.com/drive/1HynuifozAN-UEDTkQXF1HFBI2hggiYid?usp=sharing) | Trace structural weights tuning history and TextCNN validation logs. |

---

## 🧠 System Architecture

Instead of relying on basic text embeddings or costly, slow generative LLMs, LexGuard utilizes a high-performance hybrid deep learning topology:

1. **Transformer Backbone:** Leverages `nlpaueb/legal-bert-base-uncased` to extract specialized domain-specific sub-word token contexts.
2. **Feature Aggregator (TextCNN):** Utilizes multi-kernel 1D Convolutional Neural Networks (kernel sizes: 3, 4, 5) running in parallel to capture strict local phrase boundaries (e.g., *"agree to defend, indemnify"* or *"shall not compete"*).
3. **Multi-Label Head:** Employs an independent Sigmoid classification layer driven by `BCEWithLogitsLoss` to evaluate concurrent risks on a single paragraph sample.

### Key Monitored Target Risks:
* **Indemnity Risk:** Unfavorable third-party liability exposure flags.
* **Non-Compete Boundaries:** Strict geographical or timeline employment limits.
* **Governing Law Discrepancy:** Jurisdiction and venue tracking anomalies.
* **Liability Caps:** Invalid or unfavorable financial exposure limits.

---

## 🛠️ Repository Structure

```text
lexguard-contract-reviewer/
├── app.py              # Gradio User Interface and inference processing script
├── app_model.py        # Custom PyTorch Legal-BERT + TextCNN network architecture blueprint
├── requirements.txt    # Application dependencies
├── Notebook.ipynb      # Google Colab model training and validation pipeline history
├── Dockerfile          # Production container configuration
└── worker.py           # Asynchronous Celery execution layer for background processing
