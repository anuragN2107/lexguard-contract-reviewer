import gradio as gr
import torch
import os
from transformers import AutoTokenizer
from app_model import LegalBertCNNClassifier

# 1. Load the tokenizer and model structure
TOKENIZER_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = LegalBertCNNClassifier(TOKENIZER_NAME, num_classes=4)

# Load weights safely if present (handles initialization checks gracefully)
if os.path.exists("legal_bert_cnn.pt"):
    model.load_state_dict(torch.load("legal_bert_cnn.pt", map_location=torch.device('cpu')))
model.eval()

# Mapping internal targets back to risk descriptions
LABELS = ["Indemnity Risk", "Non-Compete Limit", "Governing Law Choice", "Liability Cap Warning"]

def analyze_contract_clause(clause_text):
    if not clause_text.strip():
        return "Please input a valid legal contract clause to analyze."
        
    # Process text using standard underlying calls safely
    inputs = tokenizer(
        clause_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.sigmoid(logits).flatten().tolist()
    
    # Format the probability outputs for the Gradio interface display
    results = {}
    for i, label in enumerate(LABELS):
        results[label] = float(probabilities[i])
    return results

# 2. Build an intuitive Gradio interface layout
demo = gr.Interface(
    fn=analyze_contract_clause,
    inputs=gr.Textbox(lines=5, label="Paste Contract Clause / Paragraph Here", placeholder="Ex: The vendor agrees to indemnify, defend and hold harmless..."),
    outputs=gr.Label(num_top_classes=4, label="Risk Factor Analysis Probabilities"),
    title="LexGuard AI: Legal Contract Multi-Label Risk Assessment Engine",
    description="An optimized hybrid Legal-BERT + TextCNN classification platform built to scan dense corporate contracts and flag compliance risks automatically.",
    examples=[
        ["The Vendor shall indemnify, defend, and hold harmless the Client from any third-party IP claims."],
        ["The Executive shall not engage in any competing software engineering business within North America for 24 months."],
        ["This Agreement will be explicitly governed by, and construed in accordance with, the state laws of Delaware."]
    ]
)

if __name__ == "__main__":
    demo.launch()