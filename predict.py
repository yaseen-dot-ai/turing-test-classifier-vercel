import numpy as np
import os
from typing import List

CANDIDATES = ["gpt4o", "claude", "roberta"]
CLASSES = ["HUMAN", "AMBIGUOUS", "AI"]

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- GPT-4o (OpenAI) ---
def predict_gpt4o(texts: List[str]) -> List[str]:
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _random_preds(len(texts))
    openai.api_key = api_key
    results = []
    for text in texts:
        try:
            resp = openai.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "user", "content": (
                        "You are a Turing Test classifier. Given a text, respond with only one of these labels:\n"
                        "- HUMAN: if you are confident it was written by a human,\n"
                        "- AI: if you are confident it was written by an AI,\n"
                        "- AMBIGUOUS: if you are unsure.\n\n"
                        "Reply with only the label: HUMAN, AI, or AMBIGUOUS. Do not add anything else."
                    )},
                    {"role": "user", "content": f"text: {text}"}
                ],
                max_tokens=3,
                temperature=0
            )
            out = resp.choices[0].message.content.strip().upper()
            if out not in CLASSES:
                out = "AMBIGUOUS"
            results.append(out)
        except Exception:
            results.append("AMBIGUOUS")
    return results

# --- Claude (Anthropic) ---
def predict_claude(texts: List[str]) -> List[str]:
    import httpx
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _random_preds(len(texts))
    results = []
    for text in texts:
        try:
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": "claude-3-7-sonnet-20250219",
                "max_tokens": 3,
                "temperature": 0,
                "messages": [
                    {"role": "user", "content": (
                        "You are a Turing Test classifier. Given a text, respond with only one of these labels:\n"
                        "- HUMAN: if you are confident it was written by a human,\n"
                        "- AI: if you are confident it was written by an AI,\n"
                        "- AMBIGUOUS: if you are unsure.\n\n"
                        "Reply with only the label: HUMAN, AI, or AMBIGUOUS. Do not add anything else."
                    )},
                    {"role": "user", "content": f"text: {text}"}
                ]
            }
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=10
            )
            out = resp.json()["content"][0]["text"].strip().upper()
            if out not in CLASSES:
                out = "AMBIGUOUS"
            results.append(out)
        except Exception:
            results.append("AMBIGUOUS")
    return results

# --- RoBERTa (HuggingFace) ---
_roberta_model = None
_roberta_tokenizer = None

def predict_roberta(texts: List[str]) -> List[str]:
    global _roberta_model, _roberta_tokenizer
    if _roberta_model is None or _roberta_tokenizer is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        _roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
    import torch
    label_map = {0: "HUMAN", 1: "AMBIGUOUS", 2: "AI"}
    preds = []
    for text in texts:
        inputs = _roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = _roberta_model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            preds.append(label_map.get(pred, "AMBIGUOUS"))
    return preds

# --- Random fallback ---
def _random_preds(n):
    np.random.seed(42)
    return np.random.choice(CLASSES, size=n).tolist()

def batch_predict(texts, model):
    if model == "gpt4o":
        return predict_gpt4o(list(texts))
    elif model == "claude":
        return predict_claude(list(texts))
    elif model == "roberta":
        return predict_roberta(list(texts))
    else:
        return _random_preds(len(texts)) 