import numpy as np
import os
from typing import List
import asyncio

CANDIDATES = ["gpt4o", "claude", "roberta"]
CLASSES = ["HUMAN", "AMBIGUOUS", "AI"]

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- GPT-4o (OpenAI, LangChain) ---
from langchain_openai import ChatOpenAI

gpt4o_llm = ChatOpenAI(
    model="gpt-4.1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=5,
    temperature=0,
)

def gpt4o_messages(text):
    return [
        {"role": "user", "content": (
            "You are a turing test classifier. "
            "Given a text, respond with only one of these labels:\n"
            "- HUMAN: if you are confident it was written by a human,\n"
            "- AI: if you are confident it was written by an AI,\n"
            "- AMBIGUOUS: if you are unsure.\n\n"
            "Reply with only the label: HUMAN, AI, or AMBIGUOUS. Do not add anything else.\n"
        )},
        {"role": "user", "content": f"text: {text}"}
    ]

async def predict_gpt4o_single(text):
    try:
        out = await gpt4o_llm.ainvoke(gpt4o_messages(text))
        label = out.content.strip().upper()
        if label not in CLASSES:
            label = "AMBIGUOUS"
        return label
    except Exception:
        return "AMBIGUOUS"

async def predict_gpt4o(texts: List[str]) -> List[str]:
    return await asyncio.gather(*(predict_gpt4o_single(t) for t in texts))

# --- Claude (Anthropic, LangChain) ---
from langchain_anthropic import ChatAnthropic

claude_llm = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=5,
    temperature=0,
)

def claude_messages(text):
    return [
        {"role": "user", "content": (
            "You are a turing test classifier. Given a text, respond with only one of these labels:\n"
            "- HUMAN: if you are confident it was written by a human,\n"
            "- AI: if you are confident it was written by an AI,\n"
            "- AMBIGUOUS: if you are unsure.\n\n"
            "Reply with only the label: HUMAN, AI, or AMBIGUOUS. Do not add anything else.\n"
        )},
        {"role": "user", "content": f"text: {text}"}
    ]

async def predict_claude_single(text):
    try:
        out = await claude_llm.ainvoke(claude_messages(text))
        label = out.content.strip().upper()
        if label not in CLASSES:
            label = "AMBIGUOUS"
        return label
    except Exception:
        return "AMBIGUOUS"

async def predict_claude(texts: List[str]) -> List[str]:
    return await asyncio.gather(*(predict_claude_single(t) for t in texts))

# --- RoBERTa (HuggingFace, batched) ---
_roberta_model = None
_roberta_tokenizer = None

BATCH_SIZE = 16

def predict_roberta(texts: List[str]) -> List[str]:
    global _roberta_model, _roberta_tokenizer
    if _roberta_model is None or _roberta_tokenizer is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        _roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
    import torch
    label_map = {0: "HUMAN", 1: "AMBIGUOUS", 2: "AI"}
    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = _roberta_tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = _roberta_model(**inputs).logits
            batch_preds = torch.argmax(logits, dim=1).tolist()
            preds.extend([label_map.get(p, "AMBIGUOUS") for p in batch_preds])
    return preds

# --- Random fallback ---
def _random_preds(n):
    np.random.seed(42)
    return np.random.choice(CLASSES, size=n).tolist()

# --- Unified batch_predict ---
def batch_predict(texts, model):
    if model == "gpt4o":
        return asyncio.run(predict_gpt4o(list(texts)))
    elif model == "claude":
        return asyncio.run(predict_claude(list(texts)))
    elif model == "roberta":
        return predict_roberta(list(texts))
    else:
        return _random_preds(len(texts)) 