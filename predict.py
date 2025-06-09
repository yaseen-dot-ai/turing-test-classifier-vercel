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

# --- RoBERTa (SuperAnnotate/roberta-large-llm-content-detector, RobertaClassifier, batched) ---
from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

_roberta_model = RobertaClassifier.from_pretrained("SuperAnnotate/roberta-large-llm-content-detector")
_roberta_tokenizer = AutoTokenizer.from_pretrained("SuperAnnotate/roberta-large-llm-content-detector")

BATCH_SIZE = 16

def predict_roberta(texts: List[str]) -> List[str]:
    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        tokens = _roberta_tokenizer(
            batch,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            _, logits = _roberta_model(**tokens)
            probas = F.sigmoid(logits).squeeze(1).cpu().numpy()
            for prob in probas:
                if prob > 0.7:
                    preds.append("AI")
                elif prob < 0.3:
                    preds.append("HUMAN")
                else:
                    preds.append("AMBIGUOUS")
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