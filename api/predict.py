import os
import asyncio
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import traceback
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json

load_dotenv()

# Model display names
MODELS = ["gpt", "claude", "roberta"]
fine_tuned_config = json.load(open("ft_config.json"))
DISPLAY_NAMES = {
    "gpt": fine_tuned_config["gpt"] or "gpt-4.1",
    "claude": "claude-3-7-sonnet-20250219", 
    "roberta": fine_tuned_config["roberta"] or "SuperAnnotate/roberta-large-llm-content-detector"
}
CLASSES = ["HUMAN", "AI", "AMBIGUOUS"]

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPT-4 setup
from langchain_openai import ChatOpenAI
gpt_llm = ChatOpenAI(
    model=DISPLAY_NAMES["gpt"],
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=5,
    temperature=0,
)

# Claude setup  
from langchain_anthropic import ChatAnthropic
claude_llm = ChatAnthropic(
    model=DISPLAY_NAMES["claude"],
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=5,
    temperature=0,
)

# RoBERTa setup
from generated_text_detector.utils.model.roberta_classifier import RobertaClassifier
from transformers import AutoTokenizer

_roberta_model = RobertaClassifier.from_pretrained(DISPLAY_NAMES["roberta"]).to(device)
_roberta_tokenizer = AutoTokenizer.from_pretrained(DISPLAY_NAMES["roberta"])

BATCH_SIZE = 16
interval = 5

def get_prompt(text: str) -> List[dict]:
    return [
        {"role": "user", "content": (
            "You are a turing test classifier. "
            "Given a text, respond with only one of these labels:\n"
            "- HUMAN: if you are confident it was written by a human,\n"
            "- AI: if you are confident it was written by an AI,\n"
            "- AMBIGUOUS: if you are unsure.\n\n"
            "Reply with only the label: HUMAN, AI, or AMBIGUOUS. Do not add anything else.\n"
            f"text: {text}"
        )}
    ]

async def predict_gpt_single(text: str) -> str:
    try:
        response = await gpt_llm.ainvoke(get_prompt(text))
        label = response.content.strip().upper()
        if label not in CLASSES:
            label = "AMBIGUOUS"
        return label
    except:
        traceback.print_exc()
        return "AMBIGUOUS"

async def predict_gpt(texts: List[str]) -> List[str]:
    preds = []
    for batch in tqdm(range(0, len(texts), BATCH_SIZE), desc="GPT"):
        batch_texts = texts[batch:batch+BATCH_SIZE]
        batch_preds = await asyncio.gather(*(predict_gpt_single(text) for text in batch_texts))
        preds.extend(batch_preds)
        time.sleep(interval)
    return preds

async def predict_claude_single(text: str) -> str:
    try:
        response = await claude_llm.ainvoke(get_prompt(text))
        label = response.content.strip().upper()
        if label not in CLASSES:
            label = "AMBIGUOUS"
        return label
    except:
        traceback.print_exc()
        return "AMBIGUOUS"

async def predict_claude(texts: List[str]) -> List[str]:
    preds = []
    for batch in tqdm(range(0, len(texts), BATCH_SIZE), desc="Claude"):
        batch_texts = texts[batch:batch+BATCH_SIZE]
        batch_preds = await asyncio.gather(*(predict_claude_single(text) for text in batch_texts))
        preds.extend(batch_preds)
        time.sleep(interval)
    return preds

def predict_roberta(texts: List[str]) -> List[str]:
    preds = []
    for batch in tqdm(range(0, len(texts), BATCH_SIZE), desc="RoBERTa"):
        batch_texts = texts[batch:batch+BATCH_SIZE]
        tokens = _roberta_tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=512,
            padding='longest',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            _, logits = _roberta_model(**tokens)
            probas = F.sigmoid(logits).squeeze(1).cpu().tolist()
            for prob in probas:
                if prob > 0.7:
                    preds.append("AI")
                elif prob < 0.3:
                    preds.append("HUMAN")
                else:
                    preds.append("AMBIGUOUS")
        time.sleep(interval)
    return preds

async def batch_predict(texts: List[str], model: str) -> List[str]:
    if model == "gpt":
        return await predict_gpt(texts)
    elif model == "claude":
        return await predict_claude(texts)
    elif model == "roberta":
        return predict_roberta(texts)
    else:
        return ["AMBIGUOUS"] * len(texts) 