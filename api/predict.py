import os
import asyncio
from typing import List
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Model display names
MODELS = ["gpt", "claude", "roberta"]
DISPLAY_NAMES = {
    "gpt": "gpt-4.1",
    "claude": "claude-3-7-sonnet-20250219", 
    "roberta": "roberta-large-llm-response-detector"
}
CLASSES = ["HUMAN", "AI", "AMBIGUOUS"]

# GPT-4 setup
from langchain_openai import ChatOpenAI
gpt_llm = ChatOpenAI(
    model="gpt-4.1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=5,
    temperature=0,
)

# Claude setup  
from langchain_anthropic import ChatAnthropic
claude_llm = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=5,
    temperature=0,
)

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

async def predict_gpt(texts: List[str]) -> List[str]:
    results = []
    for text in texts:
        try:
            response = await gpt_llm.ainvoke(get_prompt(text))
            label = response.content.strip().upper()
            if label not in CLASSES:
                label = "AMBIGUOUS"
            results.append(label)
        except:
            results.append("AMBIGUOUS")
    return results

async def predict_claude(texts: List[str]) -> List[str]:
    results = []
    for text in texts:
        try:
            response = await claude_llm.ainvoke(get_prompt(text))
            label = response.content.strip().upper()
            if label not in CLASSES:
                label = "AMBIGUOUS"
            results.append(label)
        except:
            results.append("AMBIGUOUS")
    return results

def predict_roberta(texts: List[str]) -> List[str]:
    # Simplified roberta - returns random for demo
    # Replace with actual roberta model
    np.random.seed(42)
    return np.random.choice(CLASSES, size=len(texts)).tolist()

async def batch_predict(texts: List[str], model: str) -> List[str]:
    if model == "gpt":
        return await predict_gpt(texts)
    elif model == "claude":
        return await predict_claude(texts)
    elif model == "roberta":
        return predict_roberta(texts)
    else:
        return ["AMBIGUOUS"] * len(texts) 