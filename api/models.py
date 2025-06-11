from pydantic import BaseModel
from typing import List, Optional, Dict

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    predictions: List[Dict[str, str]]

class Sample(BaseModel):
    text: str
    label: str

class RunRequest(BaseModel):
    samples: List[Sample]
    return_preds: Optional[bool] = False

class RunResponse(BaseModel):
    run_id: str
    winner: str
    results: Dict[str, Dict[str, Dict[str, float]]]  # {"gpt-4.1": {"precision": {...}, "recall": {...}}}
    predictions: Optional[List[Dict[str, str]]] = None 