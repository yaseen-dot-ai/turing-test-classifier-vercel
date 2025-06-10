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
    precision: Dict[str, float]
    recall: Dict[str, float]
    predictions: Optional[List[Dict[str, str]]] = None 