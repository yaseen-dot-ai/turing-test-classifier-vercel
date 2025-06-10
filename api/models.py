from pydantic import BaseModel
from typing import List, Optional, Dict, Literal, Any

class Sample(BaseModel):
    text: str
    label: Literal["HUMAN", "AMBIGUOUS", "AI"]

class RunRequest(BaseModel):
    samples: List[Sample]
    override_models: Optional[List[str]] = None
    return_preds: Optional[bool] = False

class RunResponse(BaseModel):
    run_id: str
    winner: str
    precision: Dict[str, float]
    recall: Dict[str, float]
    predictions: Optional[Any] = None

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    predictions: List[Dict[str, str]] 