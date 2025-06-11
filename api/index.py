from fastapi import FastAPI
from .models import PredictRequest, PredictResponse, RunRequest, RunResponse
from .predict import batch_predict, MODELS, DISPLAY_NAMES, CLASSES
import time
import asyncio
from typing import Dict

app = FastAPI()

def utc_stamp() -> str:
    return str(int(time.time() * 1000))

def confusion_matrix(y_true, y_pred, labels):
    """Simple confusion matrix"""
    import numpy as np
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1
    return cm

def per_class_metrics(cm, labels):
    """Calculate precision and recall"""
    precision = {}
    recall = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        precision[label] = tp / max(cm[:, i].sum(), 1)
        recall[label] = tp / max(cm[i, :].sum(), 1)
    return precision, recall

@app.post("/predict", response_model=PredictResponse)
async def predict_labels(req: PredictRequest):
    results = []
    
    # Get predictions from all models concurrently
    prediction_tasks = [batch_predict(req.texts, model) for model in MODELS]
    all_predictions_list = await asyncio.gather(*prediction_tasks)
    
    # Map results back to model names
    all_predictions = dict(zip(MODELS, all_predictions_list))
    
    # Format results
    for i, text in enumerate(req.texts):
        result = {"text": text}
        for model in MODELS:
            result[DISPLAY_NAMES[model]] = all_predictions[model][i]
        results.append(result)
    
    return PredictResponse(predictions=results)

@app.post("/run", response_model=RunResponse) 
async def run_experiment(req: RunRequest):
    run_id = utc_stamp()
    
    # Extract texts and labels
    texts = [s.text for s in req.samples]
    labels = [s.label for s in req.samples]
    
    # Get predictions from all models concurrently
    prediction_tasks = [batch_predict(texts, model) for model in MODELS]
    all_predictions_list = await asyncio.gather(*prediction_tasks)
    
    # Map results back to model names
    all_predictions = dict(zip(MODELS, all_predictions_list))
    
    # Calculate metrics for each model
    metrics = {}
    for model in MODELS:
        cm = confusion_matrix(labels, all_predictions[model], CLASSES)
        precision, recall = per_class_metrics(cm, CLASSES)
        metrics[DISPLAY_NAMES[model]] = {"precision": precision, "recall": recall}
    
    # Select winner (highest average precision)
    winner_model = max(MODELS, key=lambda m: sum(metrics[DISPLAY_NAMES[m]]["precision"].values()))
    winner_display = DISPLAY_NAMES[winner_model]
    
    # Prepare response
    response_data = {
        "run_id": run_id,
        "winner": winner_display,
        "results": metrics
    }
    
    # Add predictions if requested
    if req.return_preds:
        predictions = []
        for i, sample in enumerate(req.samples):
            pred = {
                "text": sample.text,
                "label": sample.label
            }
            for model in MODELS:
                pred[f"{DISPLAY_NAMES[model]}_pred"] = all_predictions[model][i]
            predictions.append(pred)
        response_data["predictions"] = predictions
    
    return RunResponse(**response_data) 