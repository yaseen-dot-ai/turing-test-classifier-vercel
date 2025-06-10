from fastapi import FastAPI
from models import RunRequest, RunResponse, PredictRequest, PredictResponse
from predict import batch_predict, CANDIDATES, CLASSES, DISPLAY_NAME_MAP
from metrics import confusion_matrix, per_class_pr
from utils import utc_stamp, s3_put, save_parquet_to_s3, write_local
import pandas as pd
import json


app = FastAPI()


@app.post("/run", response_model=RunResponse)
def run_experiment(req: RunRequest):
    df = pd.DataFrame([s.model_dump() for s in req.samples])
    models = req.override_models or CANDIDATES
    run_id = utc_stamp(ms=True)

    preds = {}
    for m in models:
        preds[m] = batch_predict(df["text"], model=m)

    metrics = {}
    for m, y_hat in preds.items():
        cm = confusion_matrix(df["label"], y_hat, labels=CLASSES)
        precision, recall = per_class_pr(cm, labels=CLASSES)
        metrics[DISPLAY_NAME_MAP[m]] = {"precision": precision, "recall": recall}

    def meets_bar(p):
        return all(v >= .95 for v in p.values())
    eligible = [m for m in models if meets_bar(metrics[DISPLAY_NAME_MAP[m]]["precision"])]

    if eligible:
        winner = max(eligible, key=lambda m: sum(metrics[DISPLAY_NAME_MAP[m]]["recall"].values()))
    else:
        winner = max(models, key=lambda m: metrics[DISPLAY_NAME_MAP[m]]["precision"]["HUMAN"])

    winner_cfg = {"model_type": DISPLAY_NAME_MAP[winner], "version": run_id}

    root = f"runs/{run_id}/"
    s3_put(root + "winner.json", json.dumps(winner_cfg))
    print(json.dumps(metrics))
    s3_put(root + "metrics.json", json.dumps(metrics))
    full_out = df.assign(**{f"{DISPLAY_NAME_MAP[m]}_pred": p for m, p in preds.items()})
    save_parquet_to_s3(root + "predictions.parquet", full_out)
    write_local("winner_config.json", winner_cfg)

    body = {
        "run_id": run_id,
        "winner": DISPLAY_NAME_MAP[winner],
        "precision": metrics[DISPLAY_NAME_MAP[winner]]["precision"],
        "recall": metrics[DISPLAY_NAME_MAP[winner]]["recall"]
    }
    if req.return_preds:
        body["predictions"] = full_out.to_dict("records")
    return body


@app.post("/predict", response_model=PredictResponse)
def predict_labels(req: PredictRequest):
    results = []
    for text in req.texts:
        pred = {"text": text}
        for model in CANDIDATES:
            pred[DISPLAY_NAME_MAP[model]] = batch_predict([text], model=model)[0]
        results.append(pred)
    return {"predictions": results}