from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_run_endpoint():
    data = {
        "samples": [
            {"text": "This is a human thought.", "label": "HUMAN"},
            {"text": "This is ambiguous.", "label": "AMBIGUOUS"},
            {"text": "This is an AI response.", "label": "AI"}
        ],
        "override_models": ["gpt4o", "claude", "roberta"],
        "return_preds": True
    }
    response = client.post("/run", json=data)
    assert response.status_code == 200
    out = response.json()
    print(out)
    assert "run_id" in out
    assert "winner" in out
    assert "precision" in out
    assert "recall" in out
    assert "predictions" in out
    assert len(out["predictions"]) == 3 