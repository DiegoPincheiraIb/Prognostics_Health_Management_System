import json

import pandas as pd
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from rul import api
from rul.data import FEATURES


def test_api_validation_and_persisted_history(monkeypatch):
    database = create_engine("sqlite://", connect_args={"check_same_thread": False},
                             poolclass=StaticPool)
    monkeypatch.setattr(api, "engine", lambda: database)

    class FakeModel:
        def predict(self, frame):
            assert len(json.loads(frame.iloc[0]["sequence"])[0]) == len(FEATURES)
            return pd.DataFrame({"rul": [30.0]})

    monkeypatch.setattr(api, "current_model", lambda: (FakeModel(), "7"))
    monkeypatch.setenv("RUL_ALERT_THRESHOLD", "30")
    client = TestClient(api.app)
    reading = {"cycle": 1, "settings": [0.0] * 3, "sensors": [0.0] * 21}
    response = client.post("/predict", json={"unit": "1", "readings": [reading]})
    assert response.status_code == 200
    assert response.json()["alert"] is True
    assert response.json()["model_version"] == "7"
    assert client.get("/units").json() == ["1"]
    assert client.get("/history/1").json()[0]["readings"] == [reading]
    assert client.get("/ready").json()["status"] == "ready"
    assert client.post("/predict", json={"unit": "1", "readings": [
        reading, {**reading, "cycle": 3}]}).status_code == 422
    assert client.post("/predict", json={"unit": "1", "readings": [
        {**reading, "sensors": [0]}]}).status_code == 422


def test_missing_model_is_reported(monkeypatch):
    class MissingModelClient:
        def __init__(self, tracking_uri):
            pass

        def get_model_version_by_alias(self, name, alias):
            raise MlflowException("missing")

    monkeypatch.setattr(api, "MlflowClient", MissingModelClient)
    response = TestClient(api.app).get("/model")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model unavailable"