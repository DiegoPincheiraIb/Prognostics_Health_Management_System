import json

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from torch import nn

from rul.data import FEATURES, window_for


class LSTMRegressor(nn.Module):
    def __init__(self, features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.output(output[:, -1, :]).squeeze(-1)


def predict_windows(bundle, x):
    if bundle["kind"] == "lstm":
        bundle["model"].eval()
        with torch.no_grad():
            return bundle["model"](torch.from_numpy(x.astype(np.float32))).numpy()
    return bundle["model"].predict(x.reshape(len(x), -1))


class RULModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        samples = []
        for value in model_input["sequence"]:
            rows = json.loads(value)
            frame = pd.DataFrame(rows, columns=FEATURES)
            if frame.empty or frame.isna().any().any():
                raise ValueError("sequence must contain all 24 finite features")
            values = frame.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("sequence contains non-finite features")
            scaled = self.bundle["scaler"].transform(frame)
            samples.append(window_for(scaled, self.bundle["window"]))
        prediction = predict_windows(self.bundle, np.asarray(samples, dtype=np.float32))
        return pd.DataFrame({"rul": np.maximum(prediction, 0).astype(float)})