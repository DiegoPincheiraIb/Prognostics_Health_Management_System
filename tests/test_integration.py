import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import mlflow

from rul.data import COLUMNS, FEATURES
from rul.train import run


def make_dataset(path):
    path.mkdir()
    train = [[unit, cycle, *[float(unit + cycle + i / 10) for i in range(len(FEATURES))]]
             for unit in range(1, 7) for cycle in range(1, 7)]
    test = [[unit, cycle, *[float(unit + cycle + i / 10) for i in range(len(FEATURES))]]
            for unit in range(1, 3) for cycle in range(1, 5)]
    pd.DataFrame(train, columns=COLUMNS).to_csv(path / "train_FD001.txt", sep=" ", header=False, index=False)
    pd.DataFrame(test, columns=COLUMNS).to_csv(path / "test_FD001.txt", sep=" ", header=False, index=False)
    (path / "RUL_FD001.txt").write_text("2\n4\n")
    return test


def test_all_models_tracking_registry_and_reload(tmp_path):
    data = tmp_path / "data"
    rows = make_dataset(data)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    args = SimpleNamespace(data_dir=str(data), tracking_uri=uri, experiment="test",
                           model_name="test-rul", alias="champion",
                           models=["baseline", "random_forest", "xgboost", "lstm"],
                           seed=42, window=3, epochs=1, threshold=3, wandb=False,
                           wandb_project="test")
    result = run(args)
    assert result["test_units"] == 2
    assert np.isfinite(result["test"]["rmse"])
    model = mlflow.pyfunc.load_model("models:/test-rul@champion")
    prediction = model.predict(pd.DataFrame({"sequence": [json.dumps([r[2:] for r in rows[:4]])]}))
    assert np.isfinite(prediction.iloc[0]["rul"])
    runs = mlflow.search_runs(experiment_names=["test"])
    assert len(runs) == 4
    assert set(runs["params.model"]) == {"baseline", "random_forest", "xgboost", "lstm"}
    assert runs["metrics.test_rmse"].notna().sum() == 1
    for run_id in runs["run_id"]:
        candidate = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        prediction = candidate.predict(pd.DataFrame({"sequence": [json.dumps([r[2:] for r in rows[:4]])]}))
        assert np.isfinite(prediction.iloc[0]["rul"])