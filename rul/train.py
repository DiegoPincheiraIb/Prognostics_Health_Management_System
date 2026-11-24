"""Train all candidates, select on validation, then evaluate the official test."""

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from rul.data import (FEATURES, add_rul, censor_validation, fit_scaler, load_cycles,
                      load_test_labels, split_units, windows)
from rul.metrics import evaluate
from rul.model import LSTMRegressor, RULModel, predict_windows


def train_model(kind, x, y, seed, epochs):
    if kind == "lstm":
        model = LSTMRegressor(len(FEATURES))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        model.train()
        for _ in range(epochs):
            for features, target in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(features), target)
                loss.backward()
                optimizer.step()
        return model
    estimators = {
        "baseline": lambda: DummyRegressor(strategy="mean"),
        "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "xgboost": lambda: XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05,
                                        objective="reg:squarederror", random_state=seed, n_jobs=2),
    }
    model = estimators[kind]()
    return model.fit(x.reshape(len(x), -1), y)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    if args.window < 1 or args.epochs < 1 or not math.isfinite(args.threshold) or args.threshold < 0:
        raise ValueError("window and epochs must be positive, and threshold finite and nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train_FD001.txt"
    train, valid = split_units(add_rul(load_cycles(train_path)), args.seed)
    valid = censor_validation(valid, args.seed)
    scaler = fit_scaler(train)
    x_train, y_train, _ = windows(train, scaler, args.window)
    x_valid, y_valid, valid_units = windows(valid, scaler, args.window, endpoints_only=True)
    partitions = {"train": sorted(map(int, train.unit.unique())),
                  "validation": valid_units}
    wb = None
    if args.wandb:
        import wandb
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is required when W&B is enabled")
        wb = wandb
    best = None
    for kind in args.models:
        tracking = wb.init(project=args.wandb_project, name=kind, reinit=True) if wb else None
        try:
            with mlflow.start_run(run_name=kind) as active:
                params = {"model": kind, "window": args.window, "seed": args.seed,
                          "epochs": args.epochs, "threshold": args.threshold,
                          "dataset": "C-MAPSS FD001", "train_sha256": file_hash(train_path)}
                mlflow.log_params(params)
                mlflow.log_dict(partitions, "partitions.json")
                if tracking:
                    tracking.config.update(params)
                model = train_model(kind, x_train, y_train, args.seed, args.epochs)
                hyperparams = ({"hidden_size": 32, "learning_rate": 0.001, "batch_size": 64}
                               if kind == "lstm" else
                               {k: v for k, v in model.get_params().items()
                                if k in {"n_estimators", "max_depth", "learning_rate", "strategy"}
                                and v is not None})
                mlflow.log_params(hyperparams)
                if tracking:
                    tracking.config.update(hyperparams)
                bundle = {"kind": kind, "model": model, "scaler": scaler, "window": args.window}
                predictions = np.maximum(predict_windows(bundle, x_valid), 0)
                metrics = evaluate(y_valid, predictions, args.threshold)
                mlflow.log_metrics({f"validation_{k}": v for k, v in metrics.items()})
                mlflow.log_dict({"unit": valid_units, "actual": y_valid.tolist(),
                                 "predicted": predictions.tolist()}, "validation_predictions.json")
                fig, ax = plt.subplots()
                ax.scatter(y_valid, predictions)
                ax.set(xlabel="Actual RUL (cycles)", ylabel="Predicted RUL (cycles)")
                mlflow.log_figure(fig, "validation_predictions.png")
                plt.close(fig)
                if tracking:
                    tracking.log({f"validation_{k}": v for k, v in metrics.items()})
                with tempfile.TemporaryDirectory() as tmp:
                    bundle_path = Path(tmp) / "bundle.joblib"
                    joblib.dump(bundle, bundle_path)
                    mlflow.pyfunc.log_model(
                        artifact_path="model", python_model=RULModel(),
                        artifacts={"bundle": str(bundle_path)},
                        pip_requirements=["mlflow>=2.14,<3", "numpy>=1.26,<3",
                                          "pandas>=2.2,<3", "scikit-learn>=1.4,<2",
                                          "xgboost>=2,<4", "torch>=2.2,<3", "joblib"],
                    )
                if best is None or metrics["rmse"] < best["rmse"]:
                    best = {"rmse": metrics["rmse"], "run_id": active.info.run_id,
                            "bundle": bundle, "kind": kind}
        finally:
            if tracking:
                tracking.finish()

    # Test labels are opened only after the validation-based choice.
    test_path = data_dir / "test_FD001.txt"
    label_path = data_dir / "RUL_FD001.txt"
    test = load_cycles(test_path)
    labels = load_test_labels(label_path, test)
    x_test, y_test, test_units = windows(test, scaler, args.window, endpoints_only=True, labels=labels)
    test_metrics = evaluate(y_test, predict_windows(best["bundle"], x_test), args.threshold)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_params({"test_sha256": file_hash(test_path),
                           "labels_sha256": file_hash(label_path)})
        mlflow.log_dict({"test": test_units}, "test_units.json")
    registered = mlflow.register_model(f"runs:/{best['run_id']}/model", args.model_name)
    MlflowClient().set_registered_model_alias(args.model_name, args.alias, registered.version)
    result = {"selected": best["kind"], "run_id": best["run_id"],
              "version": registered.version, "validation_rmse": best["rmse"],
              "test": test_metrics, "test_units": len(test_units)}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/CMAPSSData")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--experiment", default="industrial-rul-fd001")
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "industrial-rul"))
    parser.add_argument("--alias", default=os.environ.get("MODEL_ALIAS", "champion"))
    parser.add_argument("--models", nargs="+", choices=["baseline", "random_forest", "xgboost", "lstm"],
                        default=["baseline", "random_forest", "xgboost", "lstm"])
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float,
                        default=float(os.environ.get("RUL_ALERT_THRESHOLD", "30")))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="industrial-rul")
    run(parser.parse_args())


if __name__ == "__main__":
    main()