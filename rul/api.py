import json
import logging
import math
import os
from datetime import datetime
from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from rul.data import FEATURES

logger = logging.getLogger(__name__)
app = FastAPI(title="Industrial RUL API", version="0.1.0")


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    unit: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_version: Mapped[str] = mapped_column(String(64))
    rul: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    readings: Mapped[list] = mapped_column(JSON)


@lru_cache
def engine():
    url = os.environ.get("DATABASE_URL", "postgresql+psycopg://rul:rul@localhost:5432/rul")
    return create_engine(url, pool_pre_ping=True)


def session():
    try:
        Base.metadata.create_all(engine())
        with Session(engine()) as db:
            yield db
    except SQLAlchemyError as exc:
        logger.exception("Database operation failed")
        raise HTTPException(status_code=503, detail="Database unavailable") from exc


class Reading(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    cycle: int = Field(gt=0)
    settings: list[float] = Field(min_length=3, max_length=3)
    sensors: list[float] = Field(min_length=21, max_length=21)


class PredictRequest(BaseModel):
    unit: str = Field(min_length=1, max_length=128)
    readings: list[Reading] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def validate_cycles(self):
        cycles = [r.cycle for r in self.readings]
        if any(b != a + 1 for a, b in zip(cycles, cycles[1:])):
            raise ValueError("readings must have consecutive, increasing cycles")
        return self


def config():
    return (os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            os.environ.get("MODEL_NAME", "industrial-rul"),
            os.environ.get("MODEL_ALIAS", "champion"))


@lru_cache(maxsize=4)
def load_version(uri: str, name: str, version: str):
    mlflow.set_tracking_uri(uri)
    return mlflow.pyfunc.load_model(f"models:/{name}/{version}")


def current_model():
    uri, name, alias = config()
    try:
        version = MlflowClient(tracking_uri=uri).get_model_version_by_alias(name, alias).version
        return load_version(uri, name, version), str(version)
    except (MlflowException, OSError) as exc:
        logger.exception("Model unavailable")
        raise HTTPException(status_code=503, detail="Model unavailable") from exc


def configured_threshold():
    threshold = float(os.environ.get("RUL_ALERT_THRESHOLD", "30"))
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("RUL_ALERT_THRESHOLD must be finite and nonnegative")
    return threshold


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(db: Session = Depends(session)):
    db.execute(text("SELECT 1"))
    _, version = current_model()
    return {"status": "ready", "model_version": version}


@app.get("/model")
def model_info():
    _, version = current_model()
    _, name, alias = config()
    return {"name": name, "alias": alias, "version": version,
            "threshold": configured_threshold(), "features": FEATURES}


@app.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(session)):
    model, version = current_model()
    rows = [r.settings + r.sensors for r in request.readings]
    predicted = model.predict(pd.DataFrame({"sequence": [json.dumps(rows)]}))
    rul = float(predicted.iloc[0]["rul"])
    if not math.isfinite(rul) or rul < 0:
        logger.error("Model returned invalid RUL: %s", rul)
        raise HTTPException(status_code=503, detail="Model returned invalid prediction")
    threshold = configured_threshold()
    record = Prediction(unit=request.unit, model_version=version, rul=rul,
                        threshold=threshold, readings=[r.model_dump() for r in request.readings])
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "unit": record.unit, "rul": rul,
            "alert": rul <= threshold, "threshold": threshold,
            "model_version": version, "created_at": record.created_at}


@app.get("/units")
def units(db: Session = Depends(session)):
    return list(db.scalars(select(Prediction.unit).distinct().order_by(Prediction.unit)))


@app.get("/history/{unit}")
def history(unit: str, limit: int = 100, db: Session = Depends(session)):
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
    records = db.scalars(select(Prediction).where(Prediction.unit == unit)
                         .order_by(Prediction.id.desc()).limit(limit))
    return [{"id": r.id, "unit": r.unit, "created_at": r.created_at,
             "rul": r.rul, "threshold": r.threshold, "alert": r.rul <= r.threshold,
             "model_version": r.model_version, "readings": r.readings} for r in records]