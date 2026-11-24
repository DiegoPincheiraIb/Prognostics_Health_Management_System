"""C-MAPSS loading and causal, per-engine windows."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [f"setting_{i}" for i in range(1, 4)] + [
    f"sensor_{i}" for i in range(1, 22)
]
COLUMNS = ["unit", "cycle", *FEATURES]


def load_cycles(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.shape[1] != len(COLUMNS) or data.empty:
        raise ValueError(f"{path}: expected nonempty C-MAPSS data with 26 columns")
    data.columns = COLUMNS
    data = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"{path}: non-finite value")
    if (data[["unit", "cycle"]] <= 0).any().any() or (
        data[["unit", "cycle"]] % 1 != 0
    ).any().any():
        raise ValueError(f"{path}: unit and cycle must be positive integers")
    data[["unit", "cycle"]] = data[["unit", "cycle"]].astype(int)
    data = data.sort_values(["unit", "cycle"]).reset_index(drop=True)
    if data.duplicated(["unit", "cycle"]).any():
        raise ValueError(f"{path}: duplicate unit/cycle")
    if any(not (group["cycle"].diff().dropna() == 1).all()
           for _, group in data.groupby("unit")):
        raise ValueError(f"{path}: cycles must be contiguous per unit")
    return data


def load_test_labels(path: Path, test: pd.DataFrame) -> dict[int, float]:
    labels = pd.read_csv(path, sep=r"\s+", header=None)
    units = sorted(test.unit.unique())
    if labels.shape != (len(units), 1):
        raise ValueError(f"{path}: expected one RUL value per test unit")
    if units != list(range(1, len(units) + 1)):
        raise ValueError(f"{path}: test units must be consecutively numbered from 1")
    values = pd.to_numeric(labels.iloc[:, 0], errors="raise").to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path}: RUL labels must be finite and nonnegative")
    return dict(zip(units, values.astype(float), strict=True))


def add_rul(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["rul"] = result.groupby("unit")["cycle"].transform("max") - result["cycle"]
    return result


def split_units(data: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = sorted(data.unit.unique())
    if len(units) < 2:
        raise ValueError("at least two training units are required")
    train, valid = train_test_split(units, test_size=0.2, random_state=seed)
    return data[data.unit.isin(train)].copy(), data[data.unit.isin(valid)].copy()


def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(train[FEATURES])


def censor_validation(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Simulate incomplete lifetimes while retaining the original RUL labels."""
    rng = np.random.default_rng(seed)
    prefixes = []
    for _, group in data.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        end = int(rng.integers(1, len(group) + 1))
        prefixes.append(group.iloc[:end])
    return pd.concat(prefixes, ignore_index=True)


def window_for(values: np.ndarray, length: int) -> np.ndarray:
    if length < 1 or len(values) < 1 or values.ndim != 2 or values.shape[1] != len(FEATURES):
        raise ValueError("window requires at least one row of 24 features and positive length")
    values = values[-length:]
    if len(values) < length:
        values = np.concatenate([np.repeat(values[:1], length - len(values), axis=0), values])
    return values


def windows(data: pd.DataFrame, scaler: StandardScaler, length: int,
            endpoints_only: bool = False, labels: dict[int, float] | None = None
            ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    samples, targets, units = [], [], []
    for unit, group in data.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        scaled = scaler.transform(group[FEATURES])
        indices = [len(group) - 1] if endpoints_only else range(len(group))
        for i in indices:
            samples.append(window_for(scaled[:i + 1], length))
            targets.append(labels[int(unit)] if labels is not None else group.iloc[i]["rul"])
            units.append(int(unit))
    if not samples:
        raise ValueError("no windows available")
    return np.asarray(samples, dtype=np.float32), np.asarray(targets, dtype=np.float32), units