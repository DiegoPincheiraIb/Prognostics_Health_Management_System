import numpy as np
import pandas as pd
import pytest

from rul.data import (COLUMNS, FEATURES, add_rul, censor_validation, fit_scaler,
                      load_cycles, load_test_labels, split_units, window_for, windows)
from rul.metrics import evaluate


def frame():
    rows = [[unit, cycle, *([float(unit * 10 + cycle)] * len(FEATURES))]
            for unit in range(1, 7) for cycle in range(1, 6)]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_rul_split_and_scaler():
    data = add_rul(frame())
    assert data[data.unit == 1].rul.tolist() == [4, 3, 2, 1, 0]
    train, valid = split_units(data, seed=4)
    assert set(train.unit).isdisjoint(set(valid.unit))
    scaler = fit_scaler(train)
    assert scaler.mean_[0] == pytest.approx(train[FEATURES[0]].mean())
    assert scaler.mean_[0] != pytest.approx(data[FEATURES[0]].mean())
    prefix = censor_validation(valid, seed=4)
    x, y, ids = windows(prefix, scaler, 3, endpoints_only=True)
    assert x.shape == (len(ids), 3, len(FEATURES))
    for i, unit in enumerate(ids):
        assert y[i] == 5 - prefix[prefix.unit == unit].cycle.max()


def test_short_window_uses_only_available_history():
    values = np.tile(np.arange(2)[:, None], (1, len(FEATURES)))
    assert window_for(values, 4)[:, 0].tolist() == [0, 0, 0, 1]
    with pytest.raises(ValueError):
        window_for(values[:0], 4)


def test_official_format_validation_and_labels(tmp_path):
    path = tmp_path / "test.txt"
    frame().to_csv(path, sep=" ", index=False, header=False)
    data = load_cycles(path)
    labels = tmp_path / "labels.txt"
    labels.write_text("1\n2\n3\n4\n5\n6\n")
    assert load_test_labels(labels, data)[6] == 6
    labels.write_text("1\n")
    with pytest.raises(ValueError):
        load_test_labels(labels, data)
    broken = frame()
    broken.loc[1, "cycle"] = 4
    broken.to_csv(path, sep=" ", index=False, header=False)
    with pytest.raises(ValueError):
        load_cycles(path)


def test_metrics_and_alert_at_threshold():
    metrics = evaluate([30, 50, 20], [30, 20, 40], 30)
    assert metrics["mae"] == pytest.approx(50 / 3)
    assert (metrics["alert_tp"], metrics["alert_fp"], metrics["alert_fn"]) == (1, 1, 1)
    assert metrics["alert_recall"] == pytest.approx(0.5)


def test_eda_generates_reports(tmp_path, monkeypatch):
    from rul.eda import main

    source = tmp_path / "train.txt"
    output = tmp_path / "eda"
    frame().to_csv(source, sep=" ", index=False, header=False)
    monkeypatch.setattr("sys.argv", ["rul.eda", "--input", str(source), "--output", str(output)])
    main()
    assert (output / "summary.csv").is_file()
    assert (output / "sensor_evolution.png").stat().st_size > 0