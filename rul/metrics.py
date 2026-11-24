import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error, precision_score, recall_score


def evaluate(actual, predicted, threshold: float = 30) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    error = predicted - actual
    true_alert, predicted_alert = actual <= threshold, predicted <= threshold
    tn, fp, fn, tp = confusion_matrix(true_alert, predicted_alert, labels=[False, True]).ravel()
    result = {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "nasa_score": float(np.sum(np.expm1(np.where(error < 0, -error / 13, error / 10)))),
        "alert_precision": float(precision_score(true_alert, predicted_alert, zero_division=0)),
        "alert_recall": float(recall_score(true_alert, predicted_alert, zero_division=0)),
        "alert_tn": int(tn), "alert_fp": int(fp), "alert_fn": int(fn), "alert_tp": int(tp),
    }
    for name, mask in (("near", actual <= threshold), ("far", actual > threshold)):
        if mask.any():
            result[f"mae_{name}"] = float(np.mean(np.abs(error[mask])))
    return result