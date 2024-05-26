from typing import Any

import numpy as np


def compute_RMSE(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_MASE(y_true: np.array, y_pred: np.array, X_train: np.array, seasonality_period: int, period_weight: float = 1.) -> float:
    # * period_weight
    return np.mean(abs(y_true - y_pred)) / np.mean(np.abs(X_train[:-seasonality_period] - X_train[seasonality_period:]))


def compute_sMAPE(y_true: np.array, y_pred: np.array, period_weight: float = 1.) -> float:
    # * period_weight
    return np.mean(200 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


def compute_OWA(y_true: np.array, y_pred: np.array, X_train: np.array, seasonality_period: int, period_weight: float = 1.) -> float:
    naive2_MASE, naive2_sMAPE = 1.91211, 13.56407
    rel_MASE = compute_MASE(y_true, y_pred,
                            X_train,
                            seasonality_period,
                            period_weight) / naive2_MASE
    rel_sMAPE = compute_sMAPE(y_true, y_pred,
                              period_weight) / naive2_sMAPE
    return (rel_MASE + rel_sMAPE) / 2


def compute_metrics(y_true: np.array, y_pred: np.array, total_time: float, metrics_set: set, **kwargs) -> dict[str, Any]:
    if y_true.shape != y_pred.shape:
        raise RuntimeError(f"Y shapes must be same. Got {y_true.shape} and {y_pred.shape}")

    metrics = {}
    metrics["total_time_sec"] = total_time
    metrics["time_per_elem_sec"] = total_time / y_true.shape[0]
    metrics["sequence_length"] = y_true.shape[0]
    if "time_fit" in kwargs:
        metrics["total_fit_time_sec"] = kwargs["time_fit"]
        metrics["fit_time_per_elem_sec"] = kwargs["time_fit"] / y_true.shape[0]
    if "RMSE" in metrics_set:
        metrics["RMSE"] = compute_RMSE(y_true, y_pred)
    if "MASE" in metrics_set:
        metrics["MASE"] = compute_MASE(y_true, y_pred,
                                       kwargs["X_train"],
                                       kwargs["seasonality_period"],
                                       kwargs["period_weight"])
    if "sMAPE" in metrics_set:
        metrics["sMAPE"] = compute_sMAPE(y_true, y_pred,
                                         kwargs["period_weight"])
    if "OWA" in metrics_set:
        metrics["OWA"] = compute_OWA(y_true, y_pred,
                                     kwargs["X_train"],
                                     kwargs["seasonality_period"],
                                     kwargs["period_weight"])

    return metrics
