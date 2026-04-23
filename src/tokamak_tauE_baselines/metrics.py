from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true_log = np.asarray(y_true_log).reshape(-1)
    y_pred_log = np.asarray(y_pred_log).reshape(-1)

    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log = float(mean_absolute_error(y_true_log, y_pred_log))
    r2_log = float(r2_score(y_true_log, y_pred_log))

    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)

    rel_err = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-12, None)

    rmse_raw = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_raw = float(mean_absolute_error(y_true, y_pred))
    median_relative_error = float(np.median(rel_err))
    mean_relative_error = float(np.mean(rel_err))

    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_raw": rmse_raw,
        "mae_raw": mae_raw,
        "median_relative_error": median_relative_error,
        "mean_relative_error": mean_relative_error,
    }
