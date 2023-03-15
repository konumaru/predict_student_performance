import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    squared_error = (y_pred - y_true) ** 2
    mean_squared_error = squared_error.mean()
    rmse = np.sqrt(mean_squared_error)
    return rmse
