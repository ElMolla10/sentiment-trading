"""
Accuracy, macro-F1, and ordinal MAE (paper Section IV-A).
MAE is computed on class indices mapped to {-1, 0, +1}.
"""
import numpy as np
from sklearn.metrics import f1_score

ORDINAL = np.array([-1, 0, 1])  # [negative_idx=0, neutral_idx=1, positive_idx=2]


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)
    return float((arr_true == arr_pred).mean())


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def ordinal_mae(y_true: list[int], y_pred: list[int]) -> float:
    """MAE on ordinal values {-1, 0, +1} corresponding to class indices {0, 1, 2}."""
    t = ORDINAL[np.array(y_true)]
    p = ORDINAL[np.array(y_pred)]
    return float(np.abs(t - p).mean())


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "f1": macro_f1(y_true, y_pred),
        "mae": ordinal_mae(y_true, y_pred),
    }
