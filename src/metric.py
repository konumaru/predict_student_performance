from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score


def f1_score_with_threshold(y_true, y_pred, threshold) -> float:
    assert len(y_true) == len(
        y_pred
    ), "Length of y_true and y_pred must be equal"

    y_pred_binary = (y_pred >= threshold).astype(int)

    return f1_score(y_true, y_pred_binary, average="macro")  # type: ignore


def optimize_f1_score(y_true, y_pred) -> Tuple[float, float]:
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [
        f1_score_with_threshold(y_true, y_pred, t) for t in thresholds
    ]
    best_score = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_score, best_threshold


def optimize_f1_score_each_levels(labels, oof, levels) -> Dict[int, float]:
    threshold_levels = defaultdict(float)
    for level in range(1, 19):
        score, threshold = optimize_f1_score(
            labels[levels == level], oof[levels == level]
        )
        threshold_levels[level] = threshold
        print(
            f"f1-score of q{level}:",
            round(score, 6),
            "threshold:",
            round(threshold, 4),
        )
    return threshold_levels
