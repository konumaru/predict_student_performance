import numpy as np


def precision(true_positive, false_positive):
    if true_positive + false_positive == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_positive)


def recall(true_positive, false_negative):
    if true_positive + false_negative == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_negative)


def f1_score(y_true, y_pred, threshold):
    assert len(y_true) == len(
        y_pred
    ), "Length of y_true and y_pred must be equal"

    y_pred_binary = (y_pred >= threshold).astype(int)

    true_positive = np.sum((y_true == 1) & (y_pred_binary == 1))
    false_positive = np.sum((y_true == 0) & (y_pred_binary == 1))
    false_negative = np.sum((y_true == 1) & (y_pred_binary == 0))

    p = precision(true_positive, false_positive)
    r = recall(true_positive, false_negative)

    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)
