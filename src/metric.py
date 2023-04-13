from sklearn.metrics import f1_score


def f1_score_with_threshold(y_true, y_pred, threshold) -> float:
    assert len(y_true) == len(
        y_pred
    ), "Length of y_true and y_pred must be equal"

    y_pred_binary = (y_pred >= threshold).astype(int)

    return f1_score(y_true, y_pred_binary, average="macro")  # type: ignore
