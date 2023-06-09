import pathlib
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Union

import lightgbm
import numpy as np
import polars as pl
from xgboost import XGBClassifier

from common import create_features, parse_labels
from data.raw import jo_wilder_310 as jo_wilder  # type: ignore
from utils import timer
from utils.io import load_pickle, load_txt

N_FOLD = 5


def predict_lgbm(
    X: Union[List, np.ndarray],
    model_dir: pathlib.Path,
    level: int,
) -> np.ndarray:
    pred = []
    for fold in range(N_FOLD):
        model = lightgbm.Booster(
            model_file=str(
                model_dir / f"model_lgbm_fold_{fold}_level_{level}.txt"
            )
        )
        _pred = model.predict(X)
        pred.append(_pred)
    return np.mean(pred, axis=0)


def predict_xgb(
    X: Union[List, np.ndarray],
    model_dir: pathlib.Path,
    level: int,
) -> np.ndarray:
    model = XGBClassifier()
    pred = []
    for fold in range(N_FOLD):
        model.load_model(
            model_dir / f"model_xgb_fold_{fold}_level_{level}.json"
        )
        _pred = model.predict_proba(X)[:, 1]
        pred.append(_pred)
    return np.mean(pred, axis=0)


# TODO: モデルのロードを予測の前に行うことで高速化


def main() -> None:
    input_dir = pathlib.Path("./data/upload")

    threshold = float(load_txt(input_dir / "threshold_overall_stacking.txt"))
    clfs = load_pickle(str(input_dir / "stacking_ridge.pkl"))
    level_groups = ["0-4", "5-12", "13-22"]

    levelGroup_features: Dict[str, List[Any]] = {
        lg: defaultdict(list) for lg in level_groups
    }
    cols_to_drop = {}
    for level_group in level_groups:
        cols_to_drop[level_group] = load_pickle(
            input_dir / f"cols_to_drop_{level_group}.pkl"
        ) + ["session_id"]

    env = jo_wilder.make_env()
    iter_test = env.iter_test()
    for test, sample_submission in iter_test:
        level_group = test.iloc[0]["level_group"]
        session_id = test.iloc[0]["session_id"]

        sample_submission = parse_labels(sample_submission)

        X = (
            create_features(
                pl.from_pandas(
                    test.sort_values(by="index").reset_index(drop=True)
                ),
                level_group,
                input_dir,
            )
            .drop(cols_to_drop[level_group])
            .to_numpy()
            .astype("float32")
        )
        if level_group == "0-4":
            levelGroup_features[level_group][session_id] = X[0].tolist()
        elif level_group == "5-12":
            levelGroup_features[level_group][session_id] = sum(
                (
                    levelGroup_features["0-4"][session_id],
                    X[0].tolist(),
                ),
                [],
            )
        elif level_group == "13-22":
            levelGroup_features[level_group][session_id] = sum(
                (
                    levelGroup_features["5-12"][session_id],
                    X[0].tolist(),
                ),
                [],
            )

        for level in sample_submission["level"].unique():
            X_feat = levelGroup_features[level_group][session_id]
            pred_xgb = predict_xgb([X_feat], input_dir, level)
            pred_lgbm = predict_lgbm([X_feat], input_dir, level)
            X_pred = np.concatenate(
                (pred_xgb.reshape(-1, 1), pred_lgbm.reshape(-1, 1)), axis=1
            )

            pred = np.mean(
                [clf.predict(X_pred) for clf in clfs[level]], axis=0
            )

            sample_submission.loc[
                sample_submission["session_id"].str.contains(f"q{level}"),
                "correct",
            ] = (pred >= threshold).astype(np.int8)

        env.predict(sample_submission[["session_id", "correct"]])

        print(sample_submission[["session_id", "correct"]])


if __name__ == "__main__":
    with timer("Submission"):
        main()
    subprocess.run(["rm", "submission.csv"])
