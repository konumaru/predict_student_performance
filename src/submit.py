import pathlib
import subprocess
from collections import defaultdict
from typing import Dict, List, Tuple, Union

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
    models: List[lightgbm.Booster],
) -> np.ndarray:
    pred = [model.predict(X) for model in models]
    return np.mean(pred, axis=0)


def predict_xgb(
    X: Union[List, np.ndarray],
    models: List[XGBClassifier],
) -> np.ndarray:
    pred = [model.predict_proba(X)[:, 1] for model in models]
    return np.mean(pred, axis=0)


def load_model(input_dir: pathlib.Path) -> Tuple[Dict, Dict]:
    xgb_models = defaultdict(list)
    xgb_v2_models = defaultdict(list)
    lgbm_models = defaultdict(list)
    lgbm_v2_models = defaultdict(list)
    for level in range(1, 19):
        for fold in range(N_FOLD):
            model = XGBClassifier()
            model.load_model(
                input_dir / f"model_xgb_fold_{fold}_level_{level}.json"
            )
            xgb_models[level].append(model)

            model = XGBClassifier()
            model.load_model(
                input_dir / f"model_xgb_v2_fold_{fold}_level_{level}.json"
            )
            xgb_v2_models[level].append(model)

            model = lightgbm.Booster(
                model_file=str(
                    input_dir / f"model_lgbm_fold_{fold}_level_{level}.txt"
                )
            )
            lgbm_models[level].append(model)

            model = lightgbm.Booster(
                model_file=str(
                    input_dir / f"model_lgbm_v2_fold_{fold}_level_{level}.txt"
                )
            )
            lgbm_v2_models[level].append(model)

    return xgb_models, lgbm_models, xgb_v2_models, lgbm_v2_models


def main() -> None:
    input_dir = pathlib.Path("./data/upload")

    threshold = float(load_txt(input_dir / "threshold_overall_stacking.txt"))
    xgb_models, lgbm_models, xgb_v2_models, lgbm_v2_models = load_model(
        input_dir
    )
    clfs = load_pickle(str(input_dir / "stacking_ridge.pkl"))
    cols_top_hal_fi: List[int] = load_pickle(
        input_dir / "cols_to_drop_top_hal_fi.pkl"
    )
    level_groups = ["0-4", "5-12", "13-22"]

    cols_to_drop = {}
    for level_group in level_groups:
        cols_to_drop[level_group] = load_pickle(
            input_dir / f"cols_to_drop_{level_group}.pkl"
        ) + ["session_id"]

    levelGroup_features = defaultdict(list)

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

        levelGroup_features[session_id].extend(X[0].tolist())
        X_feat = levelGroup_features[session_id]

        for level in sample_submission["level"].unique():
            pred_xgb = predict_xgb([X_feat], xgb_models[level])[0]
            pred_lgbm = predict_lgbm([X_feat], lgbm_models[level])[0]
            pred_xgb_v2 = predict_xgb(
                [np.delete(X_feat, cols_top_hal_fi[level])],
                xgb_v2_models[level],
            )[0]
            pred_lgbm_v2 = predict_lgbm(
                [np.delete(X_feat, cols_top_hal_fi[level])],
                lgbm_v2_models[level],
            )[0]

            X_pred = np.expand_dims(
                np.array([pred_xgb, pred_lgbm, pred_xgb_v2, pred_lgbm_v2]),
                axis=0,
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
