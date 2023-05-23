import pathlib
import subprocess

import hydra
import lightgbm
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBClassifier

from common import create_features, parse_labels
from data.raw import jo_wilder_310 as jo_wilder  # type: ignore
from utils import timer
from utils.io import load_pickle, load_txt

N_FOLD = 5


def predict_lgbm(
    features: np.ndarray,
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
        _pred = model.predict(features)
        pred.append(_pred)
    return np.mean(pred, axis=0)


def predict_xgb(
    features: np.ndarray,
    model_dir: pathlib.Path,
    level: int,
) -> np.ndarray:
    model = XGBClassifier()
    pred = []
    for fold in range(N_FOLD):
        model.load_model(
            model_dir / f"model_xgb_fold_{fold}_level_{level}.json"
        )
        _pred = model.predict_proba(features)[:, 1]
        pred.append(_pred)
    return np.mean(pred, axis=0)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    input_dir = pathlib.Path("./data/upload")

    threshold = float(load_txt(input_dir / "threshold_overall_stacking.txt"))
    uniques_map = load_pickle(input_dir / "uniques_map.pkl")

    env = jo_wilder.make_env()
    iter_test = env.iter_test()
    for test, sample_submission in iter_test:
        level_group = test["level_group"].values[0]
        cols_to_drop = load_pickle(
            input_dir / f"cols_to_drop_{level_group}.pkl"
        )
        X = (
            create_features(
                pl.from_pandas(test), level_group, uniques_map[level_group]
            )
            .drop(cols_to_drop)
            .to_pandas()
            .set_index("session_id")
            .astype("float32")
            .to_numpy()
        )
        sample_submission = parse_labels(sample_submission)

        for level in sample_submission["level"].unique():
            pred_xgb = predict_xgb(X, input_dir, level)
            pred_lgbm = predict_lgbm(X, input_dir, level)
            X_pred = np.concatenate(
                (pred_xgb.reshape(-1, 1), pred_lgbm.reshape(-1, 1)), axis=1
            )

            clfs = load_pickle(str(input_dir / "stacking_ridge.pkl"))
            pred = np.mean([clf.predict(X_pred) for clf in clfs], axis=0)

            sample_submission.loc[
                sample_submission["session_id"].str.contains(f"q{level}"),
                "correct",
            ] = (pred > threshold).astype(np.int8)

        env.predict(sample_submission[["session_id", "correct"]])

        print(sample_submission[["session_id", "correct"]])


if __name__ == "__main__":
    with timer("Submission"):
        main()
    subprocess.run(["rm", "submission.csv"])
