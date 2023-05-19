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


def check_test_data(
    test: pd.DataFrame, sample_submission: pd.DataFrame
) -> None:
    grp = test["level_group"].values[0]
    submission_levels = (
        sample_submission["session_id"].str.extract(r"q(\d+)").astype(int)
    ).to_numpy()
    submission_session_id = (
        sample_submission["session_id"]
        .str.extract(r"(\d+)")[0]
        .unique()
        .astype(int)
    )

    assert len(test) > 0
    assert test["level_group"].nunique() == 1
    assert test["session_id"].nunique() == 1

    assert test["session_id"].unique() == submission_session_id

    # NOTE: The values of level_group is not same as level range.
    # limits = {"0-4": (1, 4), "5-12": (4, 12), "13-22": (13, 18)}
    limits = {"0-4": (1, 4), "5-12": (4, 14), "13-22": (14, 19)}

    for level in submission_levels:
        level_min, level_max = limits[grp]
        assert level_min <= level <= level_max


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    input_dir = pathlib.Path("./data/upload")

    threshold = float(load_txt(input_dir / "threshold-overall-stacking.txt"))

    env = jo_wilder.make_env()
    iter_test = env.iter_test()
    for test, sample_submission in iter_test:
        check_test_data(test, sample_submission)
        features = (
            create_features(pl.from_pandas(test), str(input_dir))
            .select(
                pl.exclude("session_id", "correct", "level_group").cast(
                    pl.Float32
                )
            )
            .to_numpy()
        )
        levels = (
            sample_submission["session_id"]
            .str.extract(r"q(\d+)")
            .astype("int64")
            .to_numpy()
            .ravel()
        )

        for level in levels:
            pred_xgb = predict_xgb(features, input_dir, level)
            pred_lgbm = predict_lgbm(features, input_dir, level)
            X_pred = np.concatenate(
                (pred_xgb.reshape(-1, 1), pred_lgbm.reshape(-1, 1)), axis=1
            )

            clfs = load_pickle(str(input_dir / "stacking_ridge.pkl"))
            pred = np.mean([clf.predict(X_pred) for clf in clfs], axis=0)

            sample_submission.loc[
                (
                    sample_submission["session_id"].str.contains(f"q{level}")
                ).to_list(),
                "correct",
            ] = (pred > threshold).astype(np.int8)

        sample_submission.rename(
            columns={"session_level": "session_id"}, inplace=True
        )
        env.predict(sample_submission)

        print(sample_submission)
        assert sample_submission.columns.tolist() == ["session_id", "correct"]


if __name__ == "__main__":
    with timer("Submission"):
        main()
    subprocess.run(["rm", "submission.csv"])
