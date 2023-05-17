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
            model_file=str(model_dir / f"model-lgbm_level-{level}_fold-{fold}")
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
            model_dir / f"model-xgb_level-{level}_fold-{fold}.json"
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
        features = create_features(pl.from_pandas(test), str(input_dir))
        labels = parse_labels(sample_submission)

        for level in labels["level"].unique():
            cols_drop = load_pickle(input_dir / f"colsDrop-level_{level}.pkl")
            cols_drop += ["session_id", "level_group"]
            X = features.drop(cols_drop).to_numpy()

            pred_xgb = predict_xgb(X, input_dir, level)
            pred_lgbm = predict_lgbm(X, input_dir, level)
            X_pred = np.concatenate(
                (pred_xgb.reshape(-1, 1), pred_lgbm.reshape(-1, 1)), axis=1
            )

            clf = load_pickle(
                str(input_dir / f"stack-ridge-level_{level}.pkl")
            )
            pred = clf.predict(X_pred)

            sample_submission.loc[
                (labels["level"] == level).to_numpy(), "correct"
            ] = (pred > threshold).astype(np.int8)

        env.predict(sample_submission)

        print(sample_submission)


if __name__ == "__main__":
    with timer("Submission"):
        main()
    subprocess.run(["rm", "submission.csv"])
