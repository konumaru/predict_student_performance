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
from utils.io import load_pickle


def predict_lgbm(
    features: pd.DataFrame,
    model_dir: pathlib.Path,
) -> np.ndarray:
    pred = []
    for fold in range(5):
        model = lightgbm.Booster(
            model_file=str(model_dir / f"model-lgbm_fold-{fold}")
        )
        _pred = model.predict(features)
        pred.append(_pred)
    return np.mean(pred, axis=0)


def predict_xgb(
    features: pd.DataFrame,
    model_dir: pathlib.Path,
) -> np.ndarray:
    model = XGBClassifier()
    pred = []
    for fold in range(5):
        model.load_model(model_dir / f"model-xgb_fold-{fold}.json")
        _pred = model.predict_proba(features)[:, 1]
        pred.append(_pred)
    return np.mean(pred, axis=0)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/upload")

    threshold = load_pickle(str(input_dir / "levelTresholds_stacking.pkl"))

    env = jo_wilder.make_env()
    iter_test = env.iter_test()
    for test, sample_submission in iter_test:
        cols_drop = load_pickle(input_dir / "cols_drop.pkl")
        features = create_features(test, str(input_dir))
        features = features.drop(cols_drop)

        labels = parse_labels(sample_submission)
        data = labels.join(features, how="left", on="session_id")
        X = data.select(
            pl.exclude(["session_level", "session_id", "correct"])
        ).to_pandas()
        X["level"] = X["level"].astype("category")
        X["level_group"] = X["level_group"].astype("category")

        if len(X) > 0:
            pred_xgb = predict_xgb(X, input_dir)
            pred_lgbm = predict_lgbm(X, input_dir)
            X = np.concatenate(
                (pred_xgb.reshape(-1, 1), pred_lgbm.reshape(-1, 1)), axis=1
            )

            clf = load_pickle(str(input_dir / f"stack-ridge.pkl"))
            pred = clf.predict(X)

            sample_submission.loc[:, "correct"] = (pred > threshold).astype(
                np.int8
            )

        print(sample_submission)
        env.predict(sample_submission)


if __name__ == "__main__":
    with timer("Submission"):
        main()
    subprocess.run(["rm", "submission.csv"])
