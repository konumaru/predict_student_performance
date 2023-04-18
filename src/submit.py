import pathlib
import subprocess

import hydra
import lightgbm
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBClassifier

from common import create_features, parse_labels
from data.raw import jo_wilder
from utils import timer
from utils.io import load_pickle


def predict_lgbm(
    features: np.ndarray,
    level_group: str,
    model_dir: pathlib.Path,
) -> np.ndarray:
    pred = []
    for fold in range(5):
        model = lightgbm.Booster(
            model_file=str(
                model_dir / f"model-lgbm_levelGroup-{level_group}_fold-{fold}"
            )
        )
        _pred = model.predict(features)
        pred.append(_pred)
    return np.mean(pred, axis=0)


def predict_xgb(
    features: np.ndarray,
    level_group: str,
    model_dir: pathlib.Path,
) -> np.ndarray:
    model = XGBClassifier()
    pred = []
    for fold in range(5):
        model.load_model(
            model_dir / f"model-xgb_levelGroup-{level_group}_fold-{fold}"
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

    levelThresholds = load_pickle(
        str(input_dir / "levelTresholds_stacking.pkl")
    )

    env = jo_wilder.make_env()
    iter_test = env.iter_test()
    for test, sample_submission in iter_test:
        level_group = test.level_group.values[0]

        features = create_features(test, level_group, str(input_dir))
        labels = parse_labels(sample_submission)
        labels = labels.with_columns(
            pl.when(pl.col("level") <= 4)
            .then("0-4")
            .otherwise(
                pl.when(pl.col("level") <= 12).then("5-12").otherwise("13-22")
            )
            .alias("level_group")
        )
        data = labels.join(features, how="left", on="session_id")

        cols_drop = load_pickle(input_dir / f"cols_drop_{level_group}.pkl")
        data = data.drop(cols_drop)

        _featrues = (
            data.filter(pl.col("level_group") == level_group)
            .select(
                pl.exclude(
                    [
                        "session_level",
                        "session_id",
                        "correct",
                        "level_group",
                    ]
                )
            )
            .to_numpy()
        )

        if len(_featrues) > 0:
            pred_xgb = predict_xgb(_featrues, level_group, input_dir)
            pred_lgbm = predict_lgbm(_featrues, level_group, input_dir)
            X = np.concatenate(
                (pred_xgb.reshape(-1, 1), pred_lgbm.reshape(-1, 1)), axis=1
            )

            clf = load_pickle(
                str(input_dir / f"stack-ridge_level_group-{level_group}.pkl")
            )
            pred = clf.predict(X)

            sample_submission.loc[
                (data["level_group"] == level_group).to_numpy(), "correct"
            ] = (pred > levelThresholds[level_group]).astype(np.int8)

        print(sample_submission)
        env.predict(sample_submission)


if __name__ == "__main__":
    with timer("Submission"):
        main()
    subprocess.run(["rm", "submission.csv"])
