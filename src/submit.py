import pathlib
import subprocess

import hydra
import lightgbm
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBClassifier

from common import create_features, parse_labels
from data.raw import jo_wilder_310 as jo_wilder  # type: ignore
from utils import timer
from utils.io import load_pickle

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


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/upload")

    threshold_levels = load_pickle(input_dir / "treshold_stacking.pkl")
    limits = {"0-4": (1, 4), "5-12": (5, 12), "13-22": (13, 18)}

    env = jo_wilder.make_env()
    iter_test = env.iter_test()
    for test, sample_submission in iter_test:
        level_group = test.level_group.values[0]
        level_min, level_max = limits[level_group]

        features = create_features(pl.from_pandas(test), str(input_dir))
        labels = parse_labels(sample_submission)

        for level in range(level_min, level_max + 1):
            cols_drop = load_pickle(input_dir / f"colsDrop-level_{level}.pkl")
            data = labels.join(
                features.drop(cols_drop),
                on=["session_id", "level_group"],
                how="left",
            )
            data = data.with_columns(
                pl.col("level_group")
                .map_dict({"0-4": 0, "5-12": 1, "13-22": 2}, default="unknown")
                .cast(pl.Int64)
                .alias("level_group"),
            )
            X = (
                data.filter(pl.col("level") == level)
                .select(
                    pl.exclude(
                        [
                            "session_level",
                            "session_id",
                            "correct",
                            "level",
                            "level_group",
                        ]
                    )
                )
                .to_numpy()
            )

            if len(X) > 0:
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
                ] = (pred > threshold_levels[level]).astype(np.int8)

        env.predict(sample_submission)

        print(sample_submission)


if __name__ == "__main__":
    with timer("Submission"):
        main()
    subprocess.run(["rm", "submission.csv"])
