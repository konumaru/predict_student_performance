import pathlib
import subprocess

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBClassifier

from common import create_features, parse_labels
from data.raw import jo_wilder
from utils import timer
from utils.io import load_pickle


def predict_xgb(
    features: np.ndarray, level: int, threshold: float, model_dir: pathlib.Path
):
    model = XGBClassifier()
    pred = []
    for i in range(5):
        model.load_model(model_dir / f"level{level}_fold{i}.json")

        _pred = model.predict_proba(features)[:, 1]
        pred.append(_pred)
    pred = (np.mean(pred, axis=0) > threshold).astype(int)
    return pred


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/upload")

    levelTresholds = load_pickle(str(input_dir / "levelTresholds.pkl"))
    levelgroupRanges = {"0-4": (1, 4), "5-12": (5, 12), "13-22": (13, 18)}

    env = jo_wilder.make_env()
    iter_test = env.iter_test()
    for test, sample_submission in iter_test:
        level_group = test.level_group.values[0]

        features = create_features(test, level_group, str(input_dir))
        labels = parse_labels(sample_submission)
        data = labels.join(features, how="left", on="session_id")

        a, b = levelgroupRanges[level_group]
        for level in range(a, b + 1):
            features_each_level = (
                data.filter(pl.col("level") == level)
                .select(
                    pl.exclude(
                        [
                            "session_level",
                            "session_id",
                            "level",
                            "correct",
                            "level_group",
                        ]
                    )
                )
                .to_numpy()
            )

            if len(features_each_level) > 0:
                sample_submission.loc[
                    (data["level"] == level).to_numpy(), "correct"
                ] = predict_xgb(
                    features_each_level,
                    level,
                    levelTresholds[level],
                    input_dir,
                )

        env.predict(sample_submission)


if __name__ == "__main__":
    with timer("Submission"):
        main()
    subprocess.run(["rm", "submission.csv"])
