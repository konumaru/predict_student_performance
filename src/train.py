import os
import pathlib
from copy import deepcopy
from typing import Any, List, Tuple

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from metric import f1_score_with_threshold
from utils import timer
from utils.io import save_pickle, save_txt


def save_xgb_model_as_json(
    save_dir: str, filename: str, models: List[Any]
) -> None:
    # ref: https://qiita.com/Ihmon/items/131e75ced14128e96f61
    for i, model in enumerate(models):
        model.save_model(os.path.join(save_dir, f"{filename}_{i}.json"))


def train(data: pl.DataFrame, model: Any) -> Tuple[np.ndarray, np.ndarray]:
    featrues = (
        data.select(pl.exclude("session_level", "session_id", "level_group"))
        .to_pandas()
        .reset_index(drop=True)
    )
    groups = data.select(pl.col("session_id")).to_numpy()

    oof = np.zeros(len(data))
    cv = GroupKFold(n_splits=5)
    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(featrues, groups=groups)
    ):
        features_train, features_valid = (
            featrues.iloc[train_idx],
            featrues.iloc[valid_idx],
        )

        print(f"\n##### Train fold-{fold} #####\n")
        pred_fold = np.zeros(len(valid_idx))
        for _level in range(1, 19):
            print(f">>>>> LEVEL={_level}")

            features_train_level = features_train.query("level==@_level")
            features_valid_level = features_valid.query("level==@_level")

            X_train_level = features_train_level.drop(
                ["level", "correct"], axis=1
            ).to_numpy()
            y_train_level = features_train_level["correct"].to_numpy()
            X_valid_level = features_valid_level.drop(
                ["level", "correct"], axis=1
            ).to_numpy()
            y_valid_level = features_valid_level["correct"].to_numpy()

            _model = deepcopy(model)
            _model.fit(
                X_train_level,
                y_train_level,
                eval_set=[
                    (X_train_level, y_train_level),
                    (X_valid_level, y_valid_level),
                ],
                verbose=0,
            )
            _model.save_model(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "data",
                    "models",
                    f"level{_level}_fold{fold}.json",
                )
            )

            pred = _model.predict_proba(X_valid_level)[:, 1]
            pred_fold[features_valid["level"] == _level] = pred

        oof[valid_idx] = pred_fold.copy()

        # score = f1_score(
        #     features_valid["correct"].to_numpy(), oof[valid_idx], 0.6
        # )
        score = f1_score_with_threshold(
            features_valid["correct"].to_numpy(), oof[valid_idx], 0.6
        )
        print(f"f1-score of fold-{fold} is", score)

    return oof, featrues["correct"].to_numpy()


def evaluate(y_true, y_pred) -> Tuple[float, float]:
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [
        f1_score_with_threshold(y_true, y_pred, t) for t in thresholds
    ]
    best_score = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_score, best_threshold


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/feature")
    output_dir = pathlib.Path("./data/train")

    data = pl.read_parquet(input_dir / "features.parquet")

    model = XGBClassifier(**cfg.model.params)
    model.set_params(random_state=cfg.seed)

    oof, labels = train(data, model)
    save_pickle(str(output_dir / "oof.pkl"), oof)

    levelThresholds = {}
    for level in range(1, 19):
        flag_level = (data.to_pandas()["level"] == level).to_numpy()
        score, threshold = evaluate(labels[flag_level], oof[flag_level])
        print(f"\nf1-score of oof of {level} =", score)
        levelThresholds[level] = threshold
    save_pickle("./data/models/levelTresholds.pkl", levelThresholds)

    # Evaluate oof.
    score, threshold = evaluate(labels, oof)
    save_txt("./data/train/oof_score.txt", str(score))
    print("\nf1-score of oof is =", score)
    print("threshold is:", threshold)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
