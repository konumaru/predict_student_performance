import os
from copy import deepcopy
from typing import Any, List, Tuple

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from metric import f1_score
from utils import timer
from utils.io import save_pickle, save_txt


def train(
    data: pl.DataFrame, model: Any
) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    models = []
    oof = np.zeros(len(data))

    X = data.select(
        pl.exclude("session_id", "correct", "level", "level_group")
    ).to_numpy()
    y = data.select(pl.col("correct")).to_numpy()
    group = data.select(pl.col("session_id")).to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, group)):
        print(f"Train fold-{fold} >>>")
        model_cv = deepcopy(model)
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model_cv.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=20,
        )
        models.append(model_cv)
        pred = model_cv.predict_proba(X_valid)[:, 1]
        oof[valid_idx] = pred

        score = f1_score(y_valid, pred, 0.6)
        print(f"f1-score of fold-{fold} is", score)

    return models, oof, y.ravel()


def save_xgb_model_as_json(
    save_dir: str, filename: str, models: List[Any]
) -> None:
    # ref: https://qiita.com/Ihmon/items/131e75ced14128e96f61
    for i, model in enumerate(models):
        model.save_model(os.path.join(save_dir, f"{filename}_{i}.json"))


def evaluate(y_true, y_pred) -> Tuple[float, float]:
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_true, y_pred, t) for t in thresholds]
    best_score = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_score, best_threshold


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    data = pl.read_parquet("./data/feature/train_features.parquet")

    model = XGBClassifier(**cfg.model.params)
    model.set_params(random_state=cfg.seed)

    oof = []
    oof_proba = []
    labels = []
    levelThresholds = {}
    for level in range(1, 19):
        print(f"\n#####  LEVEL={level}  #####\n")
        _data = data.filter(pl.col("level") == level)
        models, _oof, _labels = train(_data, model)

        save_xgb_model_as_json(
            "./data/model", f"{cfg.model.name}_level{level}", models
        )
        save_pickle(
            f"./data/train/{cfg.model.name}_oof_level{level}.pkl", _oof
        )

        score, threshold = evaluate(_labels, _oof)
        print(f"score={score}, threshold={threshold}")
        levelThresholds[level] = threshold

        labels.append(_labels)
        oof_proba.append(_oof)
        oof.append((_oof > threshold).astype(np.int8))

    save_pickle("./data/working/levelTresholds.pkl", levelThresholds)

    # Evaluate oof.
    oof_all = np.concatenate(oof, axis=0)
    label_all = np.concatenate(labels, axis=0)
    score, threshold = evaluate(label_all, oof_all)
    save_txt("./data/train/oof_score.txt", str(score))
    print("threshold is:", threshold)
    print("f1-score of oof is:", score)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
