import os
from copy import deepcopy
from typing import Any, List, Tuple

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

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

        score = f1_score(y_valid, (pred > 0.5).astype(int))
        print(f"f1-score of fold-{fold} is", score)

    return models, oof, y.ravel()


def save_xgb_model_as_json(
    save_dir: str, filename: str, models: List[Any]
) -> None:
    # ref: https://qiita.com/Ihmon/items/131e75ced14128e96f61

    for i, model in enumerate(models):
        model.save_model(os.path.join(save_dir, f"{filename}_{i}.json"))


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    data = pl.read_parquet("./data/feature/train_features.parquet")

    model = XGBClassifier(**cfg.model.params)
    model.set_params(random_state=cfg.seed)

    oofs = []
    labels = []
    for level in range(1, 19):
        print(f"\n#####  LEVEL={level} #####\n")
        _data = data.filter(pl.col("level") == level)
        models, _oof, _label = train(_data, model)

        save_xgb_model_as_json(
            "./data/model", f"{cfg.model.name}_level{level}", models
        )
        save_pickle(
            f"./data/train/{cfg.model.name}_oof_level{level}.pkl", _oof
        )

        labels.append(_label)
        oofs.append(_oof)

    # Evaluate oof.
    oof_all = np.concatenate(oofs, axis=0)
    label_all = np.concatenate(labels, axis=0)
    score = f1_score(label_all, (oof_all > 0.5).astype(int))
    save_txt(f"./data/train/{cfg.model.name}_oof_score.txt", str(score))
    print("f1-score of oof is:", score)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
