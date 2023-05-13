import os
import pathlib
from typing import Tuple

import hydra
import lightgbm
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from metric import f1_score_with_threshold
from utils import timer
from utils.io import save_pickle, save_txt


def fit_cat(
    params,
    X_train,
    y_train,
    X_valid,
    y_valid,
    save_filepath: str,
    seed: int = 42,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        use_best_model=True,
    )
    model.save_model(save_filepath)
    return model


def fit_lgbm(
    params,
    X_train,
    y_train,
    X_valid,
    y_valid,
    save_filepath: str,
    seed: int = 42,
) -> LGBMClassifier:
    model = LGBMClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lightgbm.log_evaluation(50)],
        categorical_feature=["level_group", "level"],
    )
    model.booster_.save_model(save_filepath)
    return model


def fit_xgb(
    params,
    X_train,
    y_train,
    X_valid,
    y_valid,
    save_filepath: str,
    seed: int = 42,
) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50,
    )
    model.save_model(save_filepath + ".json")
    return model


def train(
    cfg: DictConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = pl.read_parquet("./data/feature/features.parquet")

    X = (
        data.select(pl.exclude("session_level", "correct", "session_id"))
        .to_pandas()
        .reset_index(drop=True)
    )
    X["level"] = X["level"].astype("category")
    X["level_group"] = X["level_group"].astype("category")

    y = data.select(pl.col("correct")).to_numpy().ravel()
    session_id = data.select(pl.col("session_id")).to_numpy()

    oof = np.zeros(len(y))
    folds = np.zeros(len(y))
    cv = GroupKFold(n_splits=cfg.n_splits)
    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, groups=session_id)
    ):
        print(f">>>>> Train fold-{fold}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if cfg.model.name == "xgb":
            fit_model = fit_xgb
        elif cfg.model.name == "lgbm":
            fit_model = fit_lgbm
        elif cfg.model.name == "cat":
            fit_model = fit_cat
        else:
            fit_model = fit_xgb

        suffix = f"fold-{fold}"
        model = fit_model(
            cfg.model.params,
            X_train,
            y_train,
            X_valid,
            y_valid,
            save_filepath=os.path.join(
                hydra.utils.get_original_cwd(),
                "data",
                "models",
                f"model-{cfg.model.name}_{suffix}",
            ),
            seed=cfg.seed,
        )

        pred = model.predict_proba(X_valid)[:, 1]
        oof[valid_idx] = pred
        folds[valid_idx] = fold

    score = f1_score_with_threshold(y, oof, 0.63)
    print("f1-score:", score)
    return oof, y, folds


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

    output_dir = pathlib.Path("./data/train")

    oof, label, fold = train(cfg)
    save_pickle(str(output_dir / f"oof-{cfg.model.name}.pkl"), oof)
    save_pickle(str(output_dir / f"label-{cfg.model.name}.pkl"), label)
    save_pickle(str(output_dir / f"fold-{cfg.model.name}.pkl"), fold)

    # Evaluate oof.
    print("\n##### Evaluate #####\n")
    score, threshold = evaluate(label, oof)
    print("\nf1-score of oof is:", score)

    save_txt(f"./data/train/oof_score_{cfg.model.name}.txt", str(score))
    save_pickle(
        f"./data/models/levelTresholds_{cfg.model.name}.pkl", threshold
    )


if __name__ == "__main__":
    with timer("Train"):
        main()
