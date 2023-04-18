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
        eval_metric="auc",
        callbacks=[lightgbm.log_evaluation(50)],
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
    model.save_model(save_filepath)
    return model


def train(
    cfg: DictConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oofs = []
    labels = []
    groups = []
    folds = []

    for level_group in ("0-4", "5-12", "13-22"):
        print(f"\n##### LEVEL GROUP={level_group} #####\n")
        data = pl.read_parquet(
            f"./data/feature/features_{level_group}.parquet"
        )

        X = data.select(
            pl.exclude("session_level", "correct", "session_id", "level_group")
        ).to_numpy()
        y = data.select(pl.col("correct")).to_numpy().ravel()
        session_ids = data.select(pl.col("session_id")).to_numpy()

        oof_level_group = np.zeros(len(y))
        folds_level_group = np.zeros(len(y))
        cv = GroupKFold(n_splits=cfg.n_splits)
        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, groups=session_ids)
        ):
            print(f">>>>> Train fold-{fold}")
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            if cfg.model.name == "xgb":
                fit_model = fit_xgb
            elif cfg.model.name == "lgbm":
                fit_model = fit_lgbm
            elif cfg.model.name == "cat":
                fit_model = fit_cat
            else:
                fit_model = fit_xgb

            suffix = f"levelGroup-{level_group}_fold-{fold}"
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
            oof_level_group[valid_idx] = pred
            folds_level_group[valid_idx] = fold

        groups.append(np.repeat(level_group, len(y)))
        folds.append(folds_level_group)

        score = f1_score_with_threshold(y, oof_level_group, 0.6)
        print(f"f1-score of level_group={level_group}:", score)

        oofs.append(oof_level_group)
        labels.append(y)

    oof_all = np.concatenate(oofs)
    label_all = np.concatenate(labels)
    fold_all = np.concatenate(folds)
    group_all = np.concatenate(groups)
    return oof_all, label_all, fold_all, group_all


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

    oof, label, fold, group = train(cfg)
    save_pickle(str(output_dir / f"oof-{cfg.model.name}.pkl"), oof)
    save_pickle(str(output_dir / f"label-{cfg.model.name}.pkl"), label)
    save_pickle(str(output_dir / f"fold-{cfg.model.name}.pkl"), fold)
    save_pickle(str(output_dir / f"group-{cfg.model.name}.pkl"), group)

    # oof = load_pickle(str(output_dir / f"oof-{cfg.model.name}.pkl"))
    # label = load_pickle(str(output_dir / f"label-{cfg.model.name}.pkl"))
    # group = load_pickle(str(output_dir / f"group-{cfg.model.name}.pkl"))

    # Evaluate oof.
    print("\n##### Evaluate #####\n")
    oof_binary = np.zeros(len(oof))
    levelThresholds = {}
    for level_group in ("0-4", "5-12", "13-22"):
        flag_level = (group == level_group).astype(np.bool8)
        score, threshold = evaluate(label[flag_level], oof[flag_level])
        print(
            f"f1-score of level_group={level_group:5}:",
            score,
            f"threshold={threshold:4f}",
        )
        levelThresholds[level_group] = threshold

        oof_binary[flag_level] = (oof[flag_level] > threshold).astype(np.int8)

    score = f1_score(label, oof_binary, average="macro")
    print("\nf1-score of oof is:", score)

    save_txt(f"./data/train/oof_score_{cfg.model.name}.txt", str(score))
    save_pickle(
        f"./data/models/levelTresholds_{cfg.model.name}.pkl", levelThresholds
    )


if __name__ == "__main__":
    with timer("Train"):
        main()
