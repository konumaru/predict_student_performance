import gc
import os
import pathlib
from collections import defaultdict
from typing import Tuple, Union

import hydra
import lightgbm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from omegaconf import DictConfig, OmegaConf
from xgboost import XGBClassifier

from metric import f1_score_with_threshold
from utils import timer
from utils.io import load_pickle, save_pickle, save_txt


def fit_cat(
    params,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    save_filepath: str,
    weight_train: Union[np.ndarray, None] = None,
    weight_valid: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        use_best_model=True,
        sample_weight=weight_train,
    )
    model.save_model(save_filepath)
    return model


def fit_lgbm(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    save_filepath: str,
    weight_train: Union[np.ndarray, None] = None,
    weight_valid: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> LGBMClassifier:
    model = LGBMClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_sample_weight=weight_valid,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lightgbm.log_evaluation(50), lightgbm.early_stopping(50)],
    )
    model.booster_.save_model(
        save_filepath + ".txt", num_iteration=model.best_iteration_
    )
    return model


def fit_xgb(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    save_filepath: str,
    weight_train: Union[np.ndarray, None] = None,
    weight_valid: Union[np.ndarray, None] = None,
    seed: int = 42,
) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        sample_weight_eval_set=weight_valid,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=50,
    )
    model.save_model(save_filepath + ".json")
    return model


def train(
    cfg: DictConfig, feature_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    for fold in range(cfg.n_splits):
        print(f">>>> Train fold={fold}")
        X_train = pd.read_parquet(feature_dir / f"X_train_fold_{fold}.parquet")
        X_valid = pd.read_parquet(feature_dir / f"X_valid_fold_{fold}.parquet")
        y_train = load_pickle(feature_dir / f"y_train_fold_{fold}.pkl").ravel()
        y_valid = load_pickle(feature_dir / f"y_valid_fold_{fold}.pkl").ravel()

        level_train = X_train["level"].to_numpy()
        level_valid = X_valid["level"].to_numpy()

        X_train.drop(columns=["level"], inplace=True)
        X_valid.drop(columns=["level"], inplace=True)

        if cfg.model.name == "xgb":
            fit_model = fit_xgb
        elif cfg.model.name == "lgbm":
            fit_model = fit_lgbm
        elif cfg.model.name == "cat":
            fit_model = fit_cat
        else:
            fit_model = fit_xgb

        suffix = f"{cfg.model.name}_fold_{fold}"
        pred = np.zeros(len(y_valid), dtype=np.float32)
        for level in range(1, 19):
            print(f"training level={level}")
            flag_train = level_train == level
            flag_valid = level_valid == level

            cols_drop = load_pickle(
                feature_dir / f"cols_to_drop_level_{level}.pkl"
            )

            model = fit_model(
                cfg.model.params,
                X_train[flag_train].drop(columns=cols_drop).to_numpy(),
                y_train[flag_train],
                X_valid[flag_valid].drop(columns=cols_drop).to_numpy(),
                y_valid[flag_valid],
                save_filepath=os.path.join(
                    hydra.utils.get_original_cwd(),
                    "data",
                    "models",
                    f"model_{suffix}_level_{level}",
                ),
                seed=cfg.seed,
            )
            pred[flag_valid] = model.predict_proba(
                X_valid[flag_valid].drop(columns=cols_drop)
            )[:, 1]

        save_pickle(output_dir / f"y_pred_{suffix}.pkl", pred)

        del X_train, X_valid, y_train, y_valid
        gc.collect()


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
    feature_dir = pathlib.Path("./data/feature")

    print("\n##### Train Model #####\n")
    train(cfg, feature_dir, output_dir)

    print("\n##### Evaluate #####\n")
    oofs = []
    labels = []
    levels = []
    for fold in range(cfg.n_splits):
        oofs.append(
            load_pickle(
                output_dir / f"y_pred_{cfg.model.name}_fold_{fold}.pkl"
            )
        )
        labels.append(load_pickle(feature_dir / f"y_valid_fold_{fold}.pkl"))
        X_valid = pd.read_parquet(feature_dir / f"X_valid_fold_{fold}.parquet")
        levels.append(X_valid["level"].to_numpy())

    oof = np.concatenate(oofs)
    labels = np.concatenate(labels)
    levels = np.concatenate(levels)

    threshold_levels = defaultdict(float)

    for level in range(1, 19):
        score, threshold = evaluate(
            labels[levels == level], oof[levels == level]
        )
        threshold_levels[level] = threshold
        print(
            f"f1-score of q{level}:",
            round(score, 6),
            "threshold:",
            round(threshold, 4),
        )

    save_pickle(
        output_dir / f"threshold_{cfg.model.name}.pkl", threshold_levels
    )
    # Evaluate oof.
    score, threshold = evaluate(labels, oof)
    print("\nf1-score of oof is:", score)
    save_txt(output_dir / f"score-{cfg.model.name}.txt", str(score))
    save_txt(
        output_dir / f"threshold-overall-{cfg.model.name}.txt", str(threshold)
    )


if __name__ == "__main__":
    with timer("Train"):
        main()
