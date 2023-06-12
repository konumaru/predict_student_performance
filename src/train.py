import os
import pathlib
from typing import List

import hydra
import lightgbm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from omegaconf import DictConfig, OmegaConf
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from metric import optimize_f1_score
from utils import timer
from utils.io import load_pickle, save_pickle, save_txt


def fit_cat(
    params,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    save_filepath: str,
    seed: int = 42,
    is_balanced: bool = False,
) -> CatBoostClassifier:
    if is_balanced:
        weight_train = compute_sample_weight("balanced", y_train)
    else:
        weight_train = None

    model = CatBoostClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
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
    seed: int = 42,
    is_balanced: bool = False,
) -> LGBMClassifier:
    if is_balanced:
        weight_train = compute_sample_weight("balanced", y_train)
    else:
        weight_train = None

    model = LGBMClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="logloss",
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
    seed: int = 42,
    is_balanced: bool = False,
) -> XGBClassifier:
    if is_balanced:
        weight_train = compute_sample_weight("balanced", y_train)
    else:
        weight_train = None

    model = XGBClassifier(**params)
    model.set_params(random_state=seed)
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
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
        y_train = pd.read_parquet(
            feature_dir / f"y_train_fold_{fold}.parquet"
        ).reset_index(drop=True)
        y_valid = pd.read_parquet(
            feature_dir / f"y_valid_fold_{fold}.parquet"
        ).reset_index(drop=True)

        if "xgb" in cfg.model.name:
            fit_model = fit_xgb
        elif "lgbm" in cfg.model.name:
            fit_model = fit_lgbm
        elif "cat" in cfg.model.name:
            fit_model = fit_cat
        else:
            fit_model = fit_xgb

        suffix = f"{cfg.model.name}_fold_{fold}"
        pred = np.zeros(len(y_valid), dtype=np.float32)

        levelGroup_levels = {"0-4": (1, 4), "5-12": (4, 14), "13-22": (14, 19)}
        for level_group, levels in levelGroup_levels.items():
            X = pd.read_pickle(feature_dir / f"features_{level_group}.pkl")

            for level in range(*levels):
                print(f"fitting model of level={level}")
                params = cfg.model.params["default"]
                # params.update(cfg.model.params[f"level-{level}"])

                _y_train = y_train.query("level == @level")
                _y_valid = y_valid.query("level == @level")

                X_train = X.loc[_y_train["session"]]
                X_valid = X.loc[_y_valid["session"]]

                if cfg.model.is_only_top_features:
                    cols_top_hal_fi: List[int] = load_pickle(
                        feature_dir / "cols_top_hal_fi.pkl"
                    )[level]
                    X_train.drop(cols_top_hal_fi, axis=1, inplace=True)
                    X_valid.drop(cols_top_hal_fi, axis=1, inplace=True)

                model = fit_model(
                    params,
                    X_train.to_numpy(),
                    _y_train["correct"].to_numpy(),
                    X_valid.to_numpy(),
                    _y_valid["correct"].to_numpy(),
                    save_filepath=os.path.join(
                        hydra.utils.get_original_cwd(),
                        "data",
                        "models",
                        f"model_{suffix}_level_{level}",
                    ),
                    seed=cfg.seed,
                    is_balanced=cfg.model.is_balanced,
                )
                pred[y_valid["level"] == level] = model.predict_proba(
                    X_valid.to_numpy()
                )[:, 1]

        save_pickle(output_dir / f"y_pred_{suffix}.pkl", pred)


def evaluate(
    cfg: DictConfig, feature_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    oofs = []
    labels = []
    levels = []
    for fold in range(cfg.n_splits):
        oofs.append(
            load_pickle(
                output_dir / f"y_pred_{cfg.model.name}_fold_{fold}.pkl"
            )
        )
        y_valid = pd.read_parquet(feature_dir / f"y_valid_fold_{fold}.parquet")
        labels.append(y_valid["correct"].to_numpy())
        levels.append(y_valid["level"].to_numpy())

    oof = np.concatenate(oofs)
    labels = np.concatenate(labels)
    levels = np.concatenate(levels)

    score, threshold = optimize_f1_score(labels, oof)
    print("f1-score of oof is:", score, "threshold:", threshold)
    save_txt(output_dir / f"score_{cfg.model.name}.txt", str(score))
    save_txt(
        output_dir / f"threshold_overall_{cfg.model.name}.txt", str(threshold)
    )


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
    evaluate(cfg, feature_dir, output_dir)


if __name__ == "__main__":
    with timer("Train"):
        main()
