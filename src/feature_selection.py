import os
import pathlib
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils import timer
from utils.io import load_pickle, save_pickle


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
        verbose=None,
    )
    model.save_model(save_filepath + ".json")
    return model


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_dir = pathlib.Path("./data/feature")
    feature_dir = pathlib.Path("./data/feature")

    fold = 0
    y_train = pd.read_parquet(
        feature_dir / f"y_train_fold_{fold}.parquet"
    ).reset_index(drop=True)
    y_valid = pd.read_parquet(
        feature_dir / f"y_valid_fold_{fold}.parquet"
    ).reset_index(drop=True)
    levelGroup_levels = {"0-4": (1, 4), "5-12": (4, 14), "13-22": (14, 19)}

    drop_cols_index = defaultdict(list)
    baselines = {}
    for level_group, levels in levelGroup_levels.items():
        X = pd.read_pickle(feature_dir / f"features_{level_group}.pkl")

        for level in range(*levels):
            print(f"Training model of level={level}")
            params = cfg.model.params["default"]
            params["n_estimators"] = 100
            _y_train = y_train.query("level == @level")
            _y_valid = y_valid.query("level == @level")

            X_train = X.loc[_y_train["session"]]
            X_valid = X.loc[_y_valid["session"]]

            model = fit_xgb(
                params,
                X_train.to_numpy(),
                _y_train["correct"].to_numpy(),
                X_valid.to_numpy(),
                _y_valid["correct"].to_numpy(),
                save_filepath=os.path.join(
                    hydra.utils.get_original_cwd(), "data", "models", "tmp"
                ),
                seed=cfg.seed,
                is_balanced=cfg.model.is_balanced,
            )
            pred = model.predict_proba(X_valid.to_numpy())[:, 1]

            score = log_loss(_y_valid["correct"], pred)
            baselines[level] = score

            features = load_pickle(
                feature_dir / f"cols_features_{level_group}.pkl"
            )[1:]
            for idx_drop in track(range(X.shape[1])):
                model = fit_xgb(
                    params,
                    X_train.drop([idx_drop], axis=1).to_numpy(),
                    _y_train["correct"].to_numpy(),
                    X_valid.drop([idx_drop], axis=1).to_numpy(),
                    _y_valid["correct"].to_numpy(),
                    save_filepath=os.path.join(
                        hydra.utils.get_original_cwd(), "data", "models", "tmp"
                    ),
                    seed=cfg.seed,
                    is_balanced=cfg.model.is_balanced,
                )
                pred = model.predict_proba(
                    X_valid.drop([idx_drop], axis=1).to_numpy()
                )[:, 1]
                score = log_loss(_y_valid["correct"], pred)

                if baselines[level] > score:
                    col_drop = features[idx_drop]
                    drop_cols_index[level].append(col_drop)

            save_pickle(
                output_dir / f"drop_cols_step_{level}.pkl", drop_cols_index
            )
    save_pickle(output_dir / "drop_cols_step.pkl", drop_cols_index)


if __name__ == "__main__":
    with timer("Train"):
        main()
