import pathlib
from collections import defaultdict
from typing import List, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import Ridge

from metric import optimize_f1_score
from utils import timer
from utils.io import load_pickle, save_pickle, save_txt


def load_folds_data(
    folds: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_dir = pathlib.Path("./data/train/")
    feature_dir = pathlib.Path("./data/feature/")
    oofs_xgb = []
    oofs_lgbm = []
    labels = []
    levels = []
    for fold in folds:
        oofs_xgb.append(load_pickle(input_dir / f"y_pred_xgb_fold_{fold}.pkl"))
        oofs_lgbm.append(
            load_pickle(input_dir / f"y_pred_lgbm_fold_{fold}.pkl")
        )

        y_valid = pd.read_parquet(feature_dir / f"y_valid_fold_{fold}.parquet")
        labels.append(y_valid["correct"].to_numpy())
        levels.append(y_valid["level"].to_numpy())

    oof = np.concatenate(
        [
            np.concatenate(oofs_xgb).reshape(-1, 1),
            np.concatenate(oofs_lgbm).reshape(-1, 1),
        ],
        axis=1,
    )
    label = np.concatenate(labels)
    level = np.concatenate(levels)
    return oof, label, level


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_dir = pathlib.Path("./data/models/")

    labels = []
    oofs = []
    clfs = defaultdict(list)
    levels = []
    for fold in range(cfg.n_splits):
        X_train, y_train, levels_train = load_folds_data(
            [i for i in range(cfg.n_splits) if i != fold]
        )
        X_valid, y_valid, levels_valid = load_folds_data([fold])

        pred = np.zeros(len(y_valid))
        for level in range(1, 19):
            clf = Ridge(alpha=1.0, random_state=cfg.seed)
            clf.fit(
                X_train[levels_train == level], y_train[levels_train == level]
            )
            clfs[level].append(clf)

            pred[levels_valid == level] = clf.predict(
                X_valid[levels_valid == level]
            )

        oofs.append(pred)
        labels.append(y_valid)
        levels.append(levels_valid)

    save_pickle(str(output_dir / "stacking_ridge.pkl"), clfs)

    print("\n##### Evaluate #####\n")
    oof = np.concatenate(oofs)
    label = np.concatenate(labels)
    levels = np.concatenate(levels)

    score, threshold = optimize_f1_score(label, oof)
    print(
        "\nf1-score of oof is:",
        score,
        "threshold:",
        round(threshold, 4),
    )
    save_txt("./data/train/score_stacking.txt", str(score))
    save_txt("./data/train/threshold_overall_stacking.txt", str(threshold))


if __name__ == "__main__":
    with timer("Stacking"):
        main()
