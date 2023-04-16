import pathlib
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score

from metric import f1_score_with_threshold
from utils import timer
from utils.io import load_pickle, save_pickle


def evaluate(y_true, y_pred) -> Tuple[float, float]:
    thresholds = np.linspace(0.4, 0.8, 40)
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

    input_dir = pathlib.Path("./data/train/")
    output_dir = pathlib.Path("./data/models/")

    label = load_pickle(str(input_dir / "label-xgb.pkl")).ravel()
    group = load_pickle(str(input_dir / "group-xgb.pkl")).ravel()

    oof_xgb = load_pickle(str(input_dir / "oof-xgb.pkl"))
    oof_lgbm = load_pickle(str(input_dir / "oof-lgbm.pkl"))

    X = np.concatenate(
        (oof_xgb.reshape(-1, 1), oof_lgbm.reshape(-1, 1)), axis=1
    )

    oof_binary = np.zeros(len(X))
    levelThresholds = {}
    for level_group in ("0-4", "5-12", "13-22"):
        print(f">>>> Level Group={level_group}")
        flag_level = (group == level_group).astype(np.bool8)

        clf = Ridge(alpha=1.0)
        clf.fit(X[flag_level], label[flag_level])
        print("Weights of [xgb, lgbm]: ", clf.coef_)
        save_pickle(
            str(output_dir / f"stack-ridge_level_group-{level_group}.pkl"), clf
        )
        pred = clf.predict(X[flag_level])

        score, threshold = evaluate(label[flag_level], pred)
        print(
            f"f1-score of level_group={level_group:5}:",
            score,
            f"threshold={threshold:4f}",
        )
        levelThresholds[level_group] = threshold

        oof_binary[flag_level] = (pred > threshold).astype(np.int8)
    save_pickle(
        str(output_dir / "levelTresholds_stacking.pkl"), levelThresholds
    )

    score = f1_score(label, oof_binary, average="macro")
    print("\nf1-score of oof is:", score)


if __name__ == "__main__":
    with timer("Stacking"):
        main()
