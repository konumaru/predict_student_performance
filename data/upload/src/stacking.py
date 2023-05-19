import pathlib
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import Ridge

from metric import f1_score_with_threshold
from utils import timer
from utils.io import load_pickle, save_pickle, save_txt


def evaluate(y_true, y_pred) -> Tuple[float, float]:
    thresholds = np.linspace(0.4, 0.8, 40)
    f1_scores = [
        f1_score_with_threshold(y_true, y_pred, t) for t in thresholds
    ]
    best_score = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_score, best_threshold


def load_oof(folds: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    input_dir = pathlib.Path("./data/train/")
    feature_dir = pathlib.Path("./data/feature/")
    oofs_xgb = []
    oofs_lgbm = []
    labels = []
    for fold in folds:
        oofs_xgb.append(load_pickle(input_dir / f"y_pred_xgb_fold_{fold}.pkl"))
        oofs_lgbm.append(
            load_pickle(input_dir / f"y_pred_lgbm_fold_{fold}.pkl")
        )
        labels.append(load_pickle(feature_dir / f"y_valid_fold_{fold}.pkl"))
    oof = np.concatenate(
        [
            np.concatenate(oofs_xgb).reshape(-1, 1),
            np.concatenate(oofs_lgbm).reshape(-1, 1),
        ],
        axis=1,
    )
    label = np.concatenate(labels)
    return oof, label


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_dir = pathlib.Path("./data/models/")

    labels = []
    oofs = []
    clfs = []
    for fold in range(cfg.n_splits):
        X_train, y_train = load_oof(
            [i for i in range(cfg.n_splits) if i != fold]
        )
        X_valid, y_valid = load_oof([fold])

        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_train)
        clfs.append(clf)

        pred = clf.predict(X_valid)
        oofs.append(pred)
        labels.append(y_valid)
        print("Weights of [xgb, lgbm]: ", clf.coef_)

    save_pickle(str(output_dir / "stacking_ridge.pkl"), clfs)

    print("\n##### Evaluate #####\n")
    oof = np.concatenate(oofs)
    label = np.concatenate(labels)
    score, threshold = evaluate(label, oof)
    print("f1-score of oof is:", score)
    save_txt("./data/train/score-stacking.txt", str(score))
    save_txt("./data/train/threshold-overall-stacking.txt", str(threshold))


if __name__ == "__main__":
    with timer("Stacking"):
        main()
