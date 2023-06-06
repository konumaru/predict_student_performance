import pathlib

import hydra
import optuna
import pandas as pd
import xgboost
import yaml
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from utils import timer

LEVEL = 15

feature_dir = pathlib.Path("./data/feature")
if LEVEL < 4:
    level_group = "0-4"
if LEVEL < 14:
    level_group = "5-12"
else:
    level_group = "13-22"

labels = pd.read_parquet(feature_dir / f"labels_{level_group}.parquet")
features = pd.read_parquet(feature_dir / f"features_{level_group}.parquet")

y = labels.query(f"level == {LEVEL}")
X = features.loc[y["session"]].to_numpy()
session = y["session"].to_numpy()


def objective(trial):
    param = {
        "verbosity": 0,
        "booster": "gbtree",
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "objective": "binary:logistic",
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.1, log=True
        ),
        "n_estimators": trial.suggest_int("n_estimators", 100, 700),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.4, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6),
        "lambda": trial.suggest_float("lambda", 1e-2, 20, log=True),
        "alpha": trial.suggest_float("alpha", 1e-2, 20, log=True),
    }
    model = xgboost.XGBClassifier(**param)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "roc_auc"  # "f1_macro"
    scores = cross_validate(
        model,
        X,
        y["correct"].to_numpy(),
        groups=session,
        cv=cv,
        scoring=scoring,
    )
    score = scores["test_score"].mean()
    return score


def output_best_params(
    study: optuna.study.study.Study, output_dir: pathlib.Path
) -> None:
    with open(output_dir / f"xgb_level{LEVEL}.yaml", "w") as f:
        yaml.dump(study.best_trial.params, f)


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_dir = pathlib.Path("./data/optimize")
    output_dir.mkdir(exist_ok=True, parents=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    output_best_params(study, output_dir)


if __name__ == "__main__":
    with timer("Train"):
        main()
