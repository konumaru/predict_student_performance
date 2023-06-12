import os
import pathlib
from collections import defaultdict
from typing import List

import hydra
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from sklearn.model_selection import StratifiedGroupKFold

from common import create_features, parse_labels
from utils import timer
from utils.io import load_pickle, save_pickle


class TrainTimeSeriesIterator:
    def __init__(self, train: pd.DataFrame) -> None:
        self.train = train.sort_values(["session_id", "elapsed_time"])

        session_ids = self.train["session_id"].unique()
        level_group = ["0-4", "5-12", "13-22"]
        self.items = [(s, lg) for lg in level_group for s in session_ids]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:
        if len(self.items) == 0:
            raise StopIteration()

        session_id, level_group = self.items.pop(0)
        train_iter = self.train.query(
            f"session_id == {session_id} & level_group == '{level_group}'"
        )
        return train_iter


def get_cols_high_null_ratio(
    data: pl.DataFrame, threshold: float = 0.9
) -> List:
    null_ratio = (data.null_count() / len(data)).to_pandas().T
    return null_ratio.index[(null_ratio > threshold)[0]].tolist()


def get_cols_one_unique_value(data: pl.DataFrame) -> List:
    cols = []
    for c in data.columns:
        if data[c].n_unique() == 1:
            cols.append(c)
    return cols


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/preprocessing")
    output_dir = pathlib.Path("./data/feature")

    train = pl.read_parquet(
        input_dir / "train.parquet",
        n_rows=(10000 if cfg.debug else None),
    )
    level_groups = ["0-4", "5-12", "13-22"]

    # NOTE: Create labels.
    labels = pl.read_parquet(
        input_dir / "labels.parquet", n_rows=(10000 if cfg.debug else None)
    ).to_pandas()
    labels = parse_labels(labels)
    labels = labels.assign(level_group="13-22")
    labels.loc[labels["level"] < 14, "level_group"] = "5-12"
    labels.loc[labels["level"] < 4, "level_group"] = "0-4"

    for level_group in level_groups:
        labels.query("level_group==@level_group").to_parquet(
            output_dir / f"labels_{level_group}.parquet"
        )
    y = labels["correct"].to_numpy()
    groups = labels["session"].to_numpy()
    cv = StratifiedGroupKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
    )
    for fold, (train_idx, valid_idx) in track(
        enumerate(cv.split(labels, y, groups=groups)), total=cv.get_n_splits()
    ):
        suffix = f"fold_{fold}"
        labels.iloc[train_idx].to_parquet(
            output_dir / f"y_train_{suffix}.parquet"
        )
        labels.iloc[valid_idx].to_parquet(
            output_dir / f"y_valid_{suffix}.parquet"
        )

    # NOTE: Create features.
    levelGroup_features = {lg: defaultdict(list) for lg in level_groups}
    for level_group in level_groups:
        print(f"Create features of level_group={level_group}")
        features_pl = create_features(
            train.filter(pl.col("level_group") == level_group),
            level_group,
            input_dir,
        )

        cols_to_drop = []
        cols_to_drop += get_cols_one_unique_value(features_pl)
        cols_to_drop += get_cols_high_null_ratio(features_pl)
        save_pickle(
            output_dir / f"cols_to_drop_{level_group}.pkl", cols_to_drop
        )

        cols_features = list(features_pl.drop(cols_to_drop).columns)
        save_pickle(
            output_dir / f"cols_features_{level_group}.pkl", cols_features
        )

        session_ids = features_pl["session_id"].to_list()
        features_np = (
            features_pl.drop(cols_to_drop)
            .select(pl.exclude("session_id"))
            .to_numpy()
            .astype("float32")
        )
        assert len(features_np) == len(session_ids)

        for i, sess in enumerate(session_ids):
            levelGroup_features[level_group][sess] = features_np[i].tolist()

            if level_group == "5-12":
                levelGroup_features[level_group][sess] = sum(
                    (
                        levelGroup_features["0-4"][sess],
                        features_np[i].tolist(),
                    ),
                    [],
                )
            elif level_group == "13-22":
                levelGroup_features[level_group][sess] = sum(
                    (
                        levelGroup_features["5-12"][sess],
                        features_np[i].tolist(),
                    ),
                    [],
                )

        features_pd = pd.DataFrame.from_dict(
            levelGroup_features[level_group]
        ).T

        features_pd.to_pickle(output_dir / f"features_{level_group}.pkl")

        print(f"Number of drop features: {len(cols_to_drop)}")
        print(f"Shape of features: {features_pd.shape}")
        print(features_pd.head())


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
