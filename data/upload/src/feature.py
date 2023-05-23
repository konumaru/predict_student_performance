import os
import pathlib
from typing import List, Tuple

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
    def __init__(self, train: pl.DataFrame, labels: pl.DataFrame) -> None:
        self.train = train
        self.labels = labels
        self.groups = ["0-4", "5-12", "13-22"]

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(self.groups) == 0:
            raise StopIteration()

        group = self.groups.pop(0)
        group_min, group_max = (int(s) for s in group.split("-"))
        batch_train = self.train.filter(pl.col("level_group") == group)

        regex = "|".join([f"_q{i}" for i in range(group_min, group_max + 1)])
        batch_labels = self.labels.filter(
            pl.col("session_id").str.contains(regex)
        )
        return batch_train.to_pandas(), batch_labels.to_pandas()


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

    uniques_map = load_pickle(input_dir / "uniques_map.pkl")
    train = pl.read_parquet(
        input_dir / "train.parquet",
        n_rows=(10000 if cfg.debug else None),
    )

    for level_group in ["0-4", "5-12", "13-22"]:
        print(f"Create features of level_group={level_group}")
        features = create_features(
            train.filter(pl.col("level_group") == level_group),
            level_group,
            uniques_map[level_group],
        )

        cols_to_drop = []
        cols_to_drop += get_cols_one_unique_value(features)
        cols_to_drop += get_cols_high_null_ratio(features)
        save_pickle(
            output_dir / f"cols_to_drop_{level_group}.pkl", cols_to_drop
        )

        features = features.drop(cols_to_drop)
        features_pd = (
            features.to_pandas().set_index("session_id").astype("float32")
        )
        features_pd.to_parquet(output_dir / f"features_{level_group}.parquet")

        print(f"Number of drop features: {len(cols_to_drop)}")
        print(f"Number of features: {features.shape[1]}")
        print(features.head())

    labels = pl.read_parquet(
        input_dir / "labels.parquet", n_rows=(10000 if cfg.debug else None)
    ).to_pandas()
    labels = parse_labels(labels)
    labels = labels.assign(level_group="13-22")
    labels.loc[labels["level"] < 14, "level_group"] = "5-12"
    labels.loc[labels["level"] < 4, "level_group"] = "0-4"
    print(labels)

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


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
