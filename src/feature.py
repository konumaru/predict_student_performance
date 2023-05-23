import os
import pathlib
from typing import List, Tuple

import hydra
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from common import create_features, parse_labels
from utils import timer
from utils.io import save_pickle


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

    train = pl.read_parquet(
        input_dir / "train.parquet",
        n_rows=(10000 if cfg.debug else None),
    )
    labels = pl.read_parquet(
        input_dir / "labels.parquet", n_rows=(10000 if cfg.debug else None)
    ).to_pandas()

    features = create_features(train, "./data/preprocessing")
    labels_parsed = parse_labels(labels)

    results = labels_parsed.join(
        features, on=["session_id", "level_group"], how="left"
    )

    # NOTE: Check the number of each level_group values.
    level_group_levels = {
        "0-4": [1, 2, 3],
        "5-12": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "13-22": [14, 15, 16, 17, 18],
    }
    for level_group, levels in (
        results.groupby("level_group")
        .agg(pl.col("level").unique())
        .iter_rows()
    ):
        assert level_group_levels[level_group] == levels

    results = results.with_columns(
        pl.col("level_group").map_dict({"0-4": 0, "5-12": 1, "13-22": 2})
    )
    results = results.with_columns(
        pl.exclude(
            "session_id", "session_level", "correct", "level_group", "level"
        ).cast(pl.Float32),
        pl.col("level").cast(pl.Int32),
        pl.col("level_group").cast(pl.Int32),
    )

    print("The Number of Features:", results.shape[1])
    print(results)
    results.write_parquet(output_dir / "features.parquet")

    for level in range(1, 19):
        cols_to_drop = []
        cols_to_drop += get_cols_one_unique_value(
            results.filter(pl.col("level") == level)
        )
        cols_to_drop += get_cols_high_null_ratio(
            results.filter(pl.col("level") == level)
        )

        cols_to_drop.remove("level")
        cols_to_drop.remove("level_group")
        save_pickle(
            output_dir / f"cols_to_drop_level_{level}.pkl", cols_to_drop
        )
        print(f"Number of cols_to_drop of level={level}:", len(cols_to_drop))

    X = results.select(
        pl.exclude("session_id", "correct", "level_group")
    ).to_pandas()
    y = results.select(pl.col("correct")).to_numpy()
    groups = results.select(pl.col("session_id")).to_numpy()
    # cv = GroupKFold(n_splits=cfg.n_splits)
    cv = StratifiedGroupKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
    )
    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y, groups=groups)
    ):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        suffix = f"fold_{fold}"
        X_train.to_parquet(output_dir / f"X_train_{suffix}.parquet")
        X_valid.to_parquet(output_dir / f"X_valid_{suffix}.parquet")
        save_pickle(output_dir / f"y_train_{suffix}.pkl", y_train)
        save_pickle(output_dir / f"y_valid_{suffix}.pkl", y_valid)

        print(f">>> fold={fold}")
        print(f"X_train:\n{X_train.info()}")


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
