import os
import pathlib
from typing import Tuple

import hydra
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf

from common import create_features, parse_labels
from utils import timer


class TrainTimeSeriesIterator:
    # TODO: testデータのようにpandasでロードするほうが望ましいかも
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
    )

    iter_train = TrainTimeSeriesIterator(train, labels)
    for batch_train, batch_labels in iter_train:
        level_group = batch_train["level_group"][0]
        features = create_features(
            batch_train, level_group, "./data/preprocessing"
        )
        features = features.with_columns(
            pl.lit(level_group).alias("level_group")
        )

        batch_labels = parse_labels(batch_labels)
        batch_labels = batch_labels.with_columns(
            pl.lit(level_group).alias("level_group")
        )

        results = batch_labels.join(
            features, on=["session_id", "level_group"], how="left"
        )
        results.write_parquet(output_dir / f"features_{level_group}.parquet")


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
