import os
import pathlib
from typing import Tuple

import hydra
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf

from common import create_features
from utils import timer

FEATURE_DIR = "./data/feature"


class TrainTimeSeriesIterator:
    def __init__(self, train: pl.DataFrame) -> None:
        self.train = train
        self.groups = ["0-4", "5-12", "13-22"]

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, pl.DataFrame]:
        if len(self.groups) == 0:
            raise StopIteration()

        group = self.groups.pop(0)
        train_group = self.train.filter(pl.col("level_group") == group)
        return group, train_group


@hydra.main(config_path="../config", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/preprocessing")
    output_dir = pathlib.Path(FEATURE_DIR)

    train = pl.read_parquet(
        input_dir / "train.parquet",
        n_rows=(10000 if cfg.debug else None),
    )
    labels = pl.read_parquet(input_dir / "labels.parquet")

    train_iter = TrainTimeSeriesIterator(train)
    for level_group, train_batch in train_iter:
        # Create feature for each group level.
        features = create_features(train_batch, "./data/preprocessing", is_test=False)

        # Create label for each group level.
        level_min = int(level_group.split("-")[0])
        level_max = int(level_group.split("-")[1])
        print(level_min, level_max)
        label_batch = labels.filter(
            (pl.col("level") >= level_min) & (pl.col("level") <= level_max)
        )
        print(label_batch.head())

        output = label_batch.join(features, on="session_id", how="left")
        output.write_parquet(str(output_dir / f"{level_group}_train.parquet"))
        print(output.head())


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
