import os
import pathlib
from typing import List, Tuple

import hydra
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

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


def get_cols_high_unique_ratio(
    data: pl.DataFrame, threshold: float = 0.2
) -> List:
    cols = []
    for c in data.columns:
        if data[c].n_unique() / len(data) < threshold:
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
    results = results.with_columns(
        pl.exclude(
            "session_id", "session_level", "correct", "level_group", "level"
        ).cast(pl.Float32),
        pl.col("level").cast(pl.Int32),
    )

    print("The Number of Features:", results.shape[1])
    print(results)

    for level in track(range(1, 19)):
        results_by_level = results.filter(pl.col("level") == level)

        cols_drop = get_cols_high_null_ratio(results_by_level)
        cols_drop += get_cols_high_unique_ratio(
            results_by_level.select(
                pl.exclude("session_id", "level_group", "level", "correct")
            )
        )
        save_pickle(output_dir / f"colsDrop-level_{level}.pkl", cols_drop)
        results_by_level = results_by_level.drop(cols_drop)

        results_by_level.write_parquet(
            output_dir / f"features-level_{level}.parquet"
        )


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
