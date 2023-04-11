import os
import pathlib
from typing import List, Union

import hydra
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf

from common import preprocessing
from utils import timer
from utils.io import save_pickle


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/raw")
    output_dir = pathlib.Path("./data/preprocessing")

    train = pl.read_csv(
        input_dir / "train.csv",
        n_rows=(10000 if cfg.debug else None),
    )
    train = preprocessing(train)
    train.write_parquet(str(output_dir / "train.parquet"))
    print(train.head())

    cols_cat = [
        "event_name",
        "name",
        "fqid",
        "room_fqid",
        "room_fqid_1",
        "room_fqid_2",
    ]
    for col in cols_cat:
        unique_vals = train.select(col).unique().to_pandas()[col].tolist()
        save_pickle(str(output_dir / f"uniques_{col}.pkl"), unique_vals)

    labels = pl.read_csv(
        "./data/raw/train_labels.csv", n_rows=(10000 if cfg.debug else None)
    )
    labels = labels.with_columns(
        labels["session_id"]
        .str.split_exact("_", 1)
        .struct.rename_fields(["session_id", "level"])
        .alias("fields")
        .to_frame()
        .unnest("fields")
    )
    labels = labels.with_columns(
        pl.col("session_id").cast(pl.Int64).alias("session_id"),
        pl.col("level").str.replace("q", "").cast(pl.Int32).alias("level"),
    )
    labels.write_parquet(str(output_dir / "labels.parquet"))
    print(labels.head())


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
