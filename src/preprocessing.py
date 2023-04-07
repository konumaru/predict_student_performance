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


@hydra.main(config_path="../config", config_name="config.yaml", version_base="1.3")
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

    cols_cat = ["event_name", "name", "fqid", "room_fqid_1", "room_fqid_2"]
    for col in cols_cat:
        unique_vals = train[col].unique().sort().to_list()
        map_dict = {val: i for i, val in enumerate(unique_vals)}
        save_pickle(str(output_dir / f"map_dict_{col}.pkl"), map_dict)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
