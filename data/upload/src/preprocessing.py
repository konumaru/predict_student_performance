import os
import pathlib

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

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
        n_rows=(1000 if cfg.debug else None),
    )
    labels = pl.read_csv(input_dir / "train_labels.csv")

    print(train.head())

    train.write_parquet(output_dir / "train.parquet")
    labels.write_parquet(output_dir / "labels.parquet")

    cols_cat = [
        "event_name",
        "name",
        "text",
        "fqid",
        "room_fqid",
        "text_fqid",
    ]
    for col in track(cols_cat):
        unique_vals = train[col].drop_nulls().unique().to_list()
        save_pickle(str(output_dir / f"uniques_{col}.pkl"), unique_vals)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
