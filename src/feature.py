import os
import pathlib
from typing import Tuple

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from common import create_features
from utils import timer

FEATURE_DIR = "./data/feature"


class TrainTimeSeriesIterator:
    # TODO: testデータのようにpandasでロードするほうが望ましいかも
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


@hydra.main(
    config_path="../config", config_name="config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/preprocessing")
    output_dir = pathlib.Path(FEATURE_DIR)

    train = pl.read_parquet(
        input_dir / "train.parquet",
        n_rows=(10000 if cfg.debug else None),
    )
    labels = pl.read_parquet(input_dir / "labels.parquet")

    # Create feature for each group level.
    features = create_features(train, "./data/preprocessing", is_test=False)

    # TODO: Joinではなくindexで取得する形に変更して処理を高速化
    labels = labels.with_columns(
        pl.when(pl.col("level") < 5)
        .then("0-4")
        .otherwise(
            pl.when(pl.col("level") < 13).then("5-12").otherwise("13-22")
        )
        .alias("level_group")
    )

    output = labels.join(
        features, on=["session_id", "level_group"], how="left"
    )
    output.write_parquet(str(output_dir / "train_features.parquet"))
    print(output)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
