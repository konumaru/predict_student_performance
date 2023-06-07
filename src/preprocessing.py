import os
import pathlib

import hydra
import polars as pl
from gensim.models import Word2Vec
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from utils import timer
from utils.io import save_pickle


def train_w2v(
    data: pl.DataFrame,
    col: str,
    output_dir: pathlib.Path,
    embedded_dim: int = 8,
) -> None:
    sentences = data.groupby("session_id").agg(pl.col(col))[col].to_list()
    model = Word2Vec(
        sentences=sentences,
        vector_size=embedded_dim,
        min_count=1,
        workers=4,
        seed=42,
        epochs=10,
    )
    model.wv.save(str(output_dir / f"wv_{col}.wv"))


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
    train = train.with_columns(pl.col("page").cast(pl.Utf8))
    train = train.with_columns(pl.col("level").cast(pl.Utf8))
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
        "page",
        "level",
    ]
    level_groups = ["0-4", "5-12", "13-22"]
    uniques_map = {level_group: {} for level_group in level_groups}
    for level_group in track(level_groups, description="level_group"):
        for col in cols_cat:
            unique_vals = (
                train.filter(pl.col("level_group") == level_group)[col]
                .drop_nulls()
                .unique()
                .to_list()
            )
            uniques_map[level_group][col] = unique_vals
    save_pickle(str(output_dir / "uniques_map.pkl"), uniques_map)

    unique_all_map = {}
    for col in cols_cat:
        unique_vals = train[col].drop_nulls().unique().to_list()
        unique_all_map[col] = unique_vals
    save_pickle(str(output_dir / "unique_all_map.pkl"), unique_all_map)

    # train_w2v(train, "text", output_dir)


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
