import pathlib
from typing import List, Union

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer

from common import preprocessing
from utils import timer
from utils.io import save_pickle


@hydra.main(config_path="../config", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    input_dir = pathlib.Path("./data/raw")
    output_dir = pathlib.Path("./data/preprocessing")

    train = pd.read_csv(
        input_dir / "train.csv",
        nrows=(10000 if cfg.debug else None),
    )
    test = pd.read_csv(input_dir / "test.csv")

    # TODO: textカラムをtf-idfでベクトル化, text_fqidも結合して一緒にベクトル化するのもあり？
    # text, text_fqid

    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform(train["text"].dropna().tolist())
    print(X.toarray())

    train = preprocessing(train)
    print(train.head())
    print(train.info())

    cols_cat = ["event_name", "name", "fqid", "room_fqid_1", "room_fqid_2"]
    for col in cols_cat:
        unique_vals = sorted(train[col].unique())
        map_dict = {val: i for i, val in enumerate(unique_vals)}
        save_pickle(str(output_dir / f"map_dict_{col}.pkl"), map_dict)


if __name__ == "__main__":
    with timer("main.py"):
        main()
