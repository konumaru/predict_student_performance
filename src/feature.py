import os

import hydra
import numpy as np
import polars as pl
from omegaconf import OmegaConf

from utils import timer
from utils.feature import feature

FEATURE_DIR = "./data/feature"


@feature(FEATURE_DIR)
def dummpy_feature() -> np.ndarray:
    data = np.zeros((5, 100))
    return data


@hydra.main(version_base=None, config_name="config")
def main(cfg: OmegaConf) -> None:
    feat_funcs = [
        dummpy_feature,
    ]

    for func in feat_funcs:
        func()

    # TODO:
    # load data with poloar
    train = pl.read_csv("./data/raw/train.csv")
    # define ts class
    # create feature for each group level


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
