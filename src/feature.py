import hydra
import numpy as np

from config import Config
from utils import timer
from utils.feature import feature

FEATURE_DIR = "./data/feature"


@feature(FEATURE_DIR)
def dummpy_feature() -> np.ndarray:
    data = np.zeros((5, 100))
    return data


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    feat_funcs = [
        dummpy_feature,
    ]

    for func in feat_funcs:
        func()


if __name__ == "__main__":
    with timer("main.py"):
        main()
