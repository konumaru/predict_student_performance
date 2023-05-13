import os
import pathlib
import pickle
from typing import Any, Callable, List

import numpy as np


def feature(save_dir: str, use_cache: bool = True) -> Callable:
    def wrapper(func: Callable) -> Callable:
        def run_func(*args, **kwargs) -> Any:
            filepath = os.path.join(save_dir, func.__name__ + ".pkl")

            if use_cache and os.path.exists(filepath):
                print(f"Use cached data, {filepath}")
                with open(filepath, "rb") as file:
                    data = pickle.load(file)
                    return data

            # NOTE: Run if not use or exist cache.
            print("Run Function of", func.__name__)
            result = func(*args, **kwargs)

            assert result.ndim == 2, "Feature dim must be 2d."
            with open(filepath, "wb") as file:
                pickle.dump(result, file)

            return result

        return run_func

    return wrapper


def load_feature(dirpath: str, feature_names: List[str]) -> np.ndarray:
    saved_dir = pathlib.Path(dirpath)

    feats = []
    for feature_name in feature_names:
        filepath = str(saved_dir / (feature_name + ".pkl"))
        with open(filepath, "rb") as file:
            feat = pickle.load(file)
        feats.append(feat)

    return np.concatenate(feats, axis=1)
