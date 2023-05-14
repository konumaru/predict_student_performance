import pathlib
import pickle
from typing import Any, Union


def load_pickle(filepath: Union[str, pathlib.Path]) -> Any:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(filepath: Union[str, pathlib.Path], data: Any) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def save_txt(filepath: Union[str, pathlib.Path], data: Any) -> None:
    with open(filepath, "w") as f:
        f.write(data)


def load_txt(filepath: Union[str, pathlib.Path]) -> Any:
    with open(filepath, "r") as f:
        data = f.read()
    return data
