import pickle
from typing import Any


def load_pickle(filepath: str) -> Any:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(filepath: str, data: Any) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
