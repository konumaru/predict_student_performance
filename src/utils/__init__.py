import os
import random
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch


@contextmanager
def timer(name) -> Generator:
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore

    os.environ["PYTHONHASHSEED"] = str(seed)
