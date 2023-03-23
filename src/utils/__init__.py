import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(name) -> Generator:
    t0 = time.time()
    yield
