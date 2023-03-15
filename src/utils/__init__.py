import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(name) -> Generator:
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")
