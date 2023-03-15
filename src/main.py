import hydra

from config import Config
from utils import timer


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    print("Hello Workd!")


if __name__ == "__main__":
    with timer("main.py"):
        main()
