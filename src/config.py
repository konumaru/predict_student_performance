from dataclasses import dataclass, field
from typing import Any, Dict, List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

hydra_dir = "./data/.hydra/"
hydra_config = {
    "mode": "MULTIRUN",
    "job": {
        "config": {"override_dirname": {"exclude_keys": ["{exp}", "{seed}"]}},
    },
    "run": {"dir": hydra_dir + "${exp}/${hydra.job.override_dirname}/seed=${seed}"},
    "sweep": {"dir": hydra_dir + "${exp}/${hydra.job.override_dirname}/seed=${seed}"},
}


@dataclass
class Config:
    hydra: DictConfig = OmegaConf.create(hydra_config)
    # defaults: List[Any] = field(default_factory=lambda: defaults)

    seed: int = 42
    exp: str = "rf"

    is_eval: bool = True


cs: ConfigStore = ConfigStore.instance()
cs.store("config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
