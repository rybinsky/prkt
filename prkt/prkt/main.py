import os

import hydra
from camera import Camera
from omegaconf import DictConfig
from utils import *


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    camera_view = Camera(cfg)
    camera_view.run()


if __name__ == "__main__":
    print(
        "Текущий рабочий каталог:",
        os.path.isdir("/Users/nikita/study/python/prkt/prkt/prkt/config"),
    )
    main()
