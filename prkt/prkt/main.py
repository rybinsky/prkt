import os

import cv2
import hydra
import numpy as np
from camera import Camera
from detector import Detector
from omegaconf import DictConfig
from utils import *


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    camera_view = Camera(cfg)
    camera_view.run()


if __name__ == "__main__":
    import os

    print(
        "Текущий рабочий каталог:",
        os.path.isdir("/Users/nikita/study/python/prkt/prkt/prkt/config"),
    )
    main()
