from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

CV2Image = cv2.typing.MatLike


class Detector:
    def __init__(self, cfg: Path, weights: Path, input_size: int) -> None:
        self.input_size: int = input_size
        self.net: cv2.dnn.Net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.layer_names: Sequence[str] = self.net.getLayerNames()
        self.out_layers_indexes: Sequence[int] = self.net.getUnconnectedOutLayers()
        self.out_layers: list[str] = [self.layer_names[index - 1] for index in self.out_layers_indexes]

    def __call__(self, img: CV2Image) -> Sequence[CV2Image]:
        blob = self.__preprocess_image(img)
        self.net.setInput(blob)
        return self.net.forward(self.out_layers)

    def __preprocess_image(self, img: CV2Image) -> CV2Image:
        blob = cv2.dnn.blobFromImage(
            img,
            1 / 255,
            (self.input_size, self.input_size),
            (0, 0, 0),
            swapRB=True,
            crop=False,
        )
        return blob
