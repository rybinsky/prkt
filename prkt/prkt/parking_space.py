import numpy as np
from utils import *


class ParkingSpace:
    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
        self.is_free: bool = True
        self.free_timer: int = 0

    def update_status(self, cars_boxes: list[list[int]]) -> None:
        overlaps = compute_overlaps(np.array([self.get_box()]), np.array(cars_boxes))
        for area_overlap in overlaps:
            max_IoU: float = max(area_overlap)
            if max_IoU < 0.4:
                self.free_timer += 1
                if self.free_timer == 10:
                    self.is_free = True
            else:
                self.is_free = False
                self.free_timer = 0

    def get_box(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    def draw(self, image_to_process: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        x, y, w, h = self.get_box()
        if self.is_free:
            parking_text: str = "FREE SPACE!!!"
        else:
            parking_text: str = "No parking"
        final_image: np.ndarray = draw_bbox(image_to_process, x, y, w, h, parking_text, color)
        return final_image


class ParkingSpace:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.is_free = True
        self.free_timer = 0

    def update_status(self, cars_boxes):
        overlaps = compute_overlaps(np.array([self.get_box()]), np.array(cars_boxes))
        for area_overlap in overlaps:
            max_IoU = max(area_overlap)
            if max_IoU < 0.4:
                self.free_timer += 1
                if self.free_timer == 10:
                    self.is_free = True
            else:
                self.is_free = False
                self.free_timer = 0

    def get_box(self):
        return [self.x, self.y, self.w, self.h]

    def draw(self, image_to_process, color):
        x, y, w, h = self.get_box()
        if self.is_free:
            parking_text = "FREE SPACE!!!"
        else:
            parking_text = "No parking"
        final_image = draw_bbox(image_to_process, x, y, w, h, parking_text, color)
        return final_image
