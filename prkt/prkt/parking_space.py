import numpy as np
from utils import *


class ParkingSpace:
    def __init__(self, x: int, y: int, w: int, h: int, is_free: bool, free_time_thr: int) -> None:
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h
        self.is_free: bool = is_free
        self.free_timer: int = 0
        self.free_thr: int = free_time_thr

    def update_status(self, space_overlaps: list[list[int]]) -> None:
        max_IoU: float = max(space_overlaps)
        if 0 < max_IoU < 0.4:
            self.free_timer += 1
            if self.free_timer == self.free_thr:
                self.is_free = True
        else:
            self.is_free = False
            self.free_timer = 0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def free(self) -> bool:
        return self.is_free
