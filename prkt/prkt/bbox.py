from typing import Optional

from utils import *


class BoundingBox:
    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        """
        Инициализация bounding box с координатами левого верхнего угла (x, y),
        шириной и высотой.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def area(self) -> float:
        """
        Вычисление площади bounding box.
        """
        return self.width * self.height

    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """
        Вычисление пересечения с другим bounding box.
        Возвращает новый BoundingBox, если пересечение существует, иначе None.
        """
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        if x2 > x1 and y2 > y1:
            return BoundingBox(x1, y1, x2 - x1, y2 - y1)
        else:
            return None

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """
        Вычисление объединения с другим bounding box.
        Возвращает новый BoundingBox, который охватывает оба bounding box.
        """
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x + self.width, other.x + other.width)
        y2 = max(self.y + self.height, other.y + other.height)

        return BoundingBox(x1, y1, x2 - x1, y2 - y1)

    def iou(self, other: "BoundingBox") -> float:
        """
        Вычисление коэффициента IoU (Intersection over Union) с другим bounding box.
        """
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0

        intersection_area = intersection.area()
        union_area = self.area() + other.area() - intersection_area

        return intersection_area / union_area

    def __repr__(self) -> str:
        """
        Строковое представление bounding box.
        """
        return f"BoundingBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
