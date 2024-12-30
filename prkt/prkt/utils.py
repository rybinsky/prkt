import cv2
import numpy as np


# Функции для подсчета Intersection over Union (IoU)
def calculate_iou(box, boxes, box_area, boxes_area):
    # Считаем IoU
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2] + box[0], boxes[:, 2] + boxes[:, 0])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3] + box[1], boxes[:, 3] + boxes[:, 1])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


# Функция для расчета персечения всех со всеми через IoU
def compute_pairwise_overlaps(boxes1: np.ndarray, boxes2: np.ndarray):
    # Areas of anchors and GT boxes
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = calculate_iou(box2, boxes1, area2[i], area1)
    return overlaps


# Функция для отрисовки Bounding Box в кадре
def draw_bbox(image_to_process, x, y, w, h, parking_text, parking_color=(0, 255, 0)):
    start = (x, y)
    end = (x + w, y + h)
    color = parking_color
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    # Подпись BB
    start = (x, y - 10)
    font_size = 0.4
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 1
    text = parking_text
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)
    return final_image
