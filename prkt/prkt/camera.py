from typing import Sequence

import cv2
from bbox import *
from detector import CV2Image, Detector
from omegaconf import DictConfig


class Camera:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.video_path = cfg.paths.video
        self.net = Detector(cfg.paths.config, cfg.paths.weights, cfg.model.input_size)
        self.possible_parking_spaces = []
        self.class_idx = cfg.CLASS_IDX
        self.confidence_thr = cfg.model.confidence_threshold
        self.nms_thr = cfg.model.nms_threshold
        self.iou_thr_free = cfg.parking.iou_threshold_free
        self.free_space_timer = cfg.parking.free_space_timer
        self.iou_thr_occupied = cfg.parking.iou_threshold_occupied
        self.current_image = None

    def _load_video(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.video_path)

    def detect_objects(self, outputs: Sequence[CV2Image], image: CV2Image) -> tuple[list, list, list]:
        height, width, _ = image.shape
        class_indexes, class_scores, boxes = ([] for i in range(3))
        for out in outputs:
            for obj in out:
                scores = obj[5:]
                class_index = np.argmax(scores)

                # в классе 2 (car) только автомобили
                if class_index == self.class_idx:
                    class_score = scores[class_index]
                    if class_score > 0:
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        obj_width = int(obj[2] * width)
                        obj_height = int(obj[3] * height)
                        box = [
                            center_x - obj_width // 2,
                            center_y - obj_height // 2,
                            obj_width,
                            obj_height,
                        ]
                        # bboxes
                        boxes.append(box)
                        class_indexes.append(class_index)
                        class_scores.append(float(class_score))

        return boxes, class_scores

    def nms_filtering(self, boxes: list, scores: list) -> list:
        chosen_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thr, self.nms_thr)
        return [boxes[i] for i in chosen_boxes]

    def get_current_frame_cars(self, boxes: list, scores: list) -> list:
        chosen_cars_boxes = self.nms_filtering(boxes, scores)
        cars = []
        for car_box in chosen_cars_boxes:
            cars.append(car_box)
            x, y, w, h = car_box
            parking_text = "Car"
            self.current_image = draw_bbox(self.current_image, x, y, w, h, parking_text, (255, 255, 0))
        return cars

    def run(self):
        free_parking_timer = 0
        free_parking_timer_bag1 = 0
        free_parking_space = False
        free_parking_space_box = None
        check_det_frame = None
        video_capture = self._load_video()
        while video_capture.isOpened():
            ret, image_to_process = video_capture.read()
            if not ret:
                break
            outputs = self.net(image_to_process)
            boxes, class_scores = self.detect_objects(outputs, image_to_process)

            if not self.possible_parking_spaces:
                if not boxes:
                    raise ValueError("There is no cars now or no parking places!")
                self.possible_parking_spaces = self.nms_filtering(boxes, class_scores)
            else:
                self.current_image = image_to_process
                current_frame_cars = self.get_current_frame_cars(boxes, class_scores)
                overlaps = compute_overlaps(
                    np.array(self.possible_parking_spaces),
                    np.array(current_frame_cars),
                )

                for parking_space_one, area_overlap in zip(self.possible_parking_spaces, overlaps):
                    max_IoU = max(area_overlap)
                    sort_IoU = np.sort(area_overlap[area_overlap > 0])[::-1]

                    if free_parking_space == False:

                        if 0.0 < max_IoU < self.iou_thr_free:

                            # Количество паркомест по условию 1: 0.0 < IoU < 0.4
                            len_sort = len(sort_IoU)

                            # Количество паркомест по условию 2: IoU > 0.15
                            sort_IoU_2 = sort_IoU[sort_IoU > 0.15]
                            len_sort_2 = len(sort_IoU_2)

                            # Смотрим чтобы удовлятворяло условию 1 и условию 2
                            if (check_det_frame == parking_space_one) & (len_sort != len_sort_2):
                                # Начинаем считать кадры подряд с пустыми координатами
                                free_parking_timer += 1

                            elif check_det_frame == None:
                                check_det_frame = parking_space_one

                            else:
                                # Фильтр от чехарды мест (если место чередуется, то "скачет")
                                free_parking_timer_bag1 += 1
                                if free_parking_timer_bag1 == 2:
                                    # Обнуляем счётчик, если паркоместо "скачет"
                                    check_det_frame = parking_space_one
                                    free_parking_timer = 0

                            # Если более N кадров подряд, то предполагаем, что место свободно
                            if free_parking_timer == self.free_space_timer:
                                # Помечаем свободное место
                                free_parking_space = True
                                free_parking_space_box = parking_space_one
                                # Отрисовываем рамку парковочного места
                                x_free, y_free, w_free, h_free = parking_space_one

                    else:
                        # Если место занимают, то помечается как отсутствие свободных мест
                        overlaps = compute_overlaps(
                            np.array([free_parking_space_box]),
                            np.array(current_frame_cars),
                        )
                        for area_overlap in overlaps:
                            max_IoU = max(area_overlap)
                            if max_IoU > self.iou_thr_occupied:
                                free_parking_space = False

            for parking_space in self.possible_parking_spaces:
                if free_parking_space:
                    if parking_space == [x_free, y_free, w_free, h_free]:
                        parking_text = "FREE SPACE!!!"
                        self.current_image = draw_bbox(
                            image_to_process,
                            x_free,
                            y_free,
                            w_free,
                            h_free,
                            parking_text,
                            (0, 0, 255),
                        )
                    else:
                        x, y, w, h = parking_space
                        parking_text = "No parking"
                        self.current_image = draw_bbox(image_to_process, x, y, w, h, parking_text)
                else:
                    # Координаты и размеры BB
                    x, y, w, h = parking_space
                    parking_text = "No parking"
                    self.current_image = draw_bbox(image_to_process, x, y, w, h, parking_text)

            cv2.imshow("Parking Space", self.current_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        video_capture.release()
        cv2.destroyAllWindows()
