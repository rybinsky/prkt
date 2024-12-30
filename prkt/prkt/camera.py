from typing import Sequence

import cv2
from bbox import *
from detector import CV2Image, Detector
from omegaconf import DictConfig
from parking_space import ParkingSpace


class Camera:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.video_path = cfg.paths.video
        self.net = Detector(cfg.paths.config, cfg.paths.weights, cfg.model.input_size)
        self.possible_parking_spaces: list[ParkingSpace] = []
        self.free_parking_spaces: list[ParkingSpace] = []
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

    def _nms_filtering(self, boxes: list, scores: list) -> list:
        chosen_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thr, self.nms_thr)
        return [boxes[i] for i in chosen_boxes]

    def get_parking_spaces(self, boxes: list, scores: list) -> list[ParkingSpace]:
        filtered_boxes = self._nms_filtering(boxes, scores)
        return [ParkingSpace(*box, is_free=False, free_time_thr=self.free_space_timer) for box in filtered_boxes]

    def get_current_frame_cars_boxes(self, boxes: list, scores: list) -> list:
        chosen_cars_boxes = self._nms_filtering(boxes, scores)
        cars = []
        for car_box in chosen_cars_boxes:
            cars.append(car_box)
            x, y, w, h = car_box
            parking_text = "Car"
            self.current_image = draw_bbox(self.current_image, x, y, w, h, parking_text, (255, 255, 0))
        return cars

    def get_free_parking_spaces(self, current_frame_cars_boxes: list) -> None:
        overlaps = compute_pairwise_overlaps(
            np.array([space.box for space in self.possible_parking_spaces]),
            np.array(current_frame_cars_boxes),
        )
        for parking_space, space_overlaps in zip(self.possible_parking_spaces, overlaps):
            parking_space.update_status(space_overlaps)

    def run(self):
        video_capture = self._load_video()
        while video_capture.isOpened():
            ret, image_to_process = video_capture.read()
            if not ret:
                break
            outputs = self.net(image_to_process)
            boxes, class_scores = self.detect_objects(outputs, image_to_process)

            if not self.possible_parking_spaces:
                if not boxes:
                    raise ValueError("There is no cars now or no parking spaces!")
                self.possible_parking_spaces = self.get_parking_spaces(boxes, class_scores)
            else:
                self.current_image = image_to_process
                current_frame_cars_boxes = self.get_current_frame_cars_boxes(boxes, class_scores)
                self.get_free_parking_spaces(current_frame_cars_boxes)

            for parking_space in self.possible_parking_spaces:
                x, y, w, h = parking_space.box
                if parking_space.free:
                    parking_text = "FREE SPACE!!!"
                    color = (0, 0, 255)
                else:
                    parking_text = "No parking"
                    color = (0, 255, 0)
                self.current_image = draw_bbox(image_to_process, x, y, w, h, parking_text, color)
            cv2.imshow("Parking Space", self.current_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        video_capture.release()
        cv2.destroyAllWindows()
