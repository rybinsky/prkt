import os

import cv2


class MediaSource:
    def __init__(self, path: str):
        self.path = path

    def read(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def is_opened(self) -> bool:
        raise NotImplementedError


class VideoSource(MediaSource):
    def __init__(self, path: str):
        super().__init__(path)
        self.cap = cv2.VideoCapture()
        self.cap.open(path, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise ConnectionError(f"Не удалось открыть поток: {path}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def is_opened(self) -> bool:
        return self.cap.isOpened()


class ImageSource(MediaSource):
    def __init__(self, path: str):
        super().__init__(path)
        self.images = sorted(
            [os.path.join(path, img) for img in os.listdir(path) if img.endswith((".png", ".jpg", ".jpeg"))]
        )
        self.current_frame = 0

    def read(self):
        if self.current_frame < len(self.images):
            image = cv2.imread(self.images[self.current_frame])
            self.current_frame += 1
            return True, image
        return False, None

    def release(self):
        pass

    def is_opened(self) -> bool:
        return self.current_frame < len(self.images)
