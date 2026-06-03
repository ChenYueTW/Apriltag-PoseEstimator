import cv2
import time


class FPSCaculator:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

    def update(self):
        self.frame_count += 1

        now = time.time()
        elapsed = now - self.start_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = now

        return self.fps

    def draw(self, image, position=(8, 18)):
        cv2.putText(
            image,
            f"FPS: {self.fps:.1f}",
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )