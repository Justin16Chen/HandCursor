"""Webcam capture wrapper built on OpenCV."""

import cv2


class Camera:
    def __init__(self, source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam (source={source}).")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """Return a single BGR frame, mirrored for a natural selfie view."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.flip(frame, 1)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
