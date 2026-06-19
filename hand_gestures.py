"""Hand gesture detection from landmark data."""

import cv2

from hand_tracking import HandTracker


class HandGestures:
    WRIST = 0
    PINKY_MCP = 17
    THUMB_TIP = 4
    INDEX_TIP = 8

    TIP_COLOR = (0, 255, 255)
    TIP_RADIUS = 12

    def __init__(self, pinch_ratio=0.21):
        self.pinch_ratio = pinch_ratio

    def is_pinching(self, landmarks):
        """Return True when thumb and index fingertips are close enough to pinch."""
        hand_scale = HandTracker.landmark_distance(
            landmarks, self.WRIST, self.PINKY_MCP
        )
        pinch_distance = HandTracker.landmark_distance(
            landmarks, self.THUMB_TIP, self.INDEX_TIP
        )
        if hand_scale is None or pinch_distance is None:
            return False

        threshold = hand_scale * self.pinch_ratio
        return pinch_distance < threshold

    def index_tip(self, landmarks):
        """Return (x, y) pixel coordinates for the index fingertip, or None."""
        for lm_id, x, y in landmarks:
            if lm_id == self.INDEX_TIP:
                return x, y
        return None

    def draw(self, frame, landmarks, pinching=False):
        """Draw fingertip highlights and pinch status on the frame."""
        points = {lm_id: (x, y) for lm_id, x, y in landmarks}
        for tip_id in (self.THUMB_TIP, self.INDEX_TIP):
            if tip_id in points:
                cv2.circle(frame, points[tip_id], self.TIP_RADIUS, self.TIP_COLOR, cv2.FILLED)

        if pinching:
            cv2.putText(
                frame,
                "Pinching",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
