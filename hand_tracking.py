"""Hand landmark detection using MediaPipe Hands."""

import math
from dataclasses import dataclass

import cv2
import mediapipe as mp

WRIST = 0
PINKY_MCP = 17
FINGER_BASES = (1, 5, 9, 13, 17)
FINGER_TIPS = (4, 8, 12, 16, 20)


@dataclass
class Hand:
    landmarks: list
    label: str


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame):
        """Detect hands in a BGR frame.

        Returns a list of Hand objects with pixel landmarks and a handedness label.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        h, w = frame.shape[:2]
        hands = []
        if results.multi_hand_landmarks:
            handedness = results.multi_handedness or []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = "Unknown"
                if i < len(handedness):
                    label = handedness[i].classification[0].label
                landmarks = [
                    (idx, int(lm.x * w), int(lm.y * h))
                    for idx, lm in enumerate(hand_landmarks.landmark)
                ]
                hands.append(Hand(landmarks=landmarks, label=label))
        self._last_results = results
        return hands

    def draw(self, frame, draw_indices=False):
        """Draw the most recently detected landmarks onto the frame, with indices next to each point."""
        results = getattr(self, "_last_results", None)
        if results and results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                # Draw indices next to each landmark
                if draw_indices:
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.putText(
                            frame,
                            str(idx),
                            (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA
                        )
             
        return frame

    @staticmethod
    def landmark_distance(landmarks, index1, index2):
        """Return the Euclidean distance between two landmarks.

        `landmarks` is a list of (id, x, y) tuples as returned by `process()`.
        Returns None if either index is not present in `landmarks`.
        """
        points = {lm_id: (x, y) for lm_id, x, y in landmarks}
        if index1 not in points or index2 not in points:
            return None
        (x1, y1), (x2, y2) = points[index1], points[index2]
        return math.hypot(x2 - x1, y2 - y1)

    def close(self):
        self.hands.close()
