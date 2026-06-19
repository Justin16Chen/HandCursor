"""Real-time hand tracking mouse control system - entry point.

Opens the webcam, detects hand landmarks, and moves the mouse while
pinching. Clap once with both hands to trigger clap detection.

Press 'q' in the preview window to quit.
"""

import time

import cv2

from camera import Camera
from clap_detector import ClapDetector
from hand_gestures import HandGestures
from hand_tracking import HandTracker
from mouse_controller import MouseController


WINDOW_NAME = "HandCursor"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():
    tracker = HandTracker(max_hands=2)
    gestures = HandGestures()
    clap_detector = ClapDetector()
    mouse = MouseController(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    prev_time = time.time()
    draw_indices = False
    enable_hand_cursor = False
    last_pinching = False

    with Camera(width=FRAME_WIDTH, height=FRAME_HEIGHT) as camera:
        while True:
            frame = camera.read()
            if frame is None:
                print("Failed to read frame from webcam.")
                break

            hands = tracker.process(frame)
            tracker.draw(frame, draw_indices=draw_indices)

            now = time.time()
            clap = clap_detector.update(hands, now)
            if clap:
                enable_hand_cursor = not enable_hand_cursor
                print("Clap detected")

            # Default: No pinching and no hand
            pinching = False
            hand = None

            # Choose the first detected hand for mouse cursor
            if hands and enable_hand_cursor:
                # You may prefer to use right hand or left hand explicitly
                # Choose the hand on the right (largest x center)
                hand = max(
                    hands,
                    key=lambda h: sum(x for _, x, _ in h.landmarks) / len(h.landmarks)
                    if h.landmarks
                    else float("-inf"),
                )
         
                pinching = gestures.is_pinching(hand.landmarks)
                gestures.draw(frame, hand.landmarks, pinching=pinching)

                tip = gestures.index_tip(hand.landmarks)
                if tip:
                    mouse.update(tip[0], tip[1], enable_hand_cursor, now)
                # Simulate click (left button) based on pinching state transitions
                if pinching and not last_pinching:
                    mouse.press_left()  # Hold down left click on pinch start
                elif not pinching and last_pinching:
                    mouse.release_left()  # Release left click on pinch end
                last_pinching = pinching
            else:
                # If not enabled, release mouse just in case
                if last_pinching:
                    mouse.release_left()
                    last_pinching = False

            cv2.putText(frame, f"Enable hand cursor: {enable_hand_cursor}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            clap_detector.draw(frame, now)

            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
