"""Clap detection using finger direction, approach velocity, and parallel planes."""

import math

import cv2

from hand_tracking import HandTracker, WRIST, PINKY_MCP, FINGER_BASES, FINGER_TIPS


class ClapDetector:
    MIN_APPROACH_TIME = 0.05
    MAX_CONFIRM_TIME = 0.3
    APPROACH_RESET_DEBOUNCE = 0.025  # Seconds both must be false

    def __init__(
        self,
        clap_ratio=0.03,
        confirm_clap_ratio=0.8,
        cooldown=0.2,
    ):
        self.clap_ratio = clap_ratio
        self.confirm_clap_ratio = confirm_clap_ratio
        self.cooldown = cooldown

        self._prev_base = {"left": None, "right": None}
        self._prev_time = None
        self._approach_time = 0.0
        self._confirm_time = 0.0
        self._last_clap_time = 0.0
        self._clap_display_until = 0.0
        self._viz = None

        # Debouncer state for approach reset
        self._approach_both_false_time = 0.0

    @staticmethod
    def _plane_line_endpoints(point, plane_dir, half_length=140):
        """Return two endpoints for the line through point with the given plane direction vector."""
        px, py = point
        dx, dy = plane_dir
        return (
            (int(px - dx * half_length), int(py - dy * half_length)),
            (int(px + dx * half_length), int(py + dy * half_length)),
        )

    @staticmethod
    def _normal_gap_end(left_base, right_base, normal):
        """Point on the left plane reached by projecting right_base along the shared normal."""
        lx, ly = left_base
        rx, ry = right_base
        nx, ny = normal
        gap = (rx - lx) * nx + (ry - ly) * ny
        return int(lx + gap * nx), int(ly + gap * ny)

    @staticmethod
    def _points(landmarks):
        return {lm_id: (x, y) for lm_id, x, y in landmarks}

    @staticmethod
    def _normalize(vector):
        length = math.hypot(vector[0], vector[1])
        if length == 0:
            return None
        return vector[0] / length, vector[1] / length

    def _finger_dir(self, landmarks):
        """Average unit direction from each finger base to its tip."""
        points = self._points(landmarks)
        direction = [0.0, 0.0]
        for base_id, tip_id in zip(FINGER_BASES, FINGER_TIPS):
            if base_id not in points or tip_id not in points:
                continue
            bx, by = points[base_id]
            tx, ty = points[tip_id]
            dx, dy = tx - bx, ty - by
            unit = self._normalize((dx, dy))
            if unit:
                direction[0] += unit[0]
                direction[1] += unit[1]
        return self._normalize(tuple(direction))

    def _average_finger_base(self, landmarks):
        points = self._points(landmarks)
        xs, ys = [], []
        for base_id in FINGER_BASES:
            if base_id in points:
                xs.append(points[base_id][0])
                ys.append(points[base_id][1])
        if not xs:
            return None
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _hand_scale(self, landmarks):
        return HandTracker.landmark_distance(landmarks, WRIST, PINKY_MCP)

    def _pair_hands(self, hands):
        """Return (left_hand, right_hand) by screen x-position of finger bases."""
        if len(hands) < 2:
            return None, None

        ranked = []
        for hand in hands:
            base = self._average_finger_base(hand.landmarks)
            if base is not None:
                ranked.append((base[0], hand))

        if len(ranked) < 2:
            return None, None

        ranked.sort(key=lambda item: item[0])
        return ranked[0][1], ranked[1][1]

    def _shared_plane_dir(self, left_dir, right_dir):
        """
        Compute the average finger direction (for drawing the plane itself).
        The parallel planes should be parallel to this average direction.
        """
        if left_dir is None or right_dir is None:
            return None
        # Average the unit directions, re-normalize to get a unit vector along the plane.
        avg = (left_dir[0] + right_dir[0], left_dir[1] + right_dir[1])
        avg_dir = self._normalize(avg)
        return avg_dir

    def _shared_plane_normal(self, left_dir, right_dir):
        if left_dir is None or right_dir is None:
            return None
        # The normal is perpendicular to the *average* finger dir
        avg = (left_dir[0] + right_dir[0], left_dir[1] + right_dir[1])
        avg_dir = self._normalize(avg)
        if avg_dir is None:
            return None
        # Perpendicular vector (normal)
        return (-avg_dir[1], avg_dir[0])

    def _plane_distance(self, left_base, right_base, normal):
        """Perpendicular distance between parallel planes through each hand base."""
        d_left = normal[0] * left_base[0] + normal[1] * left_base[1]
        d_right = normal[0] * right_base[0] + normal[1] * right_base[1]
        return abs(d_left - d_right)

    def _approach_velocity_ok(self, prev_pos, curr_pos, finger_dir, toward, dt, hand_scale):
        if prev_pos is None or dt <= 0 or finger_dir is None:
            return False

        vx = (curr_pos[0] - prev_pos[0]) / dt
        vy = (curr_pos[1] - prev_pos[1]) / dt

        fx, fy = finger_dir
        dot_vf = vx * fx + vy * fy
        px = vx - dot_vf * fx
        py = vy - dot_vf * fy

        toward_speed = px * toward[0] + py * toward[1]
        threshold = hand_scale * self.clap_ratio / dt
        return toward_speed > threshold

    def _reset_tracking(self):
        self._prev_base = {"left": None, "right": None}
        self._approach_time = 0.0
        self._confirm_time = 0.0
        self._approach_both_false_time = 0.0

    def update(self, hands, timestamp):
        """Return True once when a clap matching the full procedure is detected."""
        if self._prev_time is None:
            self._prev_time = timestamp

        dt = timestamp - self._prev_time
        self._prev_time = timestamp
        if dt <= 0:
            return False

        left_hand, right_hand = self._pair_hands(hands)
        if left_hand is None or right_hand is None:
            self._reset_tracking()
            self._viz = None
            return False

        left_base = self._average_finger_base(left_hand.landmarks)
        right_base = self._average_finger_base(right_hand.landmarks)
        left_dir = self._finger_dir(left_hand.landmarks)
        right_dir = self._finger_dir(right_hand.landmarks)
        left_scale = self._hand_scale(left_hand.landmarks)
        right_scale = self._hand_scale(right_hand.landmarks)
        # Compute average finger direction (for planes)
        plane_dir = self._shared_plane_dir(left_dir, right_dir)
        normal = self._shared_plane_normal(left_dir, right_dir)

        if None in (left_base, right_base, left_dir, right_dir, left_scale, right_scale, normal, plane_dir):
            self._reset_tracking()
            self._viz = None
            return False

        left_toward = self._normalize((right_base[0] - left_base[0], right_base[1] - left_base[1]))
        right_toward = self._normalize((left_base[0] - right_base[0], left_base[1] - right_base[1]))
        if left_toward is None or right_toward is None:
            self._reset_tracking()
            self._viz = None
            return False

        left_ok = self._approach_velocity_ok(
            self._prev_base["left"], left_base, left_dir, left_toward, dt, left_scale
        )
        right_ok = self._approach_velocity_ok(
            self._prev_base["right"], right_base, right_dir, right_toward, dt, right_scale
        )

        # Debouncer logic for approach_time reset
        if left_ok or right_ok:
            self._approach_time += dt
            self._approach_both_false_time = 0.0
        else:
            self._approach_both_false_time += dt
            # Only reset approach_time if both have been false for at least debounce interval
            if self._approach_both_false_time >= self.APPROACH_RESET_DEBOUNCE:
                self._approach_time = 0.0

        avg_scale = (left_scale + right_scale) / 2
        confirm_threshold = avg_scale * self.confirm_clap_ratio
        plane_dist = self._plane_distance(left_base, right_base, normal)

        if plane_dist < confirm_threshold:
            self._confirm_time += dt
        else:
            self._confirm_time = 0.0

        self._prev_base["left"] = left_base
        self._prev_base["right"] = right_base

        self._viz = {
            "left_base": left_base,
            "right_base": right_base,
            "normal": normal,
            "plane_dir": plane_dir,
            "left_dir": left_dir,
            "right_dir": right_dir,
            "plane_dist": plane_dist,
            "confirm_threshold": confirm_threshold,
            "left_ok": left_ok,
            "right_ok": right_ok,
        }

        clap = False
        if (
            self._approach_time >= self.MIN_APPROACH_TIME
            and self._confirm_time > 0
            and self._confirm_time <= self.MAX_CONFIRM_TIME
            and timestamp - self._last_clap_time >= self.cooldown
        ):
            clap = True
            self._last_clap_time = timestamp
            self._clap_display_until = timestamp + 0.4
            self._reset_tracking()

        if self._confirm_time > self.MAX_CONFIRM_TIME:
            self._confirm_time = 0.0

        return clap

    def draw(self, frame, timestamp):
        """Draw parallel planes and clap feedback."""
        if self._viz:
            viz = self._viz
            left_base = viz["left_base"]
            right_base = viz["right_base"]
            normal = viz["normal"]
            plane_dir = viz["plane_dir"]
            plane_dist = viz["plane_dist"]
            confirm_threshold = viz["confirm_threshold"]
            close_enough = plane_dist < confirm_threshold

            # cv2.putText(frame, f"Left OK: {viz['left_ok']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # cv2.putText(frame, f"Right OK: {viz['right_ok']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            left_line = self._plane_line_endpoints(left_base, plane_dir)
            right_line = self._plane_line_endpoints(right_base, plane_dir)
            gap_end = self._normal_gap_end(left_base, right_base, normal)

            left_color = (255, 128, 0)
            right_color = (0, 128, 255)
            gap_color = (0, 255, 0) if close_enough else (0, 165, 255)

            # cv2.line(frame, left_line[0], left_line[1], left_color, 2)
            # cv2.line(frame, right_line[0], right_line[1], right_color, 2)
            # cv2.circle(frame, (int(left_base[0]), int(left_base[1])), 6, left_color, cv2.FILLED)
            # cv2.circle(frame, (int(right_base[0]), int(right_base[1])), 6, right_color, cv2.FILLED)
            # cv2.line(frame, (int(left_base[0]), int(left_base[1])), gap_end, gap_color, 2)
            # cv2.arrowedLine(
            #     frame,
            #     (int(left_base[0]), int(left_base[1])),
            #     (
            #         int(left_base[0] + normal[0] * 40),
            #         int(left_base[1] + normal[1] * 40),
            #     ),
            #     (200, 200, 200),
            #     2,
            #     tipLength=0.25,
            # )

            # for base, direction, color in (
            #     (left_base, viz["left_dir"], left_color),
            #     (right_base, viz["right_dir"], right_color),
            # ):
            #     cv2.arrowedLine(
            #         frame,
            #         (int(base[0]), int(base[1])),
            #         (
            #             int(base[0] + direction[0] * 50),
            #             int(base[1] + direction[1] * 50),
            #         ),
            #         color,
            #         2,
            #         tipLength=0.2,
            #     )

            # cv2.putText(
            #     frame,
            #     f"Plane gap: {plane_dist:.0f}px (need < {confirm_threshold:.0f})",
            #     (10, 140),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.55,
            #     gap_color,
            #     2,
            # )

        if timestamp < self._clap_display_until:
            cv2.putText(
                frame,
                "Clap!",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
