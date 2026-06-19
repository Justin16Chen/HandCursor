"""Mouse control with PyAutoGUI and One Euro Filter smoothing."""

import math

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # default 0.1s pause would cap the loop at ~10 FPS while pinching


class OneEuroFilter:
    """Adaptive low-pass filter: more smoothing when slow, less when fast."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _low_pass(self, alpha, x, x_prev):
        return alpha * x + (1.0 - alpha) * x_prev

    def filter(self, x, timestamp):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = timestamp
            return x

        dt = timestamp - self.t_prev
        if dt <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = self._low_pass(alpha_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = self._low_pass(alpha, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        return x_hat


class MouseController:
    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        min_cutoff=1.0,
        beta=0.007,
    ):
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._controlling = False

    def _map_to_screen(self, x, y):
        screen_x = x / self.frame_width * self.screen_width
        screen_y = y / self.frame_height * self.screen_height
        return screen_x, screen_y

    def update(self, x, y, allow_control, timestamp):
        """Move the cursor while pinching; reset smoothing when pinch ends."""
        if not allow_control:
            if self._controlling:
                self.filter_x.reset()
                self.filter_y.reset()
            self._controlling = False
            return

        if not self._controlling:
            self.filter_x.reset()
            self.filter_y.reset()

        screen_x, screen_y = self._map_to_screen(x, y)
        smooth_x = self.filter_x.filter(screen_x, timestamp)
        smooth_y = self.filter_y.filter(screen_y, timestamp)
        pyautogui.moveTo(int(smooth_x), int(smooth_y))
        self._controlling = True

    def press_left(self):
        """Simulate pressing and holding the left mouse button."""
        pyautogui.mouseDown()

    def release_left(self):
        """Simulate releasing the left mouse button."""
        pyautogui.mouseUp()
   