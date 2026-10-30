"""Intent-free closing-speed estimate from observed relative positions."""

from collections import deque
import math
import numpy as np


class OpponentMotionEstimator:
    def __init__(self, window_s=0.12):
        if not 0.06 <= window_s <= 0.24:
            raise ValueError("Bounded public motion history required")
        self.window_s = window_s
        self.history = deque()
        self.target = None

    def update(self, now, me, other):
        if other.name != self.target:
            self.history.clear()
            self.target = other.name
        distance = float(np.linalg.norm(np.array(other.xy) - me.xy))
        self.history.append((now, distance))
        while (
            len(self.history) > 2 and now - self.history[1][0] >= self.window_s - 1e-9
        ):
            self.history.popleft()
        elapsed = now - self.history[0][0]
        if elapsed < self.window_s * 0.75:
            return 0.0
        return (self.history[0][1] - distance) / max(elapsed, 1e-9)


def facing_twist(distance, angle):
    """Step around a visible target while turning, instead of a wide U-turn.

    Opposite orbital bearing motion closes the relative facing angle. All
    terms are native bounded walk commands; there is no pose interpolation.
    """
    turn = float(np.clip(2.8 * angle, -1.0, 1.0))
    if abs(angle) > 0.45 and distance < 0.65:
        side = 1 if angle > 0 else -1
        tangent = side * 0.3 * np.array([-math.sin(angle), math.cos(angle)])
        radial = np.clip(1.2 * (distance - 0.36), -0.1, 0.1) * np.array(
            [math.cos(angle), math.sin(angle)]
        )
        velocity = np.clip(tangent + radial, -0.3, 0.3)
        return (*velocity, turn)
    return (0.28 if distance > 0.29 or abs(angle) > 0.22 else 0.0, 0.0, turn)
