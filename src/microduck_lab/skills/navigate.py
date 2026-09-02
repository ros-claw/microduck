"""Closed-loop walk_to navigation skill (drives the walking policy's twist cmd)."""

from __future__ import annotations

import math

import numpy as np

from ..sim.runtime import DuckRuntime


class WalkTo:
    """P-controller over the velocity policy: turn to face, walk, stop."""

    def __init__(self, duck: DuckRuntime, walk_policy: str = "walk",
                 stand_policy: str = "stand"):
        self.duck = duck
        self.walk_policy = walk_policy
        self.stand_policy = stand_policy
        self.target = None
        self.arrive_tol = 0.03
        self.done = False

    def go_to(self, x: float, y: float):
        self.target = np.array([x, y])
        self.done = False
        self.duck.active_policy = self.walk_policy

    def update(self):
        d = self.duck
        if self.target is None or self.done:
            return
        pos = d.trunk_pos()[:2]
        yaw = d.trunk_yaw()
        delta = self.target - pos
        dist = float(np.linalg.norm(delta))
        if dist < self.arrive_tol:
            self.done = True
            d.set_command(twist=(0, 0, 0))
            d.active_policy = self.stand_policy
            return
        bearing = math.atan2(delta[1], delta[0]) - yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        ang = float(np.clip(2.0 * bearing, -1.2, 1.2))
        # walking policy: actual speed ≈ 0.45 × commanded — command hot
        vx = 0.30 if abs(bearing) < 0.5 else 0.0
        if dist < 0.25:
            vx = min(vx, 0.15)
        # lateral assist when close and roughly aligned
        vy = 0.0
        if dist < 0.3 and abs(bearing) < 1.2:
            vy = float(np.clip(1.0 * (delta[0] * -math.sin(yaw) + delta[1] * math.cos(yaw)),
                               -0.15, 0.15))
        d.set_command(twist=(vx, vy, ang))


class TurnTo:
    """Rotate in place to a target yaw via the walking policy's angular cmd."""

    def __init__(self, duck: DuckRuntime):
        self.duck = duck
        self.target_yaw = None
        self.done = True

    def turn_to(self, yaw: float):
        self.target_yaw = yaw
        self.done = False
        self.duck.active_policy = "walk"

    def update(self):
        if self.done or self.target_yaw is None:
            return
        err = (self.target_yaw - self.duck.trunk_yaw() + math.pi) % (2 * math.pi) - math.pi
        if abs(err) < 0.08:
            self.done = True
            self.duck.set_command(twist=(0, 0, 0))
            self.duck.active_policy = "stand"
            return
        self.duck.set_command(twist=(0.0, 0.0, float(np.clip(2.2 * err, -1.2, 1.2))))
