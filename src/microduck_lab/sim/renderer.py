"""Offscreen cinematic renderer for duck worlds."""

from __future__ import annotations

import mujoco
import numpy as np


class CinematicRenderer:
    def __init__(self, model, width=960, height=540):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.set_cam("side")

    def set_cam(self, preset: str, target=None):
        c = self.cam
        if preset == "side":            # rope plane view (watch the jump)
            c.lookat = [0, 0, 0.10]; c.distance = 1.05; c.azimuth = 90; c.elevation = -10
        elif preset == "hero34":        # 3/4 hero shot
            c.lookat = [0, 0, 0.10]; c.distance = 1.25; c.azimuth = 125; c.elevation = -18
        elif preset == "top":
            c.lookat = [0, 0, 0.0]; c.distance = 1.5; c.azimuth = 90; c.elevation = -80
        elif preset == "low":           # dramatic low angle
            c.lookat = [0, 0, 0.14]; c.distance = 0.9; c.azimuth = 60; c.elevation = -4
        elif preset == "wide":
            c.lookat = [0, 0, 0.1]; c.distance = 1.8; c.azimuth = 100; c.elevation = -22
        if target is not None:
            c.lookat = list(target)

    def track(self, pos, distance=0.9, azimuth=115, elevation=-12):
        self.cam.lookat = [float(pos[0]), float(pos[1]), float(pos[2]) + 0.05]
        self.cam.distance = distance
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation

    def render(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.cam)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()
