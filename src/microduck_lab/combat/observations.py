"""Immutable public body snapshots; no opponent controller objects or seed."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Body:
    name: str
    xy: tuple[float, float]
    z: float
    velocity: tuple[float, float]
    yaw: float
    upright: float
    foot_support: bool
    body_ground: bool
    boundary_distance: float
    touching_opponent: bool = False

    @property
    def speed(self):
        return math.hypot(*self.velocity)


def snapshot(ducks, audit, config):
    result = {}
    for name, duck in ducks.items():
        p = duck.trunk_pos()
        distance = (
            config.radius_m - np.linalg.norm(p[:2])
            if config.arena == "fixed_disc"
            else config.radius_m - max(abs(p[0]), abs(p[1]))
        )
        result[name] = Body(
            name,
            tuple(p[:2]),
            float(p[2]),
            tuple(duck.trunk_linvel()[:2]),
            duck.trunk_yaw(),
            float(-duck.projected_gravity()[2]),
            audit.support[name],
            audit.body_ground[name],
            float(distance),
            audit.touching[name],
        )
    return result


def relative(me, other):
    delta = np.array(other.xy) - me.xy
    dist = float(np.linalg.norm(delta))
    vel = np.array(other.velocity) - me.velocity
    closing = -float(np.dot(delta, vel)) / max(dist, 1e-9)
    angle = math.atan2(delta[1], delta[0]) - me.yaw
    return delta, dist, math.atan2(math.sin(angle), math.cos(angle)), closing
