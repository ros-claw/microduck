"""Measured hazard snapshots and conservative, time-indexed reachability.

No generator seed or future actuator targets are accepted by this interface.
Constant-velocity forecasts are estimates and grow an uncertainty envelope.
"""

from dataclasses import dataclass
import math
import numpy as np
import mujoco


@dataclass(frozen=True)
class HazardState:
    name: str
    kind: str
    position: tuple
    velocity: tuple
    half_size: tuple
    angle: float = 0.0
    angular_velocity: float = 0.0
    half_span: float = 0.0
    radius: float = 0.0
    warning_remaining: float | None = None
    held: bool = False

    def bounds(self, t):
        center = np.asarray(self.position, dtype=float) + t * np.asarray(self.velocity)
        extent = np.asarray(self.half_size, dtype=float).copy()
        if self.kind == "sweeper":
            angle = self.angle + self.angular_velocity * t
            extent = (
                np.abs([math.cos(angle), math.sin(angle), 0.0]) * self.half_span
                + self.radius
            )
        if self.kind == "crate":
            fall_t = max(0.0, t - (self.warning_remaining or 0.0)) if self.held else t
            if not self.held or self.warning_remaining is not None:
                center[2] -= 0.5 * 9.81 * fall_t**2
                center[2] = max(extent[2], center[2])
        # Model uncertainty grows with forecast horizon; it is not a guarantee.
        extent += 0.003 + 0.005 * t * t
        return center - extent, center + extent

    def crossing_eta(self, lane_y, robot_half_width=0.065, horizon=4.0):
        for t in np.arange(0, horizon + 0.02, 0.02):
            low, high = self.bounds(float(t))
            if low[1] - robot_half_width <= lane_y <= high[1] + robot_half_width:
                return round(float(t), 2)
        return None


def observe_hazards(m, d, hazards, driver, lanes, duck_x, duck_vx):
    states = []
    public = []
    for h in hazards:
        bid = m.body(h.name).id
        gid = m.geom(h.name + "/geom").id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, velocity, 0)
        center = d.geom_xpos[gid].copy()
        size = m.geom_size[gid].copy()
        angle = omega = half_span = radius = 0.0
        if h.kind == "sweeper":
            axis = d.geom_xmat[gid].reshape(3, 3)[:, 2]
            angle = math.atan2(axis[1], axis[0])
            omega = float(velocity[2])
            radius = float(size[0])
            half_span = float(size[1])
            size = np.array([half_span + radius] * 2 + [radius])
        elif h.kind == "boulder":
            radius = float(size[0])
            size[:] = radius
        elif h.kind == "push_bar":
            size = np.array([0.23, h.half_width + 0.009, 0.23])
        held = h.kind == "crate" and h.name not in driver.released
        remaining = (
            max(0.0, 0.85 - (d.time - driver.warnings[h.name]))
            if held and h.name in driver.warnings
            else None
        )
        state = HazardState(
            h.name,
            h.kind,
            tuple(center),
            tuple(velocity[3:]),
            tuple(size),
            angle,
            omega,
            half_span,
            radius,
            remaining,
            held,
        )
        states.append(state)
        p = dict(
            name=h.name,
            kind=h.kind,
            position=center.tolist(),
            velocity=velocity[3:].tolist(),
            angle=angle,
            angular_velocity=omega,
            warning_remaining=remaining,
            held=held,
            crossing_eta_by_lane=[state.crossing_eta(y) for y in lanes],
        )
        if h.kind == "sweeper":
            axis = np.array([math.cos(angle), math.sin(angle), 0.0]) * half_span
            p["endpoints"] = [(center - axis).tolist(), (center + axis).tolist()]
        if h.kind == "boulder":
            distance = duck_x - center[0] - radius
            closing = float(velocity[3] - duck_vx)
            p.update(
                distance=float(distance),
                closing_speed=closing,
                ttc=max(0.0, float(distance / closing)) if closing > 0 else None,
            )
        public.append(p)
    return states, public


def conflicts(path, states, time_offset=0.0):
    """Path rows: (relative time, low xyz, high xyz) from a measured skill envelope.

    Padding covers intersample motion under each current velocity estimate.
    Callers must cover the WHOLE skill, not just its endpoint. Unvalidated skill
    profiles should never be supplied as executable candidates.
    """
    if len(path) < 2:
        raise ValueError("At least two time-indexed envelope samples required")
    times = [row[0] for row in path]
    if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Path must start now and advance monotonically")
    dt = max(np.diff(times))
    hits = set()
    for t, low, high in path:
        low = np.asarray(low)
        high = np.asarray(high)
        if np.any(low > high):
            raise ValueError("Invalid envelope")
        for state in states:
            a, b = state.bounds(t + time_offset)
            padding = (
                (
                    np.linalg.norm(state.velocity)
                    + abs(state.angular_velocity) * state.half_span
                )
                * dt
                / 2
            )
            if np.all(high >= a - padding) and np.all(low <= b + padding):
                hits.add(state.name)
    return sorted(hits)
