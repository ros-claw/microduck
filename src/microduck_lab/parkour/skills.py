"""State-based skill completion at 50 Hz. Timeouts only report failure."""

from dataclasses import dataclass, field
import math
import mujoco
import numpy as np
from .world import foot_support


def lane_command(r, target_y, vx=0.7):
    """Anticipate lateral momentum rather than steering after overshooting."""
    error = target_y - r.trunk_pos()[1]
    target_yaw = float(np.clip(8 * error - 2 * r.trunk_linvel()[1], -0.9, 0.9))
    return (vx, 0.0, float(np.clip(5 * (target_yaw - r.trunk_yaw()), -3.0, 3.0)))


def roll_entry_ready(m, d, r, bar_x):
    """Enter during a measured left stance/upward phase; never alter root state."""
    distance = bar_x - float(r.trunk_pos()[0])
    if not (
        0.24 < distance < 0.50
        and 0.1 < r.trunk_linvel()[2] < 0.3
        and d.xmat[r.trunk_body_id, 8] > 0.95
    ):
        return False
    force = np.zeros(6)
    for ci, contact in enumerate(d.contact):
        names = [m.geom(g).name for g in (contact.geom1, contact.geom2)]
        if "duck/left_foot_collision" in names and any(
            n.startswith("floor/") for n in names
        ):
            mujoco.mj_contactForce(m, d, ci, force)
            if force[0] > 1e-5:
                return True
    return False


@dataclass
class Maneuver:
    name: str
    started: float
    target_y: float = 0.0
    forward_speed: float = 0.7
    phase: str = "PREPARE"
    status: str = "RUNNING"
    stable: float = 0.0
    rotation: float = 0.0
    last_pitch: float | None = None
    inverted: bool = False
    seen_fall: bool = False
    events: list = field(default_factory=list)

    def update(self, m, d, r, dt=0.02):
        if self.status != "RUNNING":
            return self.status
        up = float(d.xmat[r.trunk_body_id, 8])
        speed = float(np.linalg.norm(r.trunk_linvel()[:2]))
        support = foot_support(m, d)
        r.command[:] = 0
        if self.name not in ("TAKE_LEFT_ROUTE", "TAKE_RIGHT_ROUTE", "RUN"):
            r.bank.mirror_run = False
        if self.name in ("TAKE_LEFT_ROUTE", "TAKE_RIGHT_ROUTE", "RUN"):
            if self.phase == "PREPARE":
                r.bank.mirror_run = self.name == "TAKE_RIGHT_ROUTE"
            r.active_policy = "run"
            r.command[:3] = lane_command(r, self.target_y, self.forward_speed)
            error = abs(r.trunk_pos()[1] - self.target_y)
            self.phase = "CROSS" if error > 0.025 else "ALIGN"
            good = (
                error < 0.025
                and abs(r.trunk_yaw()) < 0.15
                and up > 0.95
                and r.trunk_linvel()[0] > 0.15
                and support
            )
            timeout = 3.0
        elif self.name == "BRAKE_AND_WAIT":
            r.active_policy = "run"
            yaw = r.trunk_yaw()
            r.command[2] = (
                0.0
                if abs(yaw) < 0.04
                else -np.sign(yaw) * np.clip(3 * abs(yaw), 0.55, 1.5)
            )
            good = speed < 0.03 and abs(r.trunk_yaw()) < 0.07 and up > 0.95 and support
            timeout = 2.0
        elif self.name == "ROLL_CENTER":
            R = d.xmat[r.trunk_body_id].reshape(3, 3)
            pitch = math.atan2(-R[2, 0], R[0, 0])
            if self.last_pitch is not None:
                self.rotation += math.atan2(
                    math.sin(pitch - self.last_pitch), math.cos(pitch - self.last_pitch)
                )
            self.last_pitch = pitch
            self.inverted |= up < -0.8
            self.phase = "CROSS" if abs(self.rotation) < 5.7 else "ALIGN"
            r.active_policy = "roulade" if abs(self.rotation) < 5.7 else "stand"
            if abs(self.rotation) > 5.7 and up > 0.95 and r.trunk_pos()[2] > 0.10:
                # Keep one controller through alignment; toggling stand/run at
                # the success yaw threshold can repeatedly break the dwell.
                r.active_policy = "run"
                yaw = r.trunk_yaw()
                r.command[2] = (
                    0.0
                    if abs(yaw) < 0.04
                    else -np.sign(yaw) * np.clip(3 * abs(yaw), 0.55, 1.5)
                )
            good = (
                abs(self.rotation) > 5.7
                and self.inverted
                and up > 0.95
                and support
                and r.trunk_pos()[2] > 0.10
                and abs(r.trunk_yaw()) < 0.15
                and speed < 0.30
            )
            timeout = 3.0
        elif self.name == "RECOVER":
            self.seen_fall |= up < 0.5
            r.active_policy = "stand"
            self.phase = "ALIGN"
            good = (
                up > 0.95
                and support
                and r.trunk_pos()[2] > 0.10
                and np.linalg.norm(r.trunk_linvel()) < 0.25
            )
            timeout = 2.0
        elif self.name == "JUMP_CENTER":
            raise ValueError(
                "JUMP_CENTER requires CandidateJump and its full-rate gap audit"
            )
        else:
            raise ValueError(self.name)
        self.stable = self.stable + dt if good else 0.0
        if good:
            self.phase = "STABLE"
        if d.time - self.started > timeout + 1e-8:
            self.status = "FAILED"
            self.phase = "TIMEOUT"
        elif self.stable >= 0.12:
            self.status = "SUCCESS"
            self.phase = "SUCCESS"
        if self.status != "RUNNING":
            self.events.append(
                dict(
                    type="SKILL_RESULT",
                    t=float(d.time),
                    skill=self.name,
                    status=self.status,
                    rotation=self.rotation,
                )
            )
        return self.status
