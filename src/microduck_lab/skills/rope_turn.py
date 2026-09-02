"""Closed-loop rope turning: two ducks swing the rope with their mouths.

The mouth (mouth_tip site on jaw_soft) is driven along a small circle in the
plane perpendicular to the rope axis, using damped least-squares IK over the
4 neck/head joints. The leg policy (stand) keeps the body upright — the head
is ~38% of body mass, so swinging is a real balance disturbance, not a
kinematic animation. Both turners share one phase clock (the "team rhythm").

Stability lessons (learned from practice — see Practice records):
- ramp frequency AND radius in over ~2.5 s, or the turners get yanked over;
- rope must spin fast enough that centrifugal force beats gravity at the
  belly: f > sqrt(g / R_loop) / 2π ≈ 1.1 Hz for a 0.2 m loop;
- rate-limit the IK correction — a 7 rad/s head is a flail, not a turn.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..sim.runtime import DuckRuntime, HEAD_IDX


class RopeTurner:
    def __init__(self, duck: DuckRuntime, circle_radius: float = 0.025,
                 radius_z: float | None = None):
        self.duck = duck
        self.radius = circle_radius
        # vertical amplitude must stay small: the XL330 neck/head pitch
        # servos physically can't fling the 0.3 kg head vertically fast
        # (τ = Iα ≈ 0.8 Nm at ±25 mm / 1.25 Hz vs 0.96 Nm limit).
        self.radius_z = radius_z if radius_z is not None else circle_radius * 0.5
        self.enabled = False
        self._site_id = None
        self._head_dofs = None
        self._head_qpos = None
        self._center = None

    def _resolve(self):
        m, d = self.duck.model, self.duck.data
        self._site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, f"{self.duck.prefix}mouth_tip")
        assert self._site_id >= 0
        self._head_dofs = np.array(
            [int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in self.duck.act_ids[HEAD_IDX]])
        self._head_qpos = np.array(
            [int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in self.duck.act_ids[HEAD_IDX]])

    def start(self):
        self._resolve()
        rest = self.duck.data.site_xpos[self._site_id].copy()
        self._center = rest + np.array([0.0, 0.0, 0.015])
        self.enabled = True

    def stop(self):
        self.enabled = False
        self.duck.head_override = None

    def circle_point(self, phase: float, radius_scale: float = 1.0) -> np.ndarray:
        """Ellipse in the Y-Z plane (rope runs along X)."""
        ry = self.radius * radius_scale
        rz = self.radius_z * radius_scale
        return self._center + np.array([0.0, ry * math.cos(phase), rz * math.sin(phase)])

    def update(self, phase: float, radius_scale: float = 1.0):
        if not self.enabled:
            return
        d, m = self.duck.data, self.duck.model
        target = self.circle_point(phase, radius_scale)
        err = target - d.site_xpos[self._site_id]

        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, self._site_id)
        J = jacp[:, self._head_dofs]                       # 3x4
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        dq = np.clip(dq, -0.06, 0.06)                      # ≤ 3 rad/s at 50 Hz
        q = d.qpos[self._head_qpos]
        self.duck.head_override = (q + dq).astype(np.float32)


class RopeTurnPair:
    """Synchronized pair of turners sharing one phase clock, with ramp-in."""

    def __init__(self, a: RopeTurner, b: RopeTurner, frequency: float = 1.2,
                 ramp_time: float = 2.5):
        self.a, self.b = a, b
        self.frequency = frequency        # Hz (target)
        self.ramp_time = ramp_time
        self.phase = 0.0                  # rad
        self.running = False
        self._t = 0.0

    def start(self):
        self.a.start()
        self.b.start()
        self.running = True
        self.phase = 0.0
        self._t = 0.0

    def stop(self):
        self.a.stop()
        self.b.stop()
        self.running = False

    @property
    def current_frequency(self) -> float:
        s = min(1.0, self._t / self.ramp_time)
        return self.frequency * (s * s * (3 - 2 * s))    # smoothstep

    @property
    def current_radius_scale(self) -> float:
        s = min(1.0, self._t / self.ramp_time)
        return s * s * (3 - 2 * s)

    def update(self, dt: float):
        if not self.running:
            return
        self._t += dt
        self.phase = (self.phase + 2 * math.pi * self.current_frequency * dt) % (2 * math.pi)
        self.a.update(self.phase, self.current_radius_scale)
        self.b.update(self.phase, self.current_radius_scale)
