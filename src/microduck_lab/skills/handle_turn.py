"""Handle-based rope turning: the duck grips a lightweight handle in its beak
and turns the rope by rotating its head — the handle's lever arm turns a small
head ATTITUDE change into a large tip circle (measured: 0.4 rad of head pitch
or yaw moves the tip 4.4 cm). The rope end connects to the handle tip through
a point-connect (a passive swivel), so the spinning rope doesn't coil the head.

The drive is the PhaseTrackingTurner mechanism (see sim/oracle_rope.py): the
handle tip traces a circle in the y-z plane whose phase tracks the measured
rope belly angle with a +90° lead — pumping energy in the direction of motion
(swing-up), so the loop inflates from rest instead of needing a coach toss.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..sim.runtime import DuckRuntime, HEAD_IDX


class HandleRopeTurner:
    """One duck turning the rope via its mouth-held handle (phase-tracking IK)."""

    def __init__(self, duck: DuckRuntime, tip_radius: float = 0.035,
                 axis_z: float = 0.18):
        self.duck = duck
        self.tip_radius = tip_radius
        self.axis_z = axis_z
        self.enabled = False
        self._site_id = None
        self._head_dofs = None
        self._head_qpos = None
        self._center = None
        self.rope_ids = None

    def _resolve(self):
        m, d = self.duck.model, self.duck.data
        self._site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, f"{self.duck.prefix}handle_tip")
        assert self._site_id >= 0, f"no handle_tip site for {self.duck.prefix}"
        self._head_dofs = np.array(
            [int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in self.duck.act_ids[HEAD_IDX]])
        self._head_qpos = np.array(
            [int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in self.duck.act_ids[HEAD_IDX]])

    def start(self, rope_body_ids):
        self._resolve()
        rest = self.duck.data.site_xpos[self._site_id].copy()
        self._center = rest
        self.rope_ids = rope_body_ids
        self._wraps = 0
        self._prev_th = None
        self.cont = 0.0
        self.enabled = True

    def stop(self):
        self.enabled = False
        self.duck.head_override = None

    def belly_theta(self) -> float:
        """Unwrapped rope-belly angle about the x-axis at the handle height."""
        pts = np.array([self.duck.data.xpos[b] for b in self.rope_ids])
        belly = pts[np.argmin(pts[:, 2])]
        th = math.atan2(belly[1] - self._center[1], -(belly[2] - self.axis_z))
        if self._prev_th is None:
            self._prev_th = th
        dd = th - self._prev_th
        if dd > math.pi:
            self._wraps -= 1
        elif dd < -math.pi:
            self._wraps += 1
        self._prev_th = th
        self.cont = th + self._wraps * 2 * math.pi
        return self.cont

    def update(self, i_step: int, lead: float = math.pi / 2,
               spin_amp_max: float = 0.045, sustain_amp: float = 0.04):
        """Pump the rope: handle tip tracks the belly angle + lead (swing-up)."""
        if not self.enabled:
            return
        cont = self.belly_theta()
        # amplitude ramps up while spinning up, then holds. NOTE: i_step is the
        # 50 Hz policy-step index — reach full amplitude in ~2 s (100 steps).
        if abs(cont) < 2 * math.pi:
            r = min(spin_amp_max, 0.012 + i_step * (spin_amp_max - 0.012) / 100.0)
        else:
            r = sustain_amp
        phi = cont + lead
        # target tip point on a y-z circle about the rest position
        target = self._center + np.array([0.0, r * math.cos(phi), r * math.sin(phi)])
        d, m = self.duck.data, self.duck.model
        err = target - d.site_xpos[self._site_id]
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, self._site_id)
        J = jacp[:, self._head_dofs]
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        dq = np.clip(dq, -0.08, 0.08)                    # rate-limit the head
        q = d.qpos[self._head_qpos]
        self.duck.head_override = (q + dq).astype(np.float32)
