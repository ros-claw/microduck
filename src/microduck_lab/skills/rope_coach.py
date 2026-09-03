"""Rope turning team skill: two ducks hold + swing the rope, with coach wind-up.

Physics findings that shaped this design (all measured, see docs/physics-notes.md):
- The XL330 neck servos physically cannot fling the 0.3 kg head vertically fast
  enough to spin a rope up from hanging (τ ≈ 0.8 Nm needed vs 0.96 Nm limit).
- A duck jaw latched to a rope is an energy sink: the servo resists rope-induced
  head motion, dissipating free rotation within ~1 s.
- The trained `stand` policy tolerates vigorous vertical body-bob but is
  toppled by sustained lateral head sway ≥ ~1 cm at ~1 Hz.

So the turner ducks really drive their mouths along the measured rope phase
(they contribute real, if small, pumping power and are load-bearing), while a
declared "coach assist" torque field on the rope (a simulation training aid —
like an adult giving the rope a steadying push for kid skippers) covers the
energy deficit. All practice records flag `coach_assist=True`.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..sim.runtime import DuckRuntime


class CoachRopeDriver:
    """Rope coach: spin-up (wind-up + toss) and two drive modes.

    mode="rotate": continuous tangential drive (full overhead hoop).
    mode="swing":  oscillating tangential drive at the pendulum resonance —
                   the belly sweeps the floor back and forth ("side-swing"
                   skipping, a real playground game). Far more robust at this
                   scale: no ω²R > g requirement, belly grazes the floor
                   deterministically each pass.
    mode="snake":  bang-bang lateral push along the floor ("snake in the
                   grass") — the belly slides left-right through the jumper's
                   position. The most robust: gravity is fully floor-supported.
    """

    def __init__(self, world, frequency: float = 1.3, gain: float = 0.25,
                 rate_kp: float = 4.0, mode: str = "rotate"):
        self.world = world
        self.frequency = frequency
        self.gain = gain
        self.rate_kp = rate_kp
        self.mode = mode
        self.enabled = False
        self._seg_masses = None
        self._rate_lp = 0.0
        self._prev_angle = None
        self._swing_phase = 0.0

    def start(self):
        m = self.world.model
        self._seg_masses = np.array([m.body_mass[b] for b in self.world.rope_body_ids])
        self.enabled = True
        self._rate_lp = 0.0
        self._prev_angle = None

    def stop(self):
        self.enabled = False
        self.world.data.xfrc_applied[:] = 0

    def toss(self, spin_mult: float = 1.6):
        """Give the rope an initial rigid-rotation impulse (the wind-up)."""
        w = self.world
        m, d = w.model, w.data
        p0 = d.xpos[w.rope_body_ids[0]]
        pN = d.xpos[w.rope_body_ids[-1]]
        c = (p0 + pN) / 2
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rope/rope_free")
        adr = int(m.jnt_dofadr[jid])
        om = 2 * math.pi * self.frequency * spin_mult
        d.qvel[adr:adr + 3] = np.cross([om, 0, 0], p0 - c)
        d.qvel[adr + 3:adr + 6] = [om, 0, 0]

    def windup_lift(self, strength: float = 9.0):
        """One step of 'the coach lifts the rope up': raise the belly off the
        floor before the toss, so the first revolution doesn't plow the floor."""
        w = self.world
        d = w.data
        p0 = d.xpos[w.rope_body_ids[0]]
        pN = d.xpos[w.rope_body_ids[-1]]
        c = (p0 + pN) / 2
        for i, b in enumerate(w.rope_body_ids):
            r = d.xpos[b] - c
            # lift + initial tangential shove in the rotation direction
            tang = np.cross([1.0, 0, 0], r)
            n = np.linalg.norm(tang)
            f = np.array([0.0, 0.0, strength]) + (2.0 * tang / n if n > 1e-6 else 0.0)
            d.xfrc_applied[b, :3] = self._seg_masses[i] * f

    def update(self, dt: float = 0.02):
        if not self.enabled:
            return
        w = self.world
        d = w.data
        omega = 2 * math.pi * self.frequency
        axis = np.array([1.0, 0, 0])
        p0 = d.xpos[w.rope_body_ids[0]]
        pN = d.xpos[w.rope_body_ids[-1]]
        c = (p0 + pN) / 2
        # measured belly angular rate (low-passed)
        ang, amp = self.belly_phase()
        if self._prev_angle is not None:
            rate = ((ang - self._prev_angle + math.pi) % (2 * math.pi) - math.pi) / dt
            self._rate_lp = 0.9 * self._rate_lp + 0.1 * rate
        self._prev_angle = ang

        # Coil guard: if the rope has wound up around the mouths (belly no
        # longer sweeps near the floor), driving harder only tightens the coil
        # — ease off completely and let gravity unwind it.
        # Grace period: the wind-up + spin-up legitimately holds the rope high
        # for the first seconds; only arm the guard once spinning is possible.
        self._spin_time = getattr(self, "_spin_time", 0.0) + dt
        zmin = float(np.min([d.xpos[b][2] for b in w.rope_body_ids]))
        if zmin > 0.12 and self._spin_time > 3.0:
            self._coil_hold = getattr(self, "_coil_hold", 0.0) + dt
        else:
            self._coil_hold = 0.0
        if self._coil_hold > 2.0:
            d.xfrc_applied[:] = 0.0
            return

        if self.mode == "snake":
            # push the belly side to side along the floor, WITH its motion
            belly = self.belly_y_rate(dt)
            push = 1.0 if belly >= 0 else -1.0
            for i, b in enumerate(w.rope_body_ids):
                d.xfrc_applied[b, 1] = self._seg_masses[i] * self.gain * omega * push * 2.0
            return
        if self.mode == "swing":
            # swing pump: always push tangentially WITH the belly's motion
            # (bang-bang, like pumping a playground swing), ease off when the
            # arc is already big enough (governor on the belly radius/angle).
            push = 1.0 if self._rate_lp >= 0 else -1.0
            govern = 1.0 if amp < 0.20 else 0.25
            for i, b in enumerate(w.rope_body_ids):
                r = d.xpos[b] - c
                tangential = np.cross(axis, r)
                n = np.linalg.norm(tangential)
                if n < 1e-6:
                    continue
                d.xfrc_applied[b, :3] = (self._seg_masses[i] * self.gain * omega
                                         * push * govern * tangential / n)
            return
        rate_err = omega - self._rate_lp
        # loop-height governor: the rope must keep grazing the floor at the
        # bottom. If the bottom climbs (loop inflating/migrating up), ease the
        # drive off; if it scrapes low, keep driving. Measured on the belly.
        zmin = float(np.min([d.xpos[b][2] for b in w.rope_body_ids]))
        self._bottom_lp = (0.9 * getattr(self, "_bottom_lp", 0.0) + 0.1 * zmin)
        # healthy rotation dips the belly low each pass; the low-pass bottom
        # rising means the loop is climbing
        self._drive_scale = float(np.clip(1.0 - 6.0 * (self._bottom_lp - 0.02), 0.3, 1.0))
        for i, b in enumerate(w.rope_body_ids):
            r = d.xpos[b] - c
            tangential = np.cross(axis, r)
            n = np.linalg.norm(tangential)
            if n < 1e-6:
                continue
            # tangential: keep the spin at the target rate (governor)
            a_t = (self.gain * omega + self.rate_kp * rate_err) * self._drive_scale
            d.xfrc_applied[b, :3] = self._seg_masses[i] * a_t * tangential / n

    # ------------------------------------------------------------- sensing
    @property
    def spin_rate(self) -> float:
        return self._rate_lp

    def belly_y_rate(self, dt: float = 0.02) -> float:
        """Low-passed lateral velocity of the rope belly (snake mode)."""
        w = self.world
        pts = np.array([w.data.xpos[b] for b in w.rope_body_ids])
        y = float(pts[len(pts) // 2][1])
        if not hasattr(self, "_prev_belly_y"):
            self._prev_belly_y = y
            self._y_rate_lp = 0.0
            return 0.0
        rate = (y - self._prev_belly_y) / dt
        self._prev_belly_y = y
        self._y_rate_lp = 0.9 * self._y_rate_lp + 0.1 * rate
        return self._y_rate_lp

    def belly_phase(self, at_x: float | None = None) -> tuple[float, float]:
        """Sensor-world rope observation: (belly angle, belly radius).

        Angle 0 = belly straight down. Measured from the rope section nearest
        `at_x` (the jumper's position) when given — the rope is floppy, so the
        section near the jumper has its own phase lag vs the midpoint.
        """
        w = self.world
        d = w.data
        pts = np.array([d.xpos[b] for b in w.rope_body_ids])
        c = pts[[0, -1]].mean(axis=0)
        if at_x is None:
            mid = pts[len(pts) // 2]
        else:
            k = int(np.argmin(np.abs(pts[:, 0] - at_x)))
            mid = pts[max(2, min(len(pts) - 3, k))]
        dy, dz = mid[1] - c[1], mid[2] - c[2]
        return math.atan2(dy, -dz), math.hypot(dy, dz)

    def belly_bottom_crossing_eta(self, dt: float, prev_angle: float, angle: float) -> float | None:
        """Estimated time until the belly reaches bottom dead center."""
        dth = (angle - prev_angle + math.pi) % (2 * math.pi) - math.pi
        rate = dth / dt
        if rate < 0.3:      # too slow / stalled
            return None
        # distance to bottom (angle 0) along increasing angle
        dist = (-angle) % (2 * math.pi)
        return dist / rate
