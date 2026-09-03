"""Scripted jump maneuver for Microduck — a candidate physical skill.

The official skill set has no jump policy (microduck_rl task registry has no
jump/hop task), so this is a *procedural* skill executed through the leg
actuators' position interface, layered on top of the standing policy:
crouch → explosive extension → tuck (feet up for rope clearance) → land and
hand control back to the stand policy.

Honesty note: this is not an RL policy — it is a hand-tuned open-loop maneuver
with closed-loop timing (the jump trigger is phase-locked to the rope). Its
parameters (crouch depth, extension target, phase durations) are the tunable
surface that ROSClaw Practice/Darwin adapts.
"""

from __future__ import annotations

import numpy as np

from ..sim.runtime import DuckRuntime, LEG_IDX, DEFAULT_POSE


# leg order in policy space: [L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle,
#                             R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle]
def leg_pose(hip_pitch: float, knee: float, ankle: float,
             hip_roll: float = 0.0873) -> np.ndarray:
    return np.array([
        0.0, -hip_roll, -hip_pitch, -knee, ankle,      # left
        0.0, hip_roll, hip_pitch, knee, -ankle,        # right
    ])


class JumpSkill:
    """State machine: IDLE → CROUCH → EXTEND → FLIGHT → LAND → IDLE."""

    def __init__(self, duck: DuckRuntime,
                 crouch_depth: float = 0.72,       # hip_pitch magnitude in crouch (rad)
                 extend_pitch: float = -0.12,      # hip_pitch target at toe-off (rad, negative=past straight)
                 extend_ankle: float = -0.25,      # ankle plantarflex at toe-off
                 crouch_time: float = 0.22,
                 extend_time: float = 0.10,
                 flight_time: float = 0.28,
                 land_time: float = 0.35):
        self.duck = duck
        self.crouch_depth = crouch_depth
        self.extend_pitch = extend_pitch
        self.extend_ankle = extend_ankle
        self.crouch_time = crouch_time
        self.extend_time = extend_time
        self.flight_time = flight_time
        self.land_time = land_time
        self.state = "idle"
        self._t = 0.0
        self._start_pose = None
        self.took_off = False
        self.max_feet_z = 0.0

    # ------------------------------------------------------------- queries
    def feet_z(self) -> tuple[float, float]:
        l = self.duck.site_pos("left_foot")[2]
        r = self.duck.site_pos("right_foot")[2]
        return float(l), float(r)

    @property
    def airborne(self) -> bool:
        l, r = self.feet_z()
        return min(l, r) > 0.012

    # ------------------------------------------------------------- control
    def trigger(self):
        if self.state != "idle":
            return False
        self.state = "crouch"
        self._t = 0.0
        self._start_pose = None
        self.took_off = False
        self.max_feet_z = 0.0
        return True

    def update(self, dt: float):
        if self.state == "idle":
            return
        self._t += dt
        d = self.duck
        cur = d.data.qpos[d.joint_qpos_idx][LEG_IDX].copy()

        if self.state == "crouch":
            if self._start_pose is None:
                self._start_pose = cur
            # crouch keeps feet flat: |hip_pitch| ≈ |ankle| (+small knee)
            a = self.crouch_depth
            target = leg_pose(a, 0.06, a * 0.97)
            alpha = min(1.0, self._t / self.crouch_time)
            d.leg_override = (1 - alpha) * self._start_pose + alpha * target
            if self._t >= self.crouch_time:
                self.state = "extend"
                self._t = 0.0

        elif self.state == "extend":
            # explosive extension: straighten legs + plantarflex (toe-off)
            target = leg_pose(self.extend_pitch, 0.0, self.extend_ankle)
            alpha = min(1.0, self._t / self.extend_time)
            a = self.crouch_depth
            crouch = leg_pose(a, 0.06, a * 0.97)
            d.leg_override = (1 - alpha) * crouch + alpha * target
            if self._t >= self.extend_time:
                self.state = "flight"
                self._t = 0.0

        elif self.state == "flight":
            # tuck legs up so the rope can pass underneath
            d.leg_override = leg_pose(0.55, 0.10, 0.60)
            lz, rz = self.feet_z()
            self.max_feet_z = max(self.max_feet_z, min(lz, rz))
            if self._t >= self.flight_time:
                self.state = "land"
                self._t = 0.0

        elif self.state == "land":
            # blend back to the policy's default pose, then release control.
            # If we landed badly, release IMMEDIATELY — the stand policy
            # recovers from fallen states on its own (verified: face-down and
            # face-up knock-overs recover in <0.4 s); holding the scripted pose
            # would fight the recovery.
            if not self.duck.is_upright(0.45):
                self.duck.leg_override = None
                self.state = "recover"
                self._t = 0.0
                return
            target = DEFAULT_POSE[LEG_IDX]
            alpha = min(1.0, self._t / self.land_time)
            tuck = leg_pose(0.55, 0.10, 0.60)
            d.leg_override = (1 - alpha) * tuck + alpha * target
            if self._t >= self.land_time:
                self.state = "idle"
                d.leg_override = None

        elif self.state == "recover":
            # policy-only recovery; go idle once upright (or give up at 3 s)
            if self.duck.is_upright(0.55) or self._t > 3.0:
                self.state = "idle"


class PolicyJump:
    """Jump driven by a TRAINED ONNX policy (the microduck.jump skill).

    Same contract as the official roulade/kick behavior policies: all-zero 13D
    command, session swap for `duration` seconds, hands back to `stand`.
    Drop-in replacement for the procedural JumpSkill once a jump policy exists
    (train: uv run train Mjlab-Jump-Flat-MicroDuck; export: scripts/export.py).
    """

    def __init__(self, duck: DuckRuntime, duration: float = 0.6):
        self.duck = duck
        self.duration = duration
        self.state = "idle"
        self._t = 0.0
        self._prev_policy = "stand"
        self.max_feet_z = 0.0

    def feet_z(self):
        l = self.duck.site_pos("left_foot")[2]
        r = self.duck.site_pos("right_foot")[2]
        return float(l), float(r)

    @property
    def airborne(self) -> bool:
        l, r = self.feet_z()
        return min(l, r) > 0.012

    def trigger(self):
        if self.state != "idle":
            return False
        self._prev_policy = self.duck.active_policy
        self.duck.active_policy = "jump"
        self.duck.set_command()  # all-zero 13D, like roulade/kick
        self.state = "jump"
        self._t = 0.0
        self.max_feet_z = 0.0
        return True

    def update(self, dt: float):
        if self.state == "idle":
            return
        self._t += dt
        l, r = self.feet_z()
        self.max_feet_z = max(self.max_feet_z, min(l, r))
        if self._t >= self.duration:
            self.duck.active_policy = self._prev_policy
            self.duck.set_command()
            self.state = "idle"
