"""Observable-motion rule baseline over frozen learned 61D motor policies.

Coach attacker is intentionally disclosed. Defense never reads its state.
"""

from dataclasses import dataclass
import math
import numpy as np
from .observations import relative


@dataclass(frozen=True)
class MotorCommand:
    owner: str
    policy: str = "stand"
    twist: tuple = (0.0, 0.0, 0.0)
    head: tuple = (0.0, 0.0, 0.0, 0.0)


class Maneuver:
    def __init__(self, name, role, config):
        self.name, self.role, self.c = name, role, config
        self.phase, self.skill = "READY", "approach_orbit"
        self.since = 0.0
        self.origin = None
        self.attack_id = None
        self.serial = 0
        self.aim = 0.0
        self.events = []
        self.evaded = False
        self.min_range = float("inf")
        self.last_dash_end = -10.0
        self.recovery_since = None
        self.stable_since = None
        self.contact_since = None

    def transition(self, now, phase, skill, me):
        self.phase, self.skill, self.since, self.origin = (
            phase,
            skill,
            now,
            np.array(me.xy),
        )
        self.events.append(
            dict(
                time_s=now,
                state="MANEUVER",
                actor=self.name,
                phase=phase,
                skill=skill,
                attack_id=self.attack_id,
            )
        )

    @property
    def active_attack(self):
        return (
            self.attack_id
            if self.skill == "dash_bump" and self.phase == "ACTIVE"
            else None
        )

    def choose(self, now, me, other):
        _, dist, angle, closing = relative(me, other)
        self.min_range = min(self.min_range, dist)
        # One owner. Mild lean uses stand to capture balance; deep falls may
        # enter a bounded get-up candidate. No root motion or target overrides.
        if me.upright < 0.72 and self.skill != "recover_guard":
            self.recovery_since, self.stable_since = now, None
            self.transition(now, "STABILIZE", "recover_guard", me)
        if self.skill == "recover_guard":
            good = me.upright > 0.94 and me.foot_support and me.speed < 0.16
            self.stable_since = (
                (self.stable_since if self.stable_since is not None else now)
                if good
                else None
            )
            if self.stable_since is not None and now - self.stable_since >= 0.3 - 1e-9:
                self.events.append(
                    dict(
                        state="RECOVERED",
                        time_s=now,
                        actor=self.name,
                        duration_s=now - self.recovery_since,
                    )
                )
                self.transition(now, "READY", "approach_orbit", me)
                return MotorCommand("recovery")
            elapsed = now - self.recovery_since
            policy = (
                "sitstand"
                if 2.0 <= elapsed < 2.8 and me.upright < 0.35 and me.z > -0.08
                else "stand"
            )
            return MotorCommand("recovery", policy)
        if now < 0.8:
            return MotorCommand("settle")
        if self.c.ruleset == "sparring_v1" and other.upright < 0.5:
            return (
                MotorCommand("sparring_retreat", "walk", (-0.30, 0.0, 0.0))
                if dist < 0.38
                else MotorCommand("sparring_wait")
            )
        # Independent safety interrupt; a legal airborne step is not a KO.
        if me.boundary_distance < 0.10 and self.phase in ("READY", "WINDUP", "ACTIVE"):
            self.transition(now, "RECOVERY", "approach_orbit", me)
            return MotorCommand("boundary", "walk", (-0.12, 0.0, 0.0))
        if self.role in ("stand", "absent"):
            return MotorCommand("tactics")
        elapsed = now - self.since
        moved = (
            np.linalg.norm(np.array(me.xy) - self.origin)
            if self.origin is not None
            else 0.0
        )
        if self.skill == "dash_bump":
            if self.phase == "WINDUP":
                # Head command is a native delta-from-HOME, never an override.
                if elapsed >= 0.12 and me.upright > 0.9 and me.foot_support:
                    self.transition(now, "ACTIVE", "dash_bump", me)
                return MotorCommand(
                    "maneuver", "walk", (0.0, 0.0, 0.0), (-0.12, -0.10, 0.0, 0.0)
                )
            if self.phase == "ACTIVE":
                if me.touching_opponent and self.contact_since is None:
                    self.contact_since = now
                # A tiny first solver force is not a completed impact. Let
                # contact develop, then brake once closing motion is arrested.
                impact_finished = (
                    self.contact_since is not None
                    and now - self.contact_since >= 0.16
                    and closing < 0.05
                )
                # Heading locked at launch. No perfect mid-dash pursuit.
                if moved >= 0.27 or elapsed >= 1.25 or impact_finished:
                    self.transition(now, "RECOVERY", "dash_bump", me)
                    self.last_dash_end = now
                    self.events.append(
                        dict(
                            state="ATTACK_WINDOW_ENDED",
                            time_s=now,
                            actor=self.name,
                            attack_id=self.attack_id,
                        )
                    )
                    return MotorCommand("maneuver", "walk")
                error = math.atan2(
                    math.sin(self.aim - me.yaw), math.cos(self.aim - me.yaw)
                )
                return MotorCommand(
                    "maneuver",
                    "walk",
                    (self.c.dash_speed, 0.0, float(np.clip(2 * error, -0.5, 0.5))),
                    (-0.12, -0.1, 0.0, 0.0),
                )
            if self.phase in ("RECOVERY", "STABILIZE"):
                if (
                    elapsed >= 0.16
                    and dist > 0.32
                    and me.speed < 0.13
                    and me.upright > 0.93
                    and me.foot_support
                ):
                    self.transition(now, "READY", "approach_orbit", me)
                return (
                    MotorCommand("maneuver", "walk", (-0.30, 0.0, 0.0))
                    if dist < 0.32
                    else MotorCommand("maneuver", "walk")
                )
        if self.skill == "evade_step":
            if self.phase == "ACTIVE":
                if moved >= 0.12 or elapsed >= 1.15:
                    self.evaded = True
                    self.transition(now, "RECOVERY", "evade_step", me)
                else:
                    return MotorCommand(
                        "maneuver",
                        "walk",
                        (-0.26, self.c.evade_sign * self.c.evade_speed, 0.0),
                    )
            if me.speed > 0.15 or elapsed < 0.12:
                return MotorCommand("maneuver", "walk")
            self.transition(now, "READY", "approach_orbit", me)
        if self.role in ("evade", "counter") and not self.evaded:
            if dist < self.c.threat_distance_m and closing > 0.18:
                self.transition(now, "ACTIVE", "evade_step", me)
                self.events.append(
                    dict(
                        state="OBSERVED_THREAT",
                        time_s=now,
                        actor=self.name,
                        distance_m=dist,
                        closing_m_s=closing,
                    )
                )
                return MotorCommand(
                    "maneuver",
                    "walk",
                    (-0.26, self.c.evade_sign * self.c.evade_speed, 0.0),
                )
            return MotorCommand("tactics")
        if self.role == "evade":
            return MotorCommand("tactics")
        # A one-exchange coach holds after its first attack, then the defender
        # must reorient using visible geometry. This is a disclosed curriculum.
        if self.serial >= 1:
            return MotorCommand("coach_exchange_complete")
        turn = float(np.clip(2.8 * angle, -1.0, 1.0))
        # Counter requires observed passing/withdrawal, not an opponent phase ID.
        counter_window = self.role != "counter" or (
            self.evaded and (closing < 0.03 or dist > 0.24)
        )
        if (
            dist < self.c.dash_start_distance_m
            and abs(angle) < 0.22
            and now - self.last_dash_end > 0.65
            and counter_window
            and me.upright > 0.94
            and me.foot_support
        ):
            self.serial += 1
            self.attack_id = f"{self.name}:{self.serial}"
            self.aim = me.yaw + angle
            self.transition(now, "WINDUP", "dash_bump", me)
            return MotorCommand("maneuver", "walk")
        speed = 0.28 if dist > 0.29 or abs(angle) > 0.22 else 0.0
        return MotorCommand("tactics", "walk", (speed, 0.0, turn))
