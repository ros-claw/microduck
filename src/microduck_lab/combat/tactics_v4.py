"""Supported contact continuation and inside-to-outside finishing geometry.

These rules request learned joint control. A finish attempt can miss, topple its
own actor or fail to eliminate; only the existing physical referee calls a win.
"""

import math
import numpy as np

from .tactics_v3 import TacticalPilot, bearing, distance, inside
from .tactics import wrap
from .geometry import boundary_distance
from .maneuver import MotorCommand


class FinisherPilot(TacticalPilot):
    def control_override(self, now, public, alive):
        me = public[self.name]
        enemies = [n for n in alive if self.teams[n] != self.team]
        if self.skill == "edge_brake":
            if self.brake.ready(now, me) and me.speed < 0.10:
                self.finish(now, "SUPPORTED_EDGE_STOP")
                if not enemies:
                    return MotorCommand("edge_stop", "walk")
                target = min(enemies, key=lambda n: distance(me, public[n]))
                self.target = target
                self.start(now, "escape", 0.8, me, public[target])
                return None
            else:
                return MotorCommand("edge_brake", self.edge_brake_policy)
        future_margin = boundary_distance(
            np.array(me.xy) + 0.35 * np.array(me.velocity), self.c
        )
        if (
            me.upright > 0.94
            and not me.body_ground
            and future_margin < 0.12
            and future_margin < me.boundary_distance - 0.025
        ):
            policy = (
                "finish_candidate"
                if self.skill == "finish_drive" and self.c.finish_motor_path
                else "dash_candidate"
                if self.action == 3
                else "evade_candidate"
                if self.skill == "sidestep"
                else "walk"
            )
            self.finish(now, "PREDICTED_EDGE_PREEMPTION")
            if not enemies:
                return MotorCommand("edge_stop", policy)
            target = min(enemies, key=lambda n: distance(me, public[n]))
            self.target = target
            self.start(
                now,
                "edge_brake",
                3.0,
                me,
                public[target],
                predicted_margin_m=future_margin,
                measured_margin_m=me.boundary_distance,
            )
            self.edge_brake_policy = policy
            self.brake.ready(now, me)
            return MotorCommand("edge_brake", policy)
        return None

    def special_choice(self, now, me, other, public, alive):
        dist = distance(me, other)
        if not (
            0.135 < dist < self.c.finish_entry_m
            and abs(bearing(me, other)) < 0.36
            and me.upright > 0.95
            and me.foot_support
            and not me.body_ground
            and me.boundary_distance > self.c.finish_safety_m
            and now - self.last_attack_end > 0.35
            and other.z > 0.02
        ):
            return None
        outward = np.array(other.xy) / max(1e-9, np.linalg.norm(other.xy))
        inward_advantage = float((np.array(other.xy) - me.xy) @ outward)
        exposed = abs(bearing(other, me)) > 1.05 or other.upright < 0.90
        inside_edge = other.boundary_distance < 0.38 and inward_advantage > 0.025
        if inside_edge or exposed or (me.touching_opponent and dist < 0.23):
            return "finish_drive", self.c.finish_active_s + 0.24
        return None

    def attack_limits(self, me, other):
        if self.skill == "finish_drive":
            return (
                self.c.finish_active_s,
                self.c.finish_contact_s,
                self.c.finish_travel_m,
            )
        return super().attack_limits(me, other)

    def drive_yaw(self, me, other):
        if self.skill == "finish_drive":
            return float(np.clip(2 * bearing(me, other), -0.5, 0.5))
        return super().drive_yaw(me, other)

    def navigate(self, me, point, speed=0.28, face=None, public=None):
        if (
            self.skill == "approach"
            and face is not None
            and face.boundary_distance < 0.37
            and np.linalg.norm(np.array(point) - face.xy) < 0.08
            and not me.touching_opponent
        ):
            # Motor-only approach from the inner side: follow an arc if the
            # straight route would cross the opponent. No collision exclusion.
            radial = np.array(me.xy) - face.xy
            theta = math.atan2(radial[1], radial[0])
            desired = math.atan2(-face.xy[1], -face.xy[0])
            difference = wrap(desired - theta)
            if abs(difference) > 0.55:
                advance = float(np.clip(difference, -0.55, 0.55))
                radius = max(0.26, min(0.37, np.linalg.norm(radial)))
                point = inside(
                    np.array(face.xy)
                    + radius
                    * np.array([math.cos(theta + advance), math.sin(theta + advance)]),
                    self.c.radius_m,
                )
        return super().navigate(me, point, speed, face, public)
