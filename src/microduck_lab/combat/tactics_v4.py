"""Supported contact continuation and inside-to-outside finishing geometry.

These rules request learned joint control. A finish attempt can miss, topple its
own actor or fail to eliminate; only the existing physical referee calls a win.
"""

import math
import numpy as np

from .tactics_v3 import TacticalPilot, bearing, distance, inside
from .tactics import wrap


class FinisherPilot(TacticalPilot):
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
