"""Public-geometry 2v2 tactics over the pinned learned body motors.

Roles are requests, not claims of success. Neither planner nor pilot reads an
opponent controller, rewrites a root, modifies collision, or awards damage.
"""

from dataclasses import asdict
import math

import numpy as np

from .maneuver import MotorCommand
from .observations import relative
from .tactics import Pilot, local, wrap
from .team_intents import Assignment, escape_waypoint
from .transition_contracts import evaluate_entry


def distance(a, b):
    return float(np.linalg.norm(np.array(a.xy) - b.xy))


def bearing(a, b):
    delta = np.array(b.xy) - a.xy
    return wrap(math.atan2(delta[1], delta[0]) - a.yaw)


def inside(point, radius, margin=0.19):
    point = np.asarray(point, dtype=float)
    limit = radius * math.cos(math.pi / 8) - margin
    return point * min(1.0, limit / max(1e-9, np.linalg.norm(point)))


class TacticalTeamPlanner:
    """Retain two lanes; exploit an exposed opponent without a four-body pile."""

    def __init__(self, team, teams, hold_s=1.6, opening_split=False, revision=3):
        self.team, self.teams, self.hold_s = team, teams, hold_s
        self.assignments, self.events = {}, []
        self.last_plan = -10.0
        self.style = "mixed"

    def assignment(self, name):
        return self.assignments.get(name)

    def update(self, now, public, alive):
        allies = sorted(n for n in alive if self.teams[n] == self.team)
        enemies = sorted(n for n in alive if self.teams[n] != self.team)
        if not allies or not enemies:
            self.assignments = {}
            return

        def dist(a, b):
            return distance(public[a], public[b])

        invalid = set(self.assignments) != set(allies) or any(
            a.target not in enemies or (a.protected and a.protected not in allies)
            for a in self.assignments.values()
        )
        endangered = [
            n
            for n in allies
            if now > 0.8
            and public[n].boundary_distance < 0.21
            and min(dist(n, e) for e in enemies) < 0.42
        ]
        emergency = endangered and not any(
            a.role == "ESCAPE" for a in self.assignments.values()
        )
        if not invalid and not emergency and now - self.last_plan < self.hold_s:
            return
        if len(allies) == 2 and endangered:
            protected = min(endangered, key=lambda n: public[n].boundary_distance)
            pursuer = min(enemies, key=lambda n: dist(protected, n))
            plan = {
                n: Assignment(
                    pursuer,
                    "ESCAPE" if n == protected else "COVER",
                    "COVER_WITHDRAWAL",
                    now,
                    protected,
                )
                for n in allies
            }
        elif len(allies) == len(enemies) == 2:
            # Match the front to its nearest foe; the second duck takes a
            # different approach lane, rather than constantly sharing a point.
            order = min(
                (enemies, enemies[::-1]),
                key=lambda es: sum(dist(a, e) for a, e in zip(allies, es)),
            )
            plan = {
                n: Assignment(e, "PRESS", "TWO_LANES", now)
                for n, e in zip(allies, order)
            }
            opportunities = [
                e
                for e in enemies
                if public[e].upright < 0.93 or public[e].boundary_distance < 0.28
            ]
            if opportunities:
                target = min(
                    opportunities,
                    key=lambda e: public[e].upright + public[e].boundary_distance,
                )
                front = min(allies, key=lambda n: dist(n, target))
                wing = next(n for n in allies if n != front)
                other = next(e for e in enemies if e != target)
                plan[front] = Assignment(target, "PRESS", "PRESS_AND_SCREEN", now)
                # A free teammate can cut the flank; a nearby second foe must
                # be screened instead of ignored to manufacture a 2-on-1.
                if dist(wing, other) > 0.45 or public[other].upright < 0.8:
                    plan[wing] = Assignment(target, "FLANK", "PRESS_AND_FLANK", now)
                else:
                    plan[wing] = Assignment(other, "SCREEN", "PRESS_AND_SCREEN", now)
            elif self.style != "pressure":
                # Counter-oriented team: the closer ally draws pressure while
                # its wing cuts behind the same enemy if the other lane is free.
                previous_draw = next(
                    (
                        n
                        for n, a in self.assignments.items()
                        if a.role == "DRAW" and n in allies
                    ),
                    None,
                )
                front = previous_draw or min(
                    allies, key=lambda n: dist(n, plan[n].target)
                )
                wing = next(n for n in allies if n != front)
                target = plan[front].target
                other = next(e for e in enemies if e != target)
                plan[front] = Assignment(target, "DRAW", "DRAW_AND_CUT", now)
                if dist(wing, other) > 0.22:
                    plan[wing] = Assignment(target, "FLANK", "DRAW_AND_CUT", now)
                else:
                    plan[wing] = Assignment(other, "SCREEN", "DRAW_AND_CUT", now)
        else:
            focus = min(enemies, key=lambda e: sum(dist(n, e) for n in allies))
            front = min(allies, key=lambda n: dist(n, focus))
            plan = {
                n: Assignment(
                    focus,
                    "PRESS" if n == front else "FLANK",
                    "PINCER" if len(allies) == 2 else "SOLO",
                    now,
                )
                for n in allies
            }
        old_signature = {
            n: (a.target, a.role, a.intent, a.protected)
            for n, a in self.assignments.items()
        }
        signature = {
            n: (a.target, a.role, a.intent, a.protected) for n, a in plan.items()
        }
        for n, a in list(plan.items()):
            if n in self.assignments and signature[n] == old_signature.get(n):
                plan[n] = self.assignments[n]
        if signature != old_signature:
            self.events.append(
                dict(
                    time_s=now,
                    state="TEAM_PLAN",
                    team=self.team,
                    assignments={n: asdict(a) for n, a in plan.items()},
                )
            )
        self.assignments, self.last_plan = plan, now


class TacticalPilot(Pilot):
    def __init__(self, name, team, teams, config, planner=None):
        super().__init__(name, team, teams, config)
        self.planner, self.role, self.motor_context = planner, "PRESS", None
        self.style = config.red_v2_style if team == "RED" else config.blue_v2_style
        self.until = 0.0
        self.side = 1
        self.counter_until = -1.0
        self.last_feint = -10.0
        self.last_evade = -10.0
        self.last_exit = -10.0
        self.start_distance = 0.0
        self.minimum_distance = 10.0
        self.visible_charge = False
        self.executed_assignment = None
        self.after_exit = None

    @property
    def active_attack(self):
        return self.attack_id if self.action == 3 and self.phase == "ACTIVE" else None

    def start(self, now, skill, duration, me, other, phase="ACTIVE", **evidence):
        self.action = 3 if skill in ("short_bump", "drive", "counter_drive") else 0
        self.skill, self.phase = skill, phase
        self.since, self.phase_since, self.until = now, now, now + duration
        self.origin = np.array(me.xy)
        self.aim = me.yaw + bearing(me, other)
        self.start_distance = self.minimum_distance = distance(me, other)
        self.contact_since = None
        self.brake.reset()
        if self.action == 3:
            self.serial += 1
            self.attack_id = f"{self.name}:{self.serial}"
        self.event(
            now,
            "MANEUVER",
            skill=skill,
            phase=phase,
            target=self.target,
            role=self.role,
            attack_id=self.attack_id if self.action == 3 else None,
            action=skill.upper(),
            public_evidence=evidence,
        )

    def navigate(self, me, point, speed=0.28, face=None, public=None):
        delta = np.array(point) - me.xy
        length = np.linalg.norm(delta)
        velocity = delta / max(1e-9, length) * min(speed, 2 * length)
        if public:
            # Avoid friendly and unrelated bodies using motor steering only.
            for n, b in public.items():
                if n == self.name or n == self.target:
                    continue
                away = np.array(me.xy) - b.xy
                d = np.linalg.norm(away)
                if 0.01 < d < 0.27:
                    velocity += away / d * (0.27 - d) * 1.8
        v = local(velocity, me.yaw)
        angle = (
            bearing(me, face) if face else wrap(math.atan2(delta[1], delta[0]) - me.yaw)
        )
        # Continuous bounded inputs avoid the old all-or-nothing turn command.
        return MotorCommand(
            "tactics",
            "walk",
            (
                float(np.clip(v[0], -0.30, 0.30)),
                float(np.clip(v[1], -0.30, 0.30)),
                float(np.clip(2.6 * angle, -1.0, 1.0)),
            ),
        )

    def free_side(self, me, public):
        lateral = np.array([-math.sin(me.yaw), math.cos(me.yaw)])

        def score(side):
            p = np.array(me.xy) + side * 0.20 * lateral
            return np.linalg.norm(p) + sum(
                max(0.0, 0.27 - np.linalg.norm(p - b.xy)) * 3
                for n, b in public.items()
                if n != self.name
            )

        return min((1, -1), key=score)

    def choose(self, now, public, alive):
        me = public[self.name]
        recovery = self.guard.choose(now, me)
        if recovery is not None:
            if self.action is not None:
                self.finish(now, "RECOVERY_PREEMPTION")
            self.phase, self.skill = "STABILIZE", "recover_guard"
            return MotorCommand("recovery", recovery)
        if now < 0.8:
            return MotorCommand("settle")
        if self.target not in alive and self.action is not None:
            self.finish(now, "TARGET_ELIMINATED")
        if self.action is None:
            assignment = self.planner.assignment(self.name) if self.planner else None
            target = (
                assignment.target
                if assignment
                else min(
                    (n for n in alive if self.teams[n] != self.team),
                    key=lambda n: distance(me, public[n]),
                    default=None,
                )
            )
            if not target:
                return MotorCommand("no_live_opponent")
            if assignment != self.executed_assignment:
                self.event(
                    now,
                    "TEAM_ASSIGNMENT_ADOPTED",
                    target=target,
                    role=assignment.role if assignment else "PRESS",
                    intent=assignment.intent if assignment else "SOLO",
                )
            self.target = target
            self.role = assignment.role if assignment else "PRESS"
            self.executed_assignment = assignment
        other = public[self.target]
        _, dist, angle, closing = relative(me, other)
        enemy_toward = float(
            np.array(other.velocity) @ ((np.array(me.xy) - other.xy) / max(1e-9, dist))
        )
        cue = other.neck_pitch_delta < -0.22 and other.head_pitch_delta < -0.02
        threat = (
            other.upright > 0.9
            and dist < 0.43
            and abs(bearing(other, me)) < 0.65
            and (enemy_toward > 0.14 or cue)
        )
        if me.boundary_distance < 0.10 and self.skill != "escape":
            self.finish(now, "BOUNDARY_PREEMPTION")
            self.start(now, "escape", 0.8, me, other)
        if (
            self.action is not None
            and self.skill in ("approach", "cut_flank", "draw", "feint")
            and now - self.since >= 0.12
            and threat
            and self.style != "pressure"
            and now - self.last_evade > 1.0
        ):
            self.finish(now, "VISIBLE_THREAT")
        if self.action is None:
            self.side = self.free_side(me, public)
            flank_angle = abs(bearing(other, me))
            entry = evaluate_entry(me, self.motor_context)
            attack = (
                0.19 < dist < self.c.dash_start_distance_m
                and abs(angle) < 0.30
                and now - self.last_attack_end > 0.50
                and me.upright > 0.96
                and me.foot_support
                and not me.body_ground
            )
            if self.role == "ESCAPE":
                skill, duration = "escape", 0.65
            elif threat and self.style != "pressure" and now - self.last_evade > 1.0:
                skill, duration = "sidestep", 0.54
                self.last_evade, self.visible_charge = now, True
            elif me.touching_opponent or dist < 0.19:
                skill, duration = "break_contact", 0.72
                self.last_exit = now
            elif attack and now < self.counter_until and entry["accepted"]:
                skill, duration = "counter_drive", 0.75
            elif attack and (
                flank_angle > 0.85
                or other.upright < 0.94
                or self.style == "pressure"
                or other.boundary_distance < 0.24
            ):
                skill, duration = "drive", 1.05
            elif attack and self.role == "DRAW" and now - self.last_feint > 3.0:
                skill, duration = "feint", 0.16
                self.last_feint = now
            elif attack:
                skill, duration = "short_bump", 0.50
            elif self.role == "FLANK":
                skill, duration = "cut_flank", 0.35
            elif self.role == "DRAW" and dist < 0.42 and enemy_toward > 0.045:
                skill, duration = "draw", 0.52
            else:
                skill, duration = "approach", 0.32
            self.start(
                now,
                skill,
                duration,
                me,
                other,
                phase="WINDUP"
                if skill in ("drive", "short_bump", "counter_drive")
                else "ACTIVE",
                target_bearing_rad=flank_angle,
                enemy_toward_m_s=enemy_toward,
                target_upright=other.upright,
                target_margin_m=other.boundary_distance,
            )
        elapsed = now - self.since
        self.minimum_distance = min(self.minimum_distance, dist)
        if self.action == 3:
            if self.phase == "WINDUP":
                windup = 0.12 if self.skill in ("counter_drive", "short_bump") else 0.24
                if now - self.phase_since >= windup - 1e-9:
                    self.phase, self.phase_since = "ACTIVE", now
                    self.event(
                        now,
                        "ATTACK_LAUNCHED",
                        target=self.target,
                        attack_id=self.attack_id,
                    )
                return MotorCommand("maneuver", "walk", head=(-0.30, -0.15, 0.0, 0.0))
            if self.phase == "ACTIVE":
                if me.touching_opponent and self.contact_since is None:
                    self.contact_since = now
                contact_done = (
                    self.contact_since is not None
                    and now - self.contact_since
                    > (0.16 if self.skill == "short_bump" else 0.44)
                )
                active_limit = 0.42 if self.skill == "short_bump" else 0.80
                if (
                    now - self.phase_since >= active_limit
                    or contact_done
                    or np.linalg.norm(np.array(me.xy) - self.origin) > 0.24
                ):
                    self.phase, self.phase_since = "RECOVERY", now
                    self.event(
                        now,
                        "ATTACK_ACTIVE_ENDED",
                        target=self.target,
                        attack_id=self.attack_id,
                    )
                else:
                    return MotorCommand(
                        "maneuver",
                        "dash_candidate",
                        (
                            self.c.dash_speed,
                            0.0,
                            float(np.clip(2 * wrap(self.aim - me.yaw), -0.5, 0.5)),
                        ),
                        (-0.30, -0.15, 0.0, 0.0),
                    )
            if self.brake.ready(now, me):
                self.finish(now)
                self.start(now, "break_contact", 0.64, me, other)
            else:
                return MotorCommand("maneuver", "dash_candidate")
        if self.skill == "sidestep":
            if elapsed < 0.40 and np.linalg.norm(np.array(me.xy) - self.origin) < 0.12:
                return MotorCommand(
                    "maneuver", "evade_candidate", (0.0, self.side * 0.3, 0.0)
                )
            # A loaded, upright stance may turn straight into walking. Otherwise
            # preserve the old zero-command brake; no joint output blending.
            if evaluate_entry(me, self.motor_context)["accepted"] or self.brake.ready(
                now, me
            ):
                self.counter_until = now + 2.0
                self.event(
                    now,
                    "EVADE_TO_TURN",
                    target=self.target,
                    separation_gain_m=dist - self.minimum_distance,
                )
                self.finish(now)
                self.start(now, "turn_counter", 0.8, me, other)
            else:
                return MotorCommand("maneuver", "evade_candidate")
        if self.skill == "turn_counter":
            if (
                dist < 0.40
                and abs(angle) < 0.30
                and evaluate_entry(me, self.motor_context)["accepted"]
            ):
                self.finish(now)
                self.start(now, "counter_drive", 0.75, me, other, phase="WINDUP")
                self.event(
                    now,
                    "COUNTER_ATTEMPT",
                    target=self.target,
                    source="PUBLIC_THREAT_EVADE",
                )
            else:
                command = self.navigate(me, other.xy, face=other, public=public)
                if elapsed > 0.8:
                    self.finish(now)
                return command
        if self.skill == "counter_drive":
            return MotorCommand("maneuver", "walk", head=(-0.30, -0.15, 0.0, 0.0))
        if now >= self.until - 1e-9:
            previous = self.skill
            self.finish(now)
            if previous == "feint":
                self.start(now, "draw", 0.6, me, other)
            else:
                return MotorCommand("tactics", "walk")
        if self.skill in ("draw", "break_contact"):
            away = np.array(me.xy) - other.xy
            away /= max(1e-9, np.linalg.norm(away))
            lateral = np.array([-away[1], away[0]]) * self.side
            point = inside(
                np.array(me.xy) + away * 0.20 + lateral * 0.12, self.c.radius_m
            )
            return self.navigate(me, point, face=other, public=public)
        if self.skill == "escape":
            return self.navigate(
                me, escape_waypoint(me, other, public, self.c.radius_m), public=public
            )
        if self.skill == "feint":
            return MotorCommand("tactics", "walk", head=(-0.30, -0.15, 0.0, 0.0))
        if self.skill == "cut_flank":
            radial = np.array(me.xy) - other.xy
            theta = math.atan2(radial[1], radial[0])
            back = wrap(other.yaw + math.pi - theta)
            side = 1 if back > 0 else -1
            # Follow an arc around the target instead of crossing its body.
            advance = side * min(0.65, abs(back))
            radius = max(0.27, min(0.37, dist))
            point = inside(
                np.array(other.xy)
                + radius
                * np.array([math.cos(theta + advance), math.sin(theta + advance)]),
                self.c.radius_m,
            )
            if abs(back) < 0.65 and 0.20 < dist < 0.38:
                point = np.array(other.xy)
            return self.navigate(me, point, face=other, public=public)
        aim = np.array(other.xy) + np.array(other.velocity) * 0.18
        return self.navigate(me, aim, face=other, public=public)
