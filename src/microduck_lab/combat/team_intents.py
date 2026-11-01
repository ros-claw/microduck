"""Transparent shared team scheduler, using one same-tick public snapshot."""

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class Assignment:
    target: str
    role: str
    intent: str
    since_s: float
    protected: str | None = None


class TeamPlanner:
    def __init__(self, team, teams, hold_s=1.6, opening_split=False, revision=1):
        self.team, self.teams, self.hold_s = team, teams, hold_s
        self.assignments, self.events = {}, []
        self.last_plan = -10.0
        self.opening_split = opening_split
        self.revision = revision
        self.cover_since = None
        self.clear_since = None

    def update(self, now, public, alive):
        allies = sorted(n for n in alive if self.teams[n] == self.team)
        enemies = sorted(n for n in alive if self.teams[n] != self.team)
        if not allies or not enemies:
            self.assignments = {}
            return

        def distance(a, b):
            return float(np.linalg.norm(np.array(public[a].xy) - public[b].xy))

        # Keep a cover assignment through the actual approach. Momentary gait
        # recovery or a single safe boundary sample does not finish a rescue.
        if self.revision == 2 and self.cover_since is not None:
            cover = next(
                (a for a in self.assignments.values() if a.role == "PEEL"), None
            )
            if (
                cover
                and cover.protected in allies
                and cover.target in enemies
                and set(self.assignments) == set(allies)
            ):
                b = public[cover.protected]
                clear = (
                    b.upright > 0.94
                    and b.boundary_distance > 0.29
                    and min(distance(cover.protected, e) for e in enemies) > 0.37
                )
                self.clear_since = (
                    (now if self.clear_since is None else self.clear_since)
                    if clear
                    else None
                )
                if now - self.cover_since < 5.0 and (
                    self.clear_since is None or now - self.clear_since < 0.8
                ):
                    return
            self.cover_since, self.clear_since = None, None

        def threatened(n):
            b = public[n]
            nearest = min(distance(n, e) for e in enemies)
            if self.revision == 1:
                return (
                    b.boundary_distance < 0.20 or b.upright < 0.8
                ) and nearest < 0.50
            # A short, public velocity projection notices outward pressure
            # before the victim has already crossed the last safe footstep.
            radial = np.array(b.xy) / max(1e-9, np.linalg.norm(b.xy))
            outward = max(0.0, float(np.array(b.velocity) @ radial))
            return (
                b.boundary_distance - 0.6 * outward < 0.26 or b.upright < 0.88
            ) and nearest < 0.46

        endangered = [n for n in allies if threatened(n)]
        if self.revision == 2 and now < 0.8:
            endangered = []  # Motors are settling, not executing team roles.
        protected = (
            min(endangered, key=lambda n: public[n].boundary_distance)
            if endangered
            else None
        )
        emergency = protected is not None and any(
            a.intent != "PEEL_COVER" or a.protected != protected
            for a in self.assignments.values()
        )
        invalid = set(self.assignments) != set(allies) or any(
            a.target not in enemies for a in self.assignments.values()
        )
        if not invalid and not emergency and now - self.last_plan < self.hold_s:
            return
        if protected is not None and len(allies) == 2:
            pursuer = min(enemies, key=lambda e: distance(protected, e))
            plan = {
                n: Assignment(
                    pursuer,
                    "ESCAPE" if n == protected else "PEEL",
                    "PEEL_COVER",
                    now,
                    protected,
                )
                for n in allies
            }
            if self.revision == 2:
                self.cover_since, self.clear_since = now, None
        elif len(allies) == len(enemies) == 2 and self.opening_split:
            order = min(
                (enemies, enemies[::-1]),
                key=lambda order: sum(distance(n, e) for n, e in zip(allies, order)),
            )
            plan = {
                n: Assignment(e, "PRESS", "SPLIT_PRESSURE_COURSE", now)
                for n, e in zip(allies, order)
            }
        elif len(allies) == 2:
            focus = min(enemies, key=lambda e: (sum(distance(n, e) for n in allies), e))
            previous_focus = (
                next(iter(self.assignments.values())).target
                if self.assignments
                else None
            )
            if (
                previous_focus in enemies
                and sum(
                    distance(n, previous_focus) - distance(n, focus) for n in allies
                )
                < 0.12
            ):
                focus = previous_focus
            front = min(allies, key=lambda n: (distance(n, focus), n))
            plan = {
                n: Assignment(
                    focus, "PRESS" if n == front else "FLANK", "PRESS_FLANK", now
                )
                for n in allies
            }
            # Swap the actual front duty periodically, not just the displayed ID.
            if self.assignments and now - self.last_plan >= self.hold_s:
                old_front = next(
                    (n for n, a in self.assignments.items() if a.role == "PRESS"), None
                )
                if old_front in allies and public[old_front].touching_opponent:
                    front = next(n for n in allies if n != old_front)
                    plan = {
                        n: Assignment(
                            focus, "PRESS" if n == front else "RESET", "RESET_SWAP", now
                        )
                        for n in allies
                    }
        else:
            plan = {
                n: Assignment(
                    min(enemies, key=lambda e: distance(n, e)), "PRESS", "SOLO", now
                )
                for n in allies
            }
        if self.revision == 2:
            for name, assignment in list(plan.items()):
                old = self.assignments.get(name)
                if old and (old.target, old.role, old.intent, old.protected) == (
                    assignment.target,
                    assignment.role,
                    assignment.intent,
                    assignment.protected,
                ):
                    plan[name] = old
        if {n: (a.target, a.role, a.intent, a.protected) for n, a in plan.items()} != {
            n: (a.target, a.role, a.intent, a.protected)
            for n, a in self.assignments.items()
        }:
            self.events.append(
                dict(
                    time_s=now,
                    state="TEAM_PLAN",
                    team=self.team,
                    assignments={n: asdict(a) for n, a in plan.items()},
                )
            )
        self.assignments, self.last_plan = plan, now

    def assignment(self, name):
        return self.assignments.get(name)


def flank_waypoint(me, target, partner, radius):
    axis = np.array(partner.xy) - target.xy
    axis /= max(1e-8, np.linalg.norm(axis))
    side = np.array([-axis[1], axis[0]])
    if np.dot(np.array(me.xy) - target.xy, side) < 0:
        side = -side
    point = np.array(target.xy) + 0.28 * side
    # A waypoint is just a command objective, not an invisible physical wall.
    length = np.linalg.norm(point)
    return point * min(
        1.0, max(0.1, radius * math.cos(math.pi / 8) - 0.17) / max(1e-9, length)
    )


def escape_waypoint(me, pursuer, public, radius):
    """Choose a short inward route which does not run through the pursuer.

    Public constant-velocity previews are steering costs, not collision edits.
    The victim moves its own motors and all bodies remain collidable.
    """
    origin = np.array(me.xy)
    inward = -origin / max(1e-9, np.linalg.norm(origin))
    if np.linalg.norm(inward) < 0.5:
        inward = origin - pursuer.xy
        inward /= max(1e-9, np.linalg.norm(inward))
    best = None
    for angle in (0.0, -math.pi / 3, math.pi / 3, -math.pi / 2, math.pi / 2):
        c, s = math.cos(angle), math.sin(angle)
        direction = np.array([[c, -s], [s, c]]) @ inward
        point = origin + 0.28 * direction
        length = np.linalg.norm(point)
        safe_radius = min(
            max(0.1, np.linalg.norm(origin) - 0.04),
            radius * math.cos(math.pi / 8) - 0.18,
        )
        point *= min(1.0, safe_radius / max(1e-9, length))
        delta = point - origin
        cost = np.linalg.norm(point) * 0.6
        for other in public.values():
            if other.name == me.name:
                continue
            obstacle = np.array(other.xy) + 0.25 * np.array(other.velocity)
            t = np.clip(
                np.dot(obstacle - origin, delta) / max(1e-9, np.dot(delta, delta)), 0, 1
            )
            clearance = np.linalg.norm(origin + t * delta - obstacle)
            cost += 4.0 * max(0.0, 0.23 - clearance)
        if best is None or cost < best[0]:
            best = cost, point
    return best[1]
