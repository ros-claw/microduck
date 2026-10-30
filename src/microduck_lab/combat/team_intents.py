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
    def __init__(self, team, teams, hold_s=1.6, opening_split=False):
        self.team, self.teams, self.hold_s = team, teams, hold_s
        self.assignments, self.events = {}, []
        self.last_plan = -10.0
        self.opening_split = opening_split

    def update(self, now, public, alive):
        allies = sorted(n for n in alive if self.teams[n] == self.team)
        enemies = sorted(n for n in alive if self.teams[n] != self.team)
        if not allies or not enemies:
            self.assignments = {}
            return

        def distance(a, b):
            return float(np.linalg.norm(np.array(public[a].xy) - public[b].xy))

        endangered = [
            n
            for n in allies
            if (public[n].boundary_distance < 0.20 or public[n].upright < 0.8)
            and min(distance(n, e) for e in enemies) < 0.50
        ]
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
