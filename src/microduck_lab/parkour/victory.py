"""Actuator-only finish flourish; existing learned gait, short learned hops."""

from dataclasses import dataclass, field
import math
import numpy as np
from .skills import lane_command
from .jump import floor_contacts
from .world import foot_support


@dataclass
class VictoryDance:
    started: float
    finish_x: float = 7.95
    finished_at: float | None = None
    events: list = field(default_factory=list)
    flights: list = field(default_factory=list)
    _air: dict | None = None
    min_upright: float = 1.0
    phase: str = "HAPPY APPROACH"

    def update(self, m, d, r):
        t = float(d.time)
        r.bank.mirror_run = False
        r.set_command()
        r.head_override = None
        r.leg_override = None
        if self.finished_at is None:
            self.phase = "HAPPY APPROACH"
            r.active_policy = "run"
            r.command[:3] = lane_command(r, 0.0, 0.5)
            q = t - self.started
            r.head_override = np.array(
                [
                    0.349 + 0.045 * math.sin(8 * q),
                    0.349,
                    0.08 * math.sin(6 * q),
                    0.035 * math.sin(6 * q),
                ]
            )
            if r.trunk_pos()[0] >= self.finish_x:
                self.finished_at = t
                self.events.append(dict(type="FINISH", t=t))
        else:
            q = t - self.finished_at
            self.phase = "BRAKE" if q < 0.8 else "SETTLE"
            r.active_policy = "run" if q < 0.8 else "stand"
            if 1.8 < q < 2.2 or 4.0 < q < 4.4:
                self.phase = "VICTORY HOP"
                r.active_policy = "victory_hop"
            if q >= 4.8:
                self.phase = "HAPPY NOD"
                envelope = min(1.0, (q - 4.8) / 0.4)
                r.head_override = np.array(
                    [
                        0.349 + 0.09 * envelope * math.sin(7 * q),
                        0.349 + 0.04 * envelope * math.sin(7 * q),
                        0.18 * envelope * math.sin(3 * q),
                        0.06 * envelope * math.sin(7 * q),
                    ]
                )
            if q >= 6.4:
                self.phase = "SETTLED"
                r.head_override = None
        r.hop_phase_source = lambda: (
            2 * math.pi * 3.1 * (float(d.time) - (self.finished_at or t)) - math.pi
        )

    def sample(self, m, d, r):
        if self.finished_at is None:
            return
        up = float(d.xmat[r.trunk_body_id, 8])
        self.min_upright = min(self.min_upright, up)
        q = float(d.time) - self.finished_at
        if q < 1.8:
            return
        supported = bool(floor_contacts(m, d))
        if not supported and self._air is None:
            self._air = dict(
                start=float(d.time), max_height=float(r.trunk_pos()[2]), min_upright=up
            )
        if self._air is not None:
            self._air["max_height"] = max(
                self._air["max_height"], float(r.trunk_pos()[2])
            )
            self._air["min_upright"] = min(self._air["min_upright"], up)
            if supported:
                f = self._air | dict(
                    end=float(d.time),
                    duration=float(d.time) - self._air["start"],
                    foot_landing=bool(foot_support(m, d)),
                )
                if f["duration"] >= 0.025:
                    self.flights.append(f)
                    self.events.append(
                        dict(type="VICTORY_HOP_LANDED", t=float(d.time), flight=f)
                    )
                self._air = None

    def report(self, m, d, r):
        qualified = [
            f for f in self.flights if f["foot_landing"] and f["min_upright"] > 0.8
        ]
        passed = (
            self.finished_at is not None
            and self.min_upright > 0.8
            and len(qualified) >= 2
            and d.xmat[r.trunk_body_id, 8] > 0.95
            and foot_support(m, d)
            and r.trunk_pos()[2] > 0.10
        )
        return dict(
            passed=bool(passed),
            phase=self.phase,
            finished_at=self.finished_at,
            min_upright=self.min_upright,
            flights=self.flights,
            qualified_hops=len(qualified),
            method="Existing walking and centered rope-hop motor policies; bounded scripted head servo targets. No root/prop pose or velocity writes; no external robot force.",
        )
