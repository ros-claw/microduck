"""Candidate long-jump executor and full-rate proof of unsupported gap crossing."""

from dataclasses import dataclass, field
import numpy as np
import mujoco
from .world import foot_support


def floor_contacts(m, d):
    contacts = set()
    for ci, c in enumerate(d.contact):
        if c.dist > 0.0002:
            continue
        for floor, duck in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            name = m.geom(floor).name
            if name.startswith("floor/") and m.body(
                m.geom_bodyid[duck]
            ).name.startswith("duck/"):
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, force)
                if force[0] > 1e-5:
                    contacts.add(name)
    return contacts


@dataclass
class GapAudit:
    near_floor: str = "floor/0"
    far_floor: str = "floor/1"
    previous_contact: set = field(default_factory=set)
    flight: dict | None = None
    flights: list = field(default_factory=list)
    short_same_platform_flights: int = 0

    def sample(self, m, d, r):
        contacts = floor_contacts(m, d)
        if not contacts and self.flight is None and self.previous_contact:
            self.flight = dict(
                start=float(d.time),
                from_floors=sorted(self.previous_contact),
                start_x=float(r.trunk_pos()[0]),
                min_upright=1.0,
            )
        if self.flight is not None:
            self.flight["min_upright"] = min(
                self.flight["min_upright"], float(d.xmat[r.trunk_body_id, 8])
            )
            if contacts:
                self.flight.update(
                    end=float(d.time),
                    to_floors=sorted(contacts),
                    end_x=float(r.trunk_pos()[0]),
                )
                self.flight["duration"] = self.flight["end"] - self.flight["start"]
                self.flight["crossed"] = bool(
                    self.near_floor in self.flight["from_floors"]
                    and self.far_floor in contacts
                    and self.flight["duration"] >= 0.04
                    and self.flight["min_upright"] > 0.8
                )
                if (
                    self.flight["duration"] >= 0.01
                    or self.flight["from_floors"] != self.flight["to_floors"]
                ):
                    self.flights.append(self.flight)
                else:
                    # Summarize sub-10 ms contact chatter; every cross-platform
                    # transition remains explicit, including failed crossings.
                    self.short_same_platform_flights += 1
                self.flight = None
        if contacts:
            self.previous_contact = contacts
        else:
            self.previous_contact = set()

    @property
    def crossed(self):
        return any(f["crossed"] for f in self.flights)


@dataclass
class CandidateJump:
    """Experimental executor, kept separate from the certified live skill registry."""

    started: float
    far_edge: float
    target_y: float = 0.0
    landing_tolerance: float = 0.19
    phase: str = "PREPARE"
    status: str = "RUNNING"
    airborne: bool = False
    stable: float = 0.0
    landed: bool = False
    settled: bool = False
    support_stable: float = 0.0
    name: str = "JUMP_CENTER"
    events: list = field(default_factory=list)

    def update(self, m, d, r, audit, dt=0.02):
        up = float(d.xmat[r.trunk_body_id, 8])
        support = foot_support(m, d, "floor/1")
        if self.phase == "PREPARE":
            r._jv_prev = d.qvel[r.joint_qvel_idx].astype(np.float32).copy()
            self.phase = "LAUNCH"
        if not floor_contacts(m, d) and up > 0.8:
            self.airborne = True
            self.phase = "FLIGHT"
        if self.airborne and support and up > 0.85 and audit.crossed:
            self.landed = True
            self.phase = "ALIGN"
        r.set_command()
        r.bank.mirror_run = False
        if self.landed:
            r.joint_vel_delay = 0
            resting = (
                up > 0.95
                and support
                and r.trunk_pos()[2] > 0.10
                and np.linalg.norm(r.trunk_linvel()) < 0.20
            )
            self.support_stable = self.support_stable + dt if resting else 0.0
            self.settled |= self.support_stable >= 0.08
            if not self.settled:
                r.active_policy = "stand"
            elif abs(r.trunk_yaw()) > 0.20:
                r.active_policy = "run"
                r.command[2] = np.clip(-4 * r.trunk_yaw(), -2.5, 2.5)
            else:
                r.active_policy = "run"
                r.command[:3] = [0.5, 0.0, np.clip(-4 * r.trunk_yaw(), -2.5, 2.5)]
        else:
            r.joint_vel_delay = 1
            r.active_policy = "jump"
        good = (
            self.landed
            and r.trunk_pos()[0] >= self.far_edge
            and audit.crossed
            and up > 0.95
            and support
            and abs(r.trunk_yaw()) < 0.2
            and abs(r.trunk_pos()[1] - self.target_y) < self.landing_tolerance
        )
        self.stable = self.stable + dt if good else 0.0
        if good:
            self.phase = "STABLE"
        if d.time - self.started > 3.0 + 1e-8 or r.trunk_pos()[2] < -0.15:
            self.status = "FAILED"
            self.phase = "TIMEOUT"
        elif self.stable >= 0.10:
            self.status = "SUCCESS"
            self.phase = "SUCCESS"
        if self.status != "RUNNING":
            self.events.append(
                dict(
                    type="SKILL_RESULT",
                    t=float(d.time),
                    skill=self.name,
                    status=self.status,
                    flights=[dict(f) for f in audit.flights if f["crossed"]],
                )
            )
        return self.status
