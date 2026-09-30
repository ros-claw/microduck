"""Contact-causal bowling lock and verified encounter combos.

No geometry, root pose or prop velocity is written by this ledger. The driver
opens the force-limited exit only after the recorded contact chain qualifies.
"""

from dataclasses import dataclass, field
import mujoco
import numpy as np


@dataclass
class BowlingLock:
    pins: tuple = ("pin0", "pin1", "pin2")
    events: list = field(default_factory=list)
    impulse: dict = field(default_factory=dict)
    direct_impulse: dict = field(default_factory=dict)
    parents: dict = field(default_factory=dict)
    struck: set = field(default_factory=set)
    toppled: set = field(default_factory=set)
    contaminated: set = field(default_factory=set)
    later_touches: set = field(default_factory=set)
    duck_push_impulse: float = 0.0
    max_push_force: float = 0.0
    unlocked: bool = False
    push_started: bool = False
    boss_ball_interference: bool = False
    _names: tuple = field(default=(), repr=False)
    _relevant: frozenset = field(default_factory=frozenset, repr=False)
    _pin_bodies: dict = field(default_factory=dict, repr=False)

    def sample(self, m, d):
        if self.unlocked:
            return
        if not self._names:
            self._names = tuple(m.body(m.geom_bodyid[g]).name for g in range(m.ngeom))
            self._relevant = frozenset(
                g
                for g, n in enumerate(self._names)
                if n == "playball" or n in self.pins
            )
            self._pin_bodies = {pin: m.body(pin).id for pin in self.pins}
        force = np.zeros(6)
        for i, c in enumerate(d.contact):
            if c.geom1 not in self._relevant and c.geom2 not in self._relevant:
                continue
            names = [self._names[c.geom1], self._names[c.geom2]]
            if "world" in names:
                continue
            mujoco.mj_contactForce(m, d, i, force)
            if force[0] <= 1e-7:
                continue
            if "playball" in names and "boss" in names and not self.unlocked:
                self.boss_ball_interference = True
            if "playball" in names and any(n.startswith("duck/") for n in names):
                self.duck_push_impulse += float(force[0]) * m.opt.timestep
                self.max_push_force = max(self.max_push_force, float(force[0]))
                if not self.push_started:
                    self.push_started = True
                    self.events.append(dict(type="BALL_PUSH", t=float(d.time)))
            for pin in self.pins:
                if pin not in names:
                    continue
                other = names[1] if names[0] == pin else names[0]
                if other.startswith("duck/") or other == "boss":
                    if pin in self.toppled:
                        self.later_touches.add(pin)
                    else:
                        self.contaminated.add(pin)
                if other == "playball":
                    self.direct_impulse[pin] = (
                        self.direct_impulse.get(pin, 0.0)
                        + float(force[0]) * m.opt.timestep
                    )
                causal = (
                    other == "playball" and self.duck_push_impulse > 0.001
                ) or other in self.struck
                if causal and not self.boss_ball_interference:
                    self.impulse[pin] = (
                        self.impulse.get(pin, 0.0) + float(force[0]) * m.opt.timestep
                    )
                    if self.impulse[pin] > 0.0005:
                        self.struck.add(pin)
                        self.parents.setdefault(pin, other)
        for pin in self.pins:
            if (
                pin in self.struck
                and pin not in self.toppled
                and pin not in self.contaminated
                and d.xmat[self._pin_bodies[pin], 8] < 0.35
            ):
                self.toppled.add(pin)
                self.events.append(
                    dict(
                        type="PIN_DOWN",
                        t=float(d.time),
                        pin=pin,
                        chain_impulse_Ns=self.impulse[pin],
                        parent=self.parents[pin],
                    )
                )
        if (
            not self.unlocked
            and not self.boss_ball_interference
            and len(self.toppled) == len(self.pins)
        ):
            self.unlocked = True
            self.events.append(
                dict(
                    type="BOWLING_STRIKE",
                    t=float(d.time),
                    pins=list(self.pins),
                    duck_push_impulse_Ns=self.duck_push_impulse,
                )
            )
            self.events.append(
                dict(
                    type="EXIT_UNLOCK",
                    t=float(d.time),
                    cause="duck → ball → three pins",
                )
            )

    def report(self):
        return dict(
            unlocked=self.unlocked,
            ball_hit_pins=sorted(self.direct_impulse),
            chain_hit_pins=sorted(self.struck),
            causal_parents=self.parents,
            qualified_toppled=sorted(self.toppled),
            direct_or_boss_hit_pins=sorted(self.contaminated),
            touches_after_qualified_topple=sorted(self.later_touches),
            duck_push_impulse_Ns=self.duck_push_impulse,
            max_push_force_N=self.max_push_force,
            ball_pin_impulses_Ns=self.direct_impulse,
            chain_impulses_Ns=self.impulse,
            boss_ball_interference=self.boss_ball_interference,
            scope="Contact causality until unlock; whole-run collisions independently audited",
            rule="Duck-ball impulse > .001 Ns; ball or previously struck pin → pin impulse > .0005 Ns; pin up < .35; no duck/boss pin hit before toppling or boss-ball interference",
        )


class ArcadeDriver:
    """Wrap existing finite-force machinery; gate stays locked until a strike."""

    def __init__(self, hazards):
        from .hazards import HazardDriver

        self.base = HazardDriver(hazards)
        self.hazards = self.base.hazards
        self.warnings = self.base.warnings
        self.released = self.base.released
        self.events = self.base.events
        self.bowling = BowlingLock()
        self.guide_impulse = {}
        self._boss_geom = None
        self._guide_geoms = {}

    def step(self, m, d, duck_x):
        self.base.step(m, d, duck_x)
        for guide, impulse in self.guide_impulse.items():
            if impulse > 0.02 and guide not in self.released:
                d.eq_active[m.equality(guide + "/hold").id] = False
                self.released.add(guide)
                self.events.append(
                    dict(
                        type="GUTTER_BREAK",
                        t=float(d.time),
                        hazard=guide,
                        impulse_Ns=impulse,
                    )
                )
        portal = next(h for h in self.hazards if h.kind == "finish_gate")
        target = 0.0 if self.bowling.unlocked and duck_x <= portal.x + 0.25 else -0.6
        d.ctrl[m.actuator("portal/motor").id] = target

    def sample(self, m, d):
        self.bowling.sample(m, d)
        if self._boss_geom is None:
            try:
                self._boss_geom = m.geom("boss/geom").id
            except KeyError:
                self._boss_geom = -1
            for guide in ["gutter_left", "gutter_right"]:
                try:
                    self._guide_geoms[m.geom(guide + "/geom").id] = guide
                except KeyError:
                    pass
        force = np.zeros(6)
        for i, c in enumerate(d.contact):
            if self._boss_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self._boss_geom else c.geom1
            guide = self._guide_geoms.get(other)
            if guide is None or guide in self.released:
                continue
            mujoco.mj_contactForce(m, d, i, force)
            self.guide_impulse[guide] = (
                self.guide_impulse.get(guide, 0)
                + max(0, float(force[0])) * m.opt.timestep
            )


def bowling_command(r, m, d):
    """Follow the observed ball while behind it, then aim for the exit roll."""
    from .skills import lane_command

    ball = d.body("playball").xpos
    target = float(np.clip(ball[1], -0.12, 0.12)) if r.trunk_pos()[0] < 5.9 else 0.0
    if r.trunk_pos()[0] < 5.15:
        target = 0.0
    r.bank.mirror_run = target - r.trunk_pos()[1] < -0.025
    return lane_command(
        r, target, 0.40 if abs(r.trunk_pos()[1] - target) > 0.045 else 0.55
    )


class EncounterCombo:
    def __init__(self):
        self.events = []
        self.verified = set()

    def certify(self, name, t, condition, evidence):
        if condition and name not in self.verified:
            self.verified.add(name)
            self.events.append(
                dict(
                    type="COMBO_VERIFIED",
                    t=float(t),
                    challenge=name,
                    combo=len(self.verified),
                    evidence=evidence,
                )
            )

    def sample(self, m, d, r, audit, gap, driver, results, level):
        h = {x.name: x for x in level.hazards}
        x = float(r.trunk_pos()[0])
        upright = bool(d.xmat[r.trunk_body_id, 8] > 0.9)
        success = lambda stage: any(
            s["stage"] == stage and s["status"] == "SUCCESS" for s in results
        )
        self.certify(
            "ROLL UNDER",
            d.time,
            success("ROLL") and x > h["bar"].x + 0.12 and "bar" not in audit.contacts,
            "completed >5.7 rad roll with foot support, crossed bar without force contact",
        )
        self.certify(
            "DROP DODGE",
            d.time,
            x > h["crate"].x + 0.22
            and upright
            and "crate" not in audit.contacts
            and "crate" in driver.released,
            "passed released crate plane upright without force contact",
        )
        self.certify(
            "VOID JUMP",
            d.time,
            gap.crossed and success("JUMP"),
            "full-rate airborne gap crossing and force-bearing landing",
        )
        self.certify(
            "SWEEPER SURVIVED",
            d.time,
            x > h["sweeper"].x + 0.4 and upright,
            dict(
                criterion="passed moving-arm plane upright",
                contact=audit.contacts.get("sweeper"),
            ),
        )
        self.certify(
            "BOWLING STRIKE",
            d.time,
            driver.bowling.unlocked,
            "duck-ball-pins force chain, all three pins toppled",
        )
        self.certify(
            "EXIT ROLL",
            d.time,
            success("EXIT_ROLL") and x > h["exit_bar"].x + 0.12,
            dict(
                criterion="second completed roll with foot support, crossed yielding exit bar",
                contact=audit.contacts.get("exit_bar"),
            ),
        )
