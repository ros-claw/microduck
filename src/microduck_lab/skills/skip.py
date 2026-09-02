"""Rope skipping orchestration: 2 turners + 1 jumper, one physics world.

Layers:
- CoachRopeDriver: rope spin-up (toss) + rate regulation (declared training aid)
- RopeTurner ×2: mouths track the measured rope phase (real actuation, load-bearing)
- JumpSkill: scripted crouch-extend-tuck-land maneuver
- JumperController: closed-loop timing — watches the rope belly phase (sensor
  world) and triggers the jump so mid-flight coincides with the belly's bottom
  dead center crossing.

The tunable parameter surface (what ROSClaw Practice/Darwin adapts):
- trigger_lead_s: how long before the predicted bottom crossing to trigger
- jump params: crouch depth, phase durations
- rope frequency (team rhythm negotiation output)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..sim.composer import compose_world, DuckSpec, RopeSpec
from ..sim.runtime import PolicyBank, DuckRuntime, SUBSTEPS
from .rope_coach import CoachRopeDriver
from .rope_turn import RopeTurner
from .jump import JumpSkill

DT = 0.02  # policy step (50 Hz)


@dataclass
class SkipConfig:
    frequency: float = 1.45          # team rhythm (rope revs/s)
    rope_length: float = 0.55
    rope_density: float = 50.0
    turner_sep: float = 0.5
    turner_crane: float = 0.2        # head-up command for turners
    coach_gain: float = 0.25
    coach_rate_kp: float = 4.0
    sway: float = 0.012              # turner mouth ellipse radius (visual+pump)
    # jumper
    trigger_lead_s: float = 0.34     # jump trigger lead before bottom crossing
    jump_crouch: float = 0.72
    jump_crouch_time: float = 0.18
    jump_extend_time: float = 0.07
    jump_flight_time: float = 0.24
    jump_land_time: float = 0.30
    jumper_offset: tuple = (0.0, -0.38)  # jumper waits outside the sweep, then enters
    seed: int = 0                      # evaluation seed (domain randomization)


@dataclass
class SkipMetrics:
    crossings: int = 0
    successful_skips: int = 0
    trips: int = 0
    misses: int = 0                  # crossing while jumper not attempting
    clearances: list = field(default_factory=list)
    consecutive_best: int = 0
    jumper_upright_frac: float = 1.0
    turner_upright_frac: float = 1.0
    jumper_max_down_s: float = 0.0   # longest continuous time jumper stayed down
    turner_max_down_s: float = 0.0
    duration_s: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successful_skips / max(1, self.crossings)


class RopeSkipSession:
    """One physical episode: rope spinning, jumper skipping, metrics recorded."""

    def __init__(self, robot_xml, policy_paths: dict, cfg: SkipConfig,
                 duck_names=("lavender", "cream", "sky"), playground=False):
        self.cfg = cfg
        # --- domain randomization per seed (Darwin: holdout seeds unseen) ---
        rng = np.random.default_rng(cfg.seed)
        jit = rng.uniform(-0.02, 0.02, size=2)          # jumper position jitter
        freq_jit = float(rng.uniform(0.97, 1.03))       # rope tempo jitter
        fric_jit = float(rng.uniform(0.03, 0.08))       # floor-rope friction
        ta, tb, j = duck_names
        cx, cy = cfg.jumper_offset[0] + jit[0], cfg.jumper_offset[1] + jit[1]
        from ..sim.composer import DUCK_COLORS
        ducks = [
            DuckSpec(ta, (-cfg.turner_sep / 2, 0.0), yaw=0.0,
                     color=DUCK_COLORS.get(ta)),
            DuckSpec(tb, (cfg.turner_sep / 2, 0.0), yaw=np.pi,
                     color=DUCK_COLORS.get(tb)),
            DuckSpec(j, (cx, cy), yaw=np.pi / 2,
                     color=DUCK_COLORS.get(j)),   # jumper faces rope plane
        ]
        self.world = compose_world(
            robot_xml, ducks,
            rope=RopeSpec(ta, tb, length=cfg.rope_length, count=30, density=cfg.rope_density),
            playground=playground, rope_height=0.19)
        # smooth gym floor under the rope
        for g in range(self.world.model.ngeom):
            nm = mujoco.mj_id2name(self.world.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm and nm.startswith("rope/"):
                self.world.model.geom_friction[g] = [fric_jit, 0.005, 0.0001]

        self.bank = PolicyBank(policy_paths)
        self.ducks = {
            d.name: DuckRuntime(self.world.model, self.world.data, self.bank,
                                prefix=f"{d.name}/", name=d.name)
            for d in ducks
        }
        self.turner_names = (ta, tb)
        self.jumper_name = j
        self.coach = CoachRopeDriver(self.world, frequency=cfg.frequency * freq_jit,
                                     gain=cfg.coach_gain, rate_kp=cfg.coach_rate_kp)
        cfgj = cfg
        self.jump = JumpSkill(
            self.ducks[j], crouch_depth=cfgj.jump_crouch,
            crouch_time=cfgj.jump_crouch_time, extend_time=cfgj.jump_extend_time,
            flight_time=cfgj.jump_flight_time, land_time=cfgj.jump_land_time)
        self.turners = None
        self._t = 0.0
        self.metrics = SkipMetrics()
        self._prev_angle = None
        self._up_h = []
        self._upj_h = []
        self._pending_crossing = None   # scheduled verdict check
        self._turner_phase = 0.0
        self._coach_on = False

    # ------------------------------------------------------------------ setup
    def settle(self, seconds: float = 2.0):
        """Ducks walk-pose settle; turners sit down and crane heads up."""
        for _ in range(int(seconds * 50)):
            for d in self.ducks.values():
                d.step()
            mujoco.mj_step(self.world.model, self.world.data, SUBSTEPS)
        for name in self.turner_names:
            self.ducks[name].active_policy = "sitstand"
            self.ducks[name].set_command(twist=(1, 0, 0),
                                         head=(-self.cfg.turner_crane,) * 2 + (0, 0))
        for _ in range(int(2.0 * 50)):
            for d in self.ducks.values():
                d.step()
            mujoco.mj_step(self.world.model, self.world.data, SUBSTEPS)

    def start_rope(self):
        self.coach.start()
        self._request_toss()
        self.turners = [RopeTurner(self.ducks[n], self.cfg.sway, radius_z=0.008)
                        for n in self.turner_names]
        for t in self.turners:
            t.start()
        self._coach_on = True
        # the jumper waits outside the rope's sweep and enters when the spin
        # is up to tempo — real skippers never stand inside a starting rope
        from .navigate import WalkTo
        j = self.ducks[self.jumper_name]
        j.active_policy = "walk"
        self._entry = WalkTo(j)
        self._entered = False

    def _entry_update(self, ang, amp):
        """Walk the jumper in once the rope is up to tempo."""
        j = self.ducks[self.jumper_name]
        if self._entered:
            return True
        # time-based entry: the toss + spin-up take ~2 s; walk in once the
        # rope has had time to establish (phase-gated when we do have tempo)
        pos = j.trunk_pos()
        dist = math.hypot(pos[0], pos[1])
        if self._t > 2.2 and self._entry.target is None:
            self._entry.go_to(0.0, 0.0)
        if self._entry.target is not None:
            self._entry.update()
            if dist < 0.07 or self._entry.done:
                j.active_policy = "stand"
                j.set_command(twist=(0, 0, 0))
                self._entered = True
                return True
        return False

    def _recenter(self):
        """A skipper who drifted (or got knocked) off the middle walks back."""
        j = self.ducks[self.jumper_name]
        pos = j.trunk_pos()
        if (self.jump.state == "idle" and j.is_upright(0.45)
                and math.hypot(pos[0], pos[1]) > 0.12):
            if not hasattr(self, "_recenter_nav"):
                from .navigate import WalkTo
                self._recenter_nav = WalkTo(j)
            if self._recenter_nav.target is None or self._recenter_nav.done:
                self._recenter_nav.go_to(0.0, 0.0)
            self._recenter_nav.update()
            return True
        return False

    def _request_toss(self):
        """Wind-up → toss: lift the belly off the floor first, then spin."""
        self._windup_until = self._t + 0.7
        self._tossed = False

    # ------------------------------------------------------------------- run
    def step(self):
        cfg = self.cfg
        w = self.world
        if self._coach_on:
            # real-world etiquette: when the jumper trips, the turners ease off
            # the rope and wait; resume (re-toss) once they're back up.
            jd = self.ducks[self.jumper_name]
            jup = jd.is_upright(0.45)
            self._jumper_down_s = 0.0 if jup else getattr(self, "_jumper_down_s", 0.0) + DT
            if self._jumper_down_s > 1.2 and self.coach.enabled:
                self.coach.enabled = False           # ease off
                w.data.xfrc_applied[:] = 0.0
                self._rope_paused = True
            elif getattr(self, "_rope_paused", False) and self._jumper_down_s == 0.0:
                self.coach.enabled = True            # resume
                self._request_toss()
                self._rope_paused = False
            if self.coach.enabled:
                if self._t < getattr(self, "_windup_until", 0.0):
                    self.coach.windup_lift()
                else:
                    if not getattr(self, "_tossed", True):
                        self.coach.toss()
                        self._tossed = True
                    self.coach.update(DT)
            ang, amp = self.coach.belly_phase()
            coiled = getattr(self.coach, "_coil_hold", 0.0) > 1.2

            # turner etiquette #2: if the rope fouls (coils) or dies, the
            # turners stop winding, wait for it to settle, and re-toss.
            self._since_crossing = getattr(self, "_since_crossing", 0.0) + DT
            if coiled:  # fouled — stop winding
                pass
                self._since_crossing = 0.0
                self._settle_wait = 1.0
            elif getattr(self, "_settle_wait", 0.0) > 0:
                self._settle_wait -= DT
            elif self._since_crossing > 3.5 and self._jumper_down_s == 0.0:
                self._request_toss()               # "one more time!"
                self._since_crossing = 0.0
            # mouths track rope phase with 90° lead (real contribution) —
            # but stop winding when the rope is fouled, or they'd coil it more
            if not coiled and getattr(self, "_settle_wait", 0.0) <= 0:
                ph_target = ang + math.pi / 2
                self._turner_phase += np.clip(
                    (ph_target - self._turner_phase + math.pi) % (2 * math.pi) - math.pi,
                    -0.3, 0.3)
            s = min(1.0, self._t / 2.0)
            for t in self.turners:
                t.update(self._turner_phase, s)

            # --- jumper entry + timing (sensor world: belly phase only) ---
            jd = self.ducks[self.jumper_name]
            self._entry_update(ang, amp)
            if self._entered and self.jump.state == "idle":
                self._recenter()
            if self._entered and self._prev_angle is not None and amp > 0.10:
                rate = ((ang - self._prev_angle + math.pi) % (2 * math.pi) - math.pi) / DT
                if rate > 0.5:
                    # time until belly reaches bottom (angle wraps to 0 ahead)
                    dist = (-ang) % (2 * math.pi)
                    ttc = dist / rate
                    # trigger so mid-flight hits the crossing
                    if (self.jump.state == "idle"
                            and jd.is_upright(0.45)
                            and abs(ttc - cfg.trigger_lead_s) < DT * 1.5):
                        self.jump.trigger()
            self._prev_angle = ang

            # --- metrics updated in _judge_step (oracle side) ---

        self.jump.update(DT)
        for d in self.ducks.values():
            d.step()
        mujoco.mj_step(w.model, w.data, SUBSTEPS)
        self._t += DT
        if self._coach_on and self._t > 3:
            up_t = all(self.ducks[n].is_upright(0.4) for n in self.turner_names)
            up_j = self.ducks[self.jumper_name].is_upright(0.4)
            self._up_h.append(up_t)
            self._upj_h.append(up_j)
            # track longest continuous down-time (recovery-aware safety metric)
            self._down_t = 0.0 if up_t else getattr(self, "_down_t", 0.0) + DT
            self._down_j = 0.0 if up_j else getattr(self, "_down_j", 0.0) + DT
            self.metrics.turner_max_down_s = max(self.metrics.turner_max_down_s, self._down_t)
            self.metrics.jumper_max_down_s = max(self.metrics.jumper_max_down_s, self._down_j)

    def run_episode(self, seconds: float = 14.0, render_cb=None) -> SkipMetrics:
        steps = int(seconds * 50)
        self._last_ang = None
        self._crossing_active = None
        for i in range(steps):
            self.step()
            if render_cb and i % 2 == 0:
                render_cb(self)
            self._judge_step()
        m = self.metrics
        m.duration_s = seconds
        m.turner_upright_frac = float(np.mean(self._up_h)) if self._up_h else 1.0
        m.jumper_upright_frac = float(np.mean(self._upj_h)) if self._upj_h else 1.0
        return m

    def _judge_step(self):
        """Oracle-side crossing detection + success/trip judgement.

        belly_phase() returns atan2(dy, -dz) ∈ (-π, π]; bottom dead center = 0.
        Forward rotation ⇒ angle increases through 0 at each crossing.
        """
        if not self._coach_on or not self._entered:
            return
        ang, amp = self.coach.belly_phase()
        if self._last_ang is not None and amp > 0.12:
            if self._last_ang < 0 <= ang:
                self._since_crossing = 0.0
                self.metrics.crossings += 1
                jd = self.ducks[self.jumper_name]
                lz, rz = self.jump.feet_z()
                clearance = min(lz, rz)
                if self.jump.state in ("flight", "extend") or clearance > 0.015:
                    self.metrics.successful_skips += 1
                    self.metrics.clearances.append(clearance)
                    self._consec = getattr(self, "_consec", 0) + 1
                    self.metrics.consecutive_best = max(
                        self.metrics.consecutive_best, self._consec)
                else:
                    self.metrics.trips += 1
                    self._consec = 0
        self._last_ang = ang
