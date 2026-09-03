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
    frequency: float = 1.15          # swing pump rhythm (Hz)
    mode: str = "swing"              # "swing" (default, robust) | "rotate" (stretch) | "snake"
    rope_length: float = 0.55
    rope_density: float = 50.0
    turner_sep: float = 0.5
    turner_crane: float = 0.0        # head-up command for turners
    coach_gain: float = 0.25
    coach_rate_kp: float = 4.0
    sway: float = 0.012              # turner mouth ellipse radius (visual+pump)
    # jumper
    trigger_lead_s: float = 0.60     # jump trigger lead before the pass (hop takes ~0.6 s to lift)
    jump_crouch: float = 0.72
    jump_crouch_time: float = 0.18
    jump_extend_time: float = 0.07
    jump_flight_time: float = 0.24
    jump_land_time: float = 0.30
    jumper_offset: tuple = (0.0, -0.38)
    blind: bool = False            # True = hop periodically WITHOUT watching the rope (naive baseline)  # jumper waits outside the sweep, then enters
    never_jump_until: float = 0.0  # scene tool: jumper just stands there until t (the "hasn't learned yet" beat)
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
                                     gain=cfg.coach_gain, rate_kp=cfg.coach_rate_kp,
                                     mode=cfg.mode)
        cfgj = cfg
        if "jump" in policy_paths:
            # trained policy present — the real microduck.jump skill
            from .jump import PolicyJump
            self.jump = PolicyJump(self.ducks[j])
        else:
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
        if self.cfg.mode == "rotate":
            self._request_toss()
        else:
            self._tossed = True              # swing pumps up from rest
            self._windup_until = 0.0
            self.coach.toss(spin_mult=0.35)  # small seed perturbation
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
        # enter when the rope is live (real swing amplitude) AND the belly is
        # on the FAR side from the jumper (who approaches from -y)
        pos = j.trunk_pos()
        dist = math.hypot(pos[0], pos[1])
        if self._t > 2.2 and self._entry.target is None and amp > 0.10 and ang > 0.8:
            self._entry.go_to(0.0, 0.0)
        if self._entry.target is not None:
            self._entry.update()
            if dist < 0.035 or self._entry.done:
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
                and math.hypot(pos[0], pos[1]) > 0.05):
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

            # turner etiquette #2: rope fouled (coiled around a head)?
            # drop it, let it fall, pick it up, re-toss. Rope just dead?
            # re-toss in place.
            self._since_crossing = getattr(self, "_since_crossing", 0.0) + DT
            if coiled and not getattr(self, "_resetting", False):
                self._resetting = True
                self._reset_phase_t = self._t
                self.world.set_rope_latched(False)   # drop
                self._since_crossing = 0.0
            if getattr(self, "_resetting", False):
                dt_reset = self._t - self._reset_phase_t
                if dt_reset > 0.9:                   # rope fell to the floor
                    self.world.set_rope_latched(True)   # pick up
                    self._request_toss()
                    self._resetting = False
                    self._settle_wait = 0.5
            elif getattr(self, "_settle_wait", 0.0) > 0:
                self._settle_wait -= DT
            elif self._since_crossing > 3.5 and self._jumper_down_s == 0.0:
                # rope died — turners give it another go. Rotate gets the full
                # wind-up+toss; swing just takes a velocity kick (no lift).
                if self.cfg.mode == "rotate":
                    self._request_toss()
                else:
                    self.coach.toss(spin_mult=0.6)
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
            ang, amp = self.coach.belly_phase(at_x=float(jd.trunk_pos()[0]))
            self._entry_update(ang, amp)
            if self._entered and self.jump.state == "idle":
                self._recenter()
            if self._entered and self._t < cfg.never_jump_until:
                pass  # hasn't learned to jump yet — just stands and watches
            elif self._entered and cfg.blind:
                # naive baseline: hop on a fixed timer, no rope perception
                self._blind_next = getattr(self, "_blind_next", 0.0)
                if (self.jump.state == "idle" and jd.is_upright(0.45)
                        and self._t > self._blind_next):
                    self.jump.trigger()
                    self._blind_next = self._t + 0.55   # duck natural cadence ≠ rope tempo
            # --- unified pass detection (sensor world) → rhythm model ---
            crossed_now = False
            if self._prev_angle is not None:
                if cfg.mode == "rotate":
                    crossed_now = amp > 0.10 and self._prev_angle < 0 <= ang
                elif cfg.mode == "swing":
                    crossed_now = amp > 0.08 and ((self._prev_angle < 0 <= ang) or (self._prev_angle > 0 >= ang))
                elif cfg.mode == "snake":
                    y = float(np.mean([w.data.xpos[b][1] for b in w.rope_body_ids[len(w.rope_body_ids)//2-1:len(w.rope_body_ids)//2+1]]))
                    prev_y = getattr(self, "_prev_belly_y_j", y)
                    self._prev_belly_y_j = y
                    yrate = (y - prev_y) / DT
                    crossed_now = (prev_y * y < 0) and abs(yrate) > 0.03
            if crossed_now:
                pt = getattr(self, "_pass_times", [])
                pt.append(self._t)
                self._pass_times = pt[-6:]
            self._crossed_now = crossed_now

            if self._entered and not cfg.blind and self._prev_angle is not None and amp > 0.10:
                rate = ((ang - self._prev_angle + math.pi) % (2 * math.pi) - math.pi) / DT
                self._rate_lp_j = (0.85 * getattr(self, "_rate_lp_j", 0.0)
                                   + 0.15 * rate)
                rate_s = self._rate_lp_j
                if cfg.mode == "rotate":
                    if rate_s > 0.5:
                        # time until belly reaches bottom (angle wraps to 0 ahead)
                        dist = (-ang) % (2 * math.pi)
                        ttc = dist / rate_s
                        # EDGE trigger: fire once when ttc crosses the lead time
                        prev_ttc = getattr(self, "_prev_ttc", None)
                        self._prev_ttc = ttc
                        if (self.jump.state == "idle"
                                and jd.is_upright(0.45)
                                and prev_ttc is not None
                                and prev_ttc > cfg.trigger_lead_s >= ttc
                                and prev_ttc - ttc < 0.2):
                            self.jump.trigger()
                else:
                    # swing/snake: predict the next pass from the measured rhythm
                    if len(getattr(self, "_pass_times", [])) >= 3:
                        intervals = np.diff(self._pass_times)
                        P = float(np.median(intervals))
                        next_pass = self._pass_times[-1] + P
                        # the hop needs ~0.6 s from trigger to airborne — aim
                        # several passes ahead if the lead exceeds the interval
                        k = 0
                        while next_pass + k * P - self._t < cfg.trigger_lead_s - 0.02:
                            k += 1
                        ttg = next_pass + k * P - self._t
                        prev_ttg = getattr(self, "_prev_ttg", None)
                        self._prev_ttg = ttg
                        if (self.jump.state == "idle"
                                and jd.is_upright(0.45)
                                and prev_ttg is not None
                                and prev_ttg > cfg.trigger_lead_s >= ttg):
                            self.jump.trigger()
            self._prev_angle = ang

            # --- metrics updated in _judge_step (oracle side) ---

        self.jump.update(DT)
        for d in self.ducks.values():
            d.step()
        mujoco.mj_step(w.model, w.data, SUBSTEPS)
        self._t += DT
        self._judge_step()
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
        m = self.metrics
        m.duration_s = seconds
        m.turner_upright_frac = float(np.mean(self._up_h)) if self._up_h else 1.0
        m.jumper_upright_frac = float(np.mean(self._upj_h)) if self._upj_h else 1.0
        return m

    def _judge_step(self):
        """Oracle-side verdicts with a look-back/look-ahead window.

        A crossing is a skip iff at the belly's bottom pass the jumper was
        airborne above the rope AND the rope touched no part of the jumper in
        a ±0.2 s window AND the jumper is upright 0.5 s later. A trip is a
        rope-jumper contact in the window (or a fall right after). A pass
        with no contact and no jump attempt is a miss.
        """
        if not self._coach_on or not self._entered:
            return
        w = self.world
        jx = float(self.ducks[self.jumper_name].trunk_pos()[0])
        ang, amp = self.coach.belly_phase(at_x=jx)
        # --- track rope↔jumper contacts continuously (oracle evidence) ---
        sky_bodies = getattr(self, "_sky_bodies", None)
        if sky_bodies is None:
            m = w.model
            self._sky_bodies = {b for b in range(m.nbody)
                                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(self.jumper_name + "/")}
            self._rope_geoms = {g for g in range(m.ngeom)
                                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("rope/")}
            self._sky_bodies = self._sky_bodies
            sky_bodies = self._sky_bodies
        contact = False
        for ci in range(w.data.ncon):
            con = w.data.contact[ci]
            g1, g2 = con.geom1, con.geom2
            if g1 in self._rope_geoms or g2 in self._rope_geoms:
                b1 = w.model.geom_bodyid[g1]
                b2 = w.model.geom_bodyid[g2]
                if b1 in sky_bodies or b2 in sky_bodies:
                    contact = True
                    break
        if not hasattr(self, "_contact_hist"):
            self._contact_hist = []
            self._feet_hist = []
        self._contact_hist.append((self._t, contact))
        # keep 1 s of history
        self._contact_hist = [(t, c) for t, c in self._contact_hist if self._t - t < 1.0]
        # feet height history for windowed airborne check (50 Hz sampling would
        # miss the hop peak otherwise)
        if hasattr(self, "_feet_hist"):
            lz0, rz0 = self.jump.feet_z()
            self._feet_hist.append((self._t, min(lz0, rz0)))
            self._feet_hist = [(t, f) for t, f in self._feet_hist if self._t - t < 1.0]

        jd = self.ducks[self.jumper_name]
        # --- resolve pending verdicts (0.5 s after the crossing) ---
        if getattr(self, "_verdicts_due", None):
            due = [v for v in self._verdicts_due if self._t >= v["t_cross"] + 0.5]
            self._verdicts_due = [v for v in self._verdicts_due if self._t < v["t_cross"] + 0.5]
            for v in due:
                window_contact = any(c for t, c in self._contact_hist
                                     if abs(t - v["t_cross"]) <= 0.2)
                upright_after = jd.is_upright(0.45)
                attempted = v["attempted"]
                # airborne: peak feet height within ±0.12 s of the pass
                air_peak = max((f for t, f in getattr(self, "_feet_hist", [])
                                if abs(t - v["t_cross"]) <= 0.12), default=0.0)
                airborne = air_peak > 0.015
                if window_contact:
                    self.metrics.trips += 1
                    self._consec = 0
                elif attempted and airborne and upright_after:
                    self.metrics.successful_skips += 1
                    self.metrics.clearances.append(v["clearance"])
                    self._consec = getattr(self, "_consec", 0) + 1
                    self.metrics.consecutive_best = max(
                        self.metrics.consecutive_best, self._consec)
                else:
                    self.metrics.misses += 1
                    self._consec = 0

        # --- detect crossings, schedule verdict (event computed in step) ---
        crossed = getattr(self, "_crossed_now", False)
        self._crossed_now = False
        if crossed:
            if True:
                self._since_crossing = 0.0
                self.metrics.crossings += 1
                lz, rz = self.jump.feet_z()
                if not hasattr(self, "_verdicts_due"):
                    self._verdicts_due = []
                self._verdicts_due.append({
                    "t_cross": self._t,
                    "clearance": min(lz, rz),
                    "attempted": self.jump.state in ("crouch", "extend", "flight", "land"),
                })
        self._last_ang = ang
