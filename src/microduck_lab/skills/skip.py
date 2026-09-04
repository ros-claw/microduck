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
import re
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
    rope_mode: str = "free"          # "free" (chain+connect) | "mocap" (driven carriers, deterministic loop)
    mocap_radius: float = 0.11       # drive-circle radius for mocap rope (the "handle" amplification)
    mocap_pattern: str = "circle"    # "circle" (full loop) | "pendulum" (side-swing) | "snake" (low floor sweep, turners park&release)
    mocap_z: float = 0.05            # carrier height for snake (turners hold the rope low to the floor)
    snake_amplitude: float = 0.16    # lateral sweep amplitude for snake (m)
    wait_pos: tuple = (0.0, -0.06)   # jumper ready position (snake parks away from it)
    rope_flight_s: float = 0.90      # snake: target rope release→pass time (the hop apex)
    rope_length: float = 0.55
    rope_density: float = 50.0
    turner_sep: float = 0.5
    turner_crane: float = 0.0        # head-up command for turners
    coach_gain: float = 0.25
    coach_rate_kp: float = 4.0
    sway: float = 0.012              # turner mouth ellipse radius (visual+pump)
    # jumper
    trigger_lead_s: float = 0.60     # jump trigger lead before the pass (hop takes ~0.6 s to lift)
    jump_crouch: float = 0.55        # 0.55 + instant handoff = 5.5 cm hop, 100% upright landings
    jump_crouch_time: float = 0.22   # (the extension needs the full 0.10 s — shorter
    jump_extend_time: float = 0.10   #  extension = weak push, the duck just walks forward)
    jump_flight_time: float = 0.01   # hand to the stand policy right after toe-off;
    jump_land_time: float = 0.01     # it flies the body level and sticks the landing
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
            playground=playground, rope_height=0.19, grippy_ducks=[], rope_mode=cfg.rope_mode)
        # Grippy PU soles are applied DYNAMICALLY: on only while the jumper is
        # actually hopping (the launch needs them), off while standing/walking
        # (the soft grippy sole makes the stand policy drift ~14 cm/s — the
        # duck would wander out of the rope between hops). Like a real skipper
        # digging in only for the takeoff.
        m0 = self.world.model
        self._jumper_soles = [g for g in range(m0.ngeom)
                              if (lambda nm: nm and nm.startswith(j + "/") and
                                  re.search(r"(left|right)_foot_collision$", nm))(
                                  mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        self._sole_normal = [(float(m0.geom_friction[g, 0]), m0.geom_solref[g].copy())
                             for g in self._jumper_soles]
        self._grip_on = False
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
            crane = self.cfg.turner_crane if self.cfg.mode != "rotate" else 0.25
            self.ducks[name].set_command(twist=(1, 0, 0),
                                         head=(-crane,) * 2 + (0, 0))
        for _ in range(int(2.0 * 50)):
            for d in self.ducks.values():
                d.step()
            mujoco.mj_step(self.world.model, self.world.data, SUBSTEPS)

    def start_rope(self):
        self.coach.start()
        if self.cfg.rope_mode == "mocap":
            # deterministic driven loop: mocap carriers drive the rope ends.
            # No coach force needed — the driver IS the rope motion.
            from ..sim.stiff_rope import StiffRope
            self._stiff = StiffRope(self.world.model, self.world.data,
                                    "rope/carryA", "rope/carryB")
            self._stiff_phase = 0.0
            # anchor circles at the measured mouth positions (post-settle)
            self._stiff_cA = self.ducks[self.turner_names[0]].site_pos("mouth_tip").copy()
            self._stiff_cB = self.ducks[self.turner_names[1]].site_pos("mouth_tip").copy()
            if self.cfg.mocap_pattern == "snake":
                # snake: turners hold the rope LOW and sweep it along the floor.
                # Carriers anchor at the turner BODY x (±0.25), not the mouths:
                # the mouth span (0.33 m) leaves 19 cm of slack rope that piles
                # on the floor and kills the sweep (measured). A taut rope
                # (L=0.52 over the 0.50 m span) sweeps cleanly past the sitting
                # turners (their feet sprawl forward, clear of the line).
                # Start parked at the +y extreme, away from the jumper's spot.
                self._stiff_cA0 = self._stiff_cA.copy()
                self._stiff_cB0 = self._stiff_cB.copy()
                self._stiff_cA = np.array([*self.ducks[self.turner_names[0]].trunk_pos()[:2] * 1.0,
                                           self.cfg.mocap_z])
                self._stiff_cB = np.array([*self.ducks[self.turner_names[1]].trunk_pos()[:2] * 1.0,
                                           self.cfg.mocap_z])
                self._stiff_phase = math.pi / 2
                self._snake_parked = True
                self._snake_armed = True       # parked → a departure can fire the hop
                self._snake_mid_y = None
                self._snake_rate = 0.0
        if self.cfg.mode == "rotate":
            # toss the resting rope DIRECTLY (the windup-lift releases it from a
            # moving held state, which kills the rotation — measured).
            self.coach.toss()
            self._tossed = True
            self._windup_until = 0.0
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
        self._entry.arrive_tol = 0.03
        self._entered = False
        self._wait_pos = (float(j.trunk_pos()[0]), float(j.trunk_pos()[1]))

    def _skip_spot(self):
        """Where the jumper should stand to skip (snake: opposite the rope park)."""
        if self.cfg.mocap_pattern == "snake":
            dest = getattr(self, "_snake_dest", 1)
            return (0.0, -dest * 0.10)
        return (0.0, 0.0)

    def _entry_update(self, ang, amp):
        """Walk the jumper in once the rope is up to tempo."""
        j = self.ducks[self.jumper_name]
        if self._entered:
            return True
        # enter when the rope is live (real swing amplitude) AND the belly is
        # on the FAR side from the jumper (who approaches from -y)
        pos = j.trunk_pos()
        spot = self._skip_spot()
        dist = math.hypot(pos[0] - spot[0], pos[1] - spot[1])
        if self.cfg.mocap_pattern == "snake":
            # rope starts parked away from the wait spot — entry is safe at once
            rope_live = self._t > 1.0
        else:
            rope_live = (amp > 0.08 and
                         (abs(ang) > 2.0 or self.cfg.mocap_pattern == "pendulum"))
        if self._entry.target is None and ((self._t > 2.2 and rope_live)
                                           or self._t > 5.0
                                           or (self.cfg.mocap_pattern == "snake" and rope_live)):
            self._entry.go_to(*spot)
        elif self._entry.target is None:
            # hold the wait position against the stand policy's drift
            wx, wy = self._wait_pos
            if math.hypot(pos[0] - wx, pos[1] - wy) > 0.05:
                self._entry.go_to(wx, wy)
        if self._entry.target is not None:
            self._entry.update()
            if dist < 0.035 or self._entry.done:
                j.active_policy = "stand"
                j.set_command(twist=(0, 0, 0))
                self._entered = True
                return True
        return False

    def _recenter(self):
        """A skipper who drifted (or got knocked) off the spot walks back."""
        j = self.ducks[self.jumper_name]
        pos = j.trunk_pos()
        spot = self._skip_spot()
        if (self.jump.state in ("idle",) and j.is_upright(0.45)
                and math.hypot(pos[0] - spot[0], pos[1] - spot[1]) > 0.04):
            if not hasattr(self, "_recenter_nav"):
                from .navigate import WalkTo
                self._recenter_nav = WalkTo(j)
            if self._recenter_nav.target is None or self._recenter_nav.done:
                self._recenter_nav.go_to(*spot)
            self._recenter_nav.update()
            return True
        return False

    def _try_jump(self):
        """Trigger the hop; dig the soles in for the takeoff+landing window.
        (Grip toggling only applies to the trained PolicyJump — and the policy
        needs the grippy-converged stand state, so this is currently disabled;
        the procedural JumpSkill hops 7.8 cm on stock soles.)"""
        if self.jump.trigger():
            self._last_jump_t = self._t
            return True
        return False

    def _set_grip(self, on: bool):
        """Dig-in soles for the takeoff window only (see __init__ note)."""
        if on == self._grip_on:
            return
        m0 = self.world.model
        for g, (mu, solref) in zip(self._jumper_soles, self._sole_normal):
            if on:
                m0.geom_friction[g, 0] = 2.0
                m0.geom_solref[g] = [0.04, 1.0]
            else:
                m0.geom_friction[g, 0] = mu
                m0.geom_solref[g] = solref
        self._grip_on = on

    def _request_toss(self):
        """Wind-up → toss: lift the belly off the floor first, then spin."""
        self._windup_until = self._t + 0.7
        self._tossed = False

    def _snake_step(self):
        """Snake mode: the turners sweep the rope along the FLOOR side to side
        (a real beginner variant) and PARK it at the extreme while the jumper
        repositions, releasing when it's ready — patient turners, like for a
        kid skipper. Every pass is therefore a genuine attempt.

        Geometry (measured): the hop is a forward LEAP (~20 cm) — so the duck
        always jumps TOWARD the oncoming rope, landing on the side the rope
        came from, which the sweep never revisits. The duck ping-pongs between
        the two sides, turning to face the parked rope each cycle.
        """
        cfg = self.cfg
        w = self.world
        jd = self.ducks[self.jumper_name]

        # --- sensor world: rope middle lateral position + rate ---
        mid_y = self._stiff.mid_y()
        prev_mid = self._snake_mid_y if self._snake_mid_y is not None else mid_y
        raw_rate = (mid_y - prev_mid) / DT
        self._snake_rate = 0.8 * self._snake_rate + 0.2 * raw_rate
        self._snake_mid_y = mid_y

        pos = jd.trunk_pos()
        # turner etiquette #3: the jumper is DOWN — lift the rope up off it so
        # the stand policy can recover (a fallen duck tangled in the rope never
        # gets up; measured). Lift while down, lower once upright.
        if not jd.is_upright(0.45):
            self._jdown = getattr(self, "_jdown", 0.0) + DT
        else:
            self._jdown = 0.0
        lift_target = 0.16 if self._jdown > 1.0 else cfg.mocap_z
        self._snake_z = getattr(self, "_snake_z", cfg.mocap_z)
        self._snake_z += float(np.clip(lift_target - self._snake_z, -DT*0.08, DT*0.08))

        # carrier ramp: mouths → low snake hold over the first 1.5 s
        a = min(1.0, self._t / 1.5)
        a = a * a * (3 - 2 * a)          # smoothstep
        cA = (1 - a) * self._stiff_cA0 + a * self._stiff_cA
        cB = (1 - a) * self._stiff_cB0 + a * self._stiff_cB
        cA = cA.copy(); cB = cB.copy()
        cA[2] = cB[2] = self._snake_z

        # --- where should the jumper be? Opposite the rope's park side,
        # facing it (so the leap carries it toward the oncoming rope). ---
        dest = getattr(self, "_snake_dest", 1)
        spot = (0.0, -dest * 0.10)
        face_yaw = dest * math.pi / 2

        # --- jumper navigation: walk to spot, then turn to face the rope ---
        self._entry_update(0.0, 0.2)
        pos = jd.trunk_pos()
        nav_busy = False
        if self._entered and self.jump.state == "idle":
            nav_busy = self._snake_reposition(spot, face_yaw)
        ready_now = (self._entered and self.jump.state == "idle" and not nav_busy
                     and jd.is_upright(0.45)
                     and math.hypot(pos[0] - spot[0], pos[1] - spot[1]) < 0.05)
        # require a settle dwell at LOW VELOCITY: a hop from a creeping duck
        # comes out a weak stagger-step (measured: 2.8 cm vs 5.5 cm from still).
        lv = np.linalg.norm(jd.trunk_linvel()[:2]) if hasattr(jd, "trunk_linvel") else 0.0
        still = lv < 0.05
        if ready_now and still and getattr(self, "_ready_since", None) is None:
            self._ready_since = self._t
        elif not (ready_now and still):
            self._ready_since = None
        ready = ready_now and still and (self._t - getattr(self, "_ready_since", self._t)) > 0.5
        if ready:
            # hop-quality gate: fire only when the stand policy is near its
            # canonical pose (legs near default, trunk level). The hop apex is
            # chaotic in the micro-state (3-7 cm); this SELECTS the good ones.
            obs = jd.get_obs()
            legs_near_default = bool(np.max(np.abs(obs[6:20])) < 0.15)
            level = bool(obs[5] < -0.985)     # projected gravity z ≈ -1
            ready = ready and legs_near_default and level

        # --- turner coordination: the rope PARKS at an extreme while the jumper
        # repositions. DUCK-LEAD release (real cooperative timing — "ready...
        # NOW"): the jumper crouches when IT is ready; the turners see the
        # crouch and release; the sweep's ~0.9 s flight to the duck lands on
        # the hop's apex (~0.9 s). Every pass is a genuine attempt. ---
        parked = getattr(self, "_snake_parked", True)
        ph = self._stiff_phase
        hopping = self.jump.state != "idle"
        if parked and hopping:
            parked = False                  # release! (turners saw the crouch)
            self._sweep_t0 = self._t
            self._sweep_mid0 = mid_y
            self._sweep_target_y = getattr(self, "_takeoff_y", spot[1])
        if not parked:
            # closed-loop arrival servo: the turners speed up / slow down the
            # sweep on the MEASURED rope position so the rope reaches the duck
            # exactly rope_flight_s after the release (= the hop apex).
            # Ease-in over 0.45 s: a jerked release WHIPS the rope (it coasts
            # through faster than the carriers and the servo can't slow it).
            t_rel = getattr(self, "_sweep_t0", self._t)
            ease = min(1.0, (self._t - t_rel) / 0.45)
            ease = ease * ease * (3 - 2 * ease)
            T = cfg.rope_flight_s
            mid0 = getattr(self, "_sweep_mid0", mid_y)
            ty = getattr(self, "_sweep_target_y", spot[1])
            dirn = float(np.sign(ty - mid0) or 1.0)
            dist_total = abs(ty - mid0) + 0.05        # aim a touch past the line
            p_actual = (mid_y - mid0) * dirn          # grows as the rope nears
            p_des = min(1.15, (self._t - t_rel) / T) * dist_total
            err = p_des - p_actual                    # >0: rope behind schedule
            rate_scale = float(np.clip(1.0 + 2.0 * err / max(dist_total, 0.05), 0.30, 2.0))
            self._snake_rate_scale = 0.7 * getattr(self, "_snake_rate_scale", 1.0) + 0.3 * rate_scale
            advance = 2 * math.pi * cfg.frequency * DT * self._snake_rate_scale * ease
            k = math.ceil((ph - math.pi / 2 + 1e-6) / math.pi)
            nxt = math.pi / 2 + k * math.pi
            new = ph + advance
            if new >= nxt:
                new = nxt
                parked = True               # park at the opposite extreme
            self._stiff_phase = new % (2 * math.pi)
        self._snake_parked = parked
        self._stiff.place(self._stiff_phase, cfg.snake_amplitude,
                          cA, cB, pattern="pendulum")
        # turners' mouths track the rope ends (they hold it)
        ph_vis = self._stiff_phase + math.pi / 2
        s_ = min(1.0, self._t / 2.0)
        for t in self.turners:
            t.update(ph_vis, s_)

        # --- duck's jump trigger: the rope is parked (sensor: |mid_y| high,
        # |rate| ~0 sustained) and the duck is settled → crouch NOW ---
        parked_sensor = abs(mid_y) > 0.11 and abs(self._snake_rate) < 0.03
        self._parked_for = (getattr(self, "_parked_for", 0.0) + DT) if parked_sensor else 0.0
        if ready and self._parked_for > 0.25 and self.jump.state == "idle":
            if self._try_jump():
                self._takeoff_y = float(pos[1])

        # --- crossing event for the judge: rope middle passes the takeoff line,
        # only during a real sweep (never while parked — the rope settling or
        # the duck brushing the pile must not flip the spot or count a pass).
        dy = getattr(self, "_takeoff_y", spot[1])
        if (not self._snake_parked
                and (prev_mid - dy) * (mid_y - dy) < 0
                and self._t - getattr(self, "_last_snake_cross", -9) > 0.8):
            self._crossed_now = True
            self._last_snake_cross = self._t
            # the pass is decided — the jumper now prepares for the new park
            # side (it's landing there already): flip the target spot/facing
            self._snake_dest = -dest

        self._judge_step()
        self._t += DT
        prev_state = self.jump.state
        self.jump.update(DT)
        # detect hop completion → brief no-walk settle so the landing settles
        if prev_state != "idle" and self.jump.state == "idle":
            self._land_settle_until = self._t + 0.6
        for d in self.ducks.values():
            d.step()
        mujoco.mj_step(w.model, w.data, SUBSTEPS)
        # metrics bookkeeping (mirrors the main step path)
        if self._t > 3:
            up_t = all(self.ducks[n].is_upright(0.4) for n in self.turner_names)
            up_j = jd.is_upright(0.4)
            self._up_h.append(up_t)
            self._upj_h.append(up_j)
            self._down_t = 0.0 if up_t else getattr(self, "_down_t", 0.0) + DT
            self._down_j = 0.0 if up_j else getattr(self, "_down_j", 0.0) + DT
            self.metrics.turner_max_down_s = max(self.metrics.turner_max_down_s, self._down_t)
            self.metrics.jumper_max_down_s = max(self.metrics.jumper_max_down_s, self._down_j)

    def _snake_reposition(self, spot, face_yaw) -> bool:
        """Walk to the spot, then turn to face the parked rope. Returns True
        while still moving (not settled). Never walks/turns a wobbling duck —
        stepping during a landing transient compounds into a fall (measured:
        the early-session deaths)."""
        from .navigate import WalkTo, TurnTo
        j = self.ducks[self.jumper_name]
        if not hasattr(self, "_repo_nav"):
            self._repo_nav = WalkTo(j)
            self._repo_turn = TurnTo(j)
        pos = j.trunk_pos()
        dist = math.hypot(pos[0] - spot[0], pos[1] - spot[1])
        yaw_err = abs((face_yaw - j.trunk_yaw() + math.pi) % (2 * math.pi) - math.pi)
        # post-landing settle ONLY: hold still for 0.6 s right after a hop so
        # the landing transient dies before walking (stepping on a wobbling
        # duck compounds into a fall). After that, walk freely — a velocity
        # gate here would abort the walk as soon as it builds speed (measured:
        # the duck once crept for 27 s without arriving). A fallen duck never
        # walks (the stand policy recovers it first).
        if self._t < getattr(self, "_land_settle_until", 0.0) or not j.is_upright(0.45):
            if j.active_policy != "stand":
                j.active_policy = "stand"
                j.set_command(twist=(0, 0, 0))
            return True
        if dist > 0.05:
            if self._repo_nav.target is None or self._repo_nav.done:
                self._repo_nav.arrive_tol = 0.04
                self._repo_nav.go_to(*spot)
            self._repo_nav.update()
            return True
        # facing with hysteresis: start a turn at >0.40 rad, accept at <0.30
        # (the duck's stand yaw settles ~0.26 rad off — without hysteresis the
        # turn start/cancel churns forever and readiness never holds)
        if not self._repo_turn.done:
            if yaw_err < 0.30:
                self._repo_turn.done = True
                self._repo_turn.target_yaw = None
            else:
                self._repo_turn.update()
                return True
        elif yaw_err > 0.40:
            self._repo_turn.turn_to(face_yaw)
            self._repo_turn.update()
            return True
        # hold: stand still
        if j.active_policy != "stand":
            j.active_policy = "stand"
            j.set_command(twist=(0, 0, 0))
        return False

    # ------------------------------------------------------------------- run
    def step(self):
        cfg = self.cfg
        w = self.world
        if self._coach_on and self.cfg.rope_mode == "mocap":
            pendulum = self.cfg.mocap_pattern == "pendulum"
            snake = self.cfg.mocap_pattern == "snake"
            if snake:
                self._snake_step()
                return
            # deterministic driven loop: move the carriers, mouths track visually
            self._stiff_phase = self._stiff.drive(
                self._stiff_phase, self.cfg.mocap_radius,
                self._stiff_cA, self._stiff_cB, self.cfg.frequency, DT,
                pattern=self.cfg.mocap_pattern)
            # turners' mouths follow the rope ends (they hold it)
            ang_r, amp_r, _ = self._stiff.belly()
            ph = self._stiff_phase + math.pi / 2
            s_ = min(1.0, self._t / 2.0)
            for t in self.turners:
                t.update(ph, s_)
            ang, amp = ang_r, amp_r
            self._stiff_prev_phase = self._stiff_phase
            # detect the belly crossing the bottom via the measured belly angle
            if self._prev_angle is not None:
                if pendulum:
                    # side-swing: belly passes under the duck in BOTH directions
                    self._crossed_now = (amp > 0.08 and
                                         ((self._prev_angle < 0 <= ang) or
                                          (self._prev_angle > 0 >= ang)))
                else:
                    self._crossed_now = (amp > 0.08 and
                                         self._prev_angle < 0 <= ang)
            if getattr(self, "_crossed_now", False):
                pt = getattr(self, "_pass_times", [])
                pt.append(self._t)
                self._pass_times = pt[-6:]
            # jump trigger on the measured rope phase (sensor world)
            jd = self.ducks[self.jumper_name]
            self._entry_update(ang, amp)
            if self._entered and self.jump.state == "idle":
                self._recenter()
            if self._entered and self._prev_angle is not None and amp > 0.10:
                if pendulum:
                    # rhythm model from measured pass times: the pendulum's two
                    # passes per period are NOT evenly spaced (rope dynamics lag
                    # the drive), so predict the next interval as
                    # period − last_interval (period = pass[t-1] − pass[t-3]).
                    pt = getattr(self, "_pass_times", [])
                    if len(pt) >= 4:
                        period = pt[-1] - pt[-3]
                        last_iv = pt[-1] - pt[-2]
                        next_pass = pt[-1] + (period - last_iv)
                        k = 0
                        while next_pass + k * (period / 2) - self._t < cfg.trigger_lead_s - 0.02:
                            k += 1
                        ttg = next_pass + k * (period / 2) - self._t
                        prev_ttg = getattr(self, "_prev_ttg", None)
                        self._prev_ttg = ttg
                        if (self.jump.state == "idle" and jd.is_upright(0.45)
                                and prev_ttg is not None
                                and prev_ttg > cfg.trigger_lead_s >= ttg):
                            if self._try_jump():
                                pass
                else:
                    rate = ((ang - self._prev_angle + math.pi) % (2 * math.pi) - math.pi) / DT
                    self._rate_lp_j = (0.85 * getattr(self, "_rate_lp_j", 0.0) + 0.15 * rate)
                    rate_s = self._rate_lp_j
                    if rate_s > 0.5:
                        dist = (-ang) % (2 * math.pi)
                        ttc = dist / rate_s
                        prev_ttc = getattr(self, "_prev_ttc", None)
                        self._prev_ttc = ttc
                        if (self.jump.state == "idle" and jd.is_upright(0.45)
                                and prev_ttc is not None
                                and prev_ttc > cfg.trigger_lead_s >= ttc
                                and prev_ttc - ttc < 0.2):
                            if self._try_jump():
                                pass
            self._prev_angle = ang
            self._judge_step()
            self._t += DT
            self.jump.update(DT)
            if self.jump.state == "idle":
                self._set_grip(False)
            for d in self.ducks.values():
                d.step()
            mujoco.mj_step(w.model, w.data, SUBSTEPS)
            return
        if self._coach_on:
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
                    if self._try_jump():
                        pass
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
                            if self._try_jump():
                                pass
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
                            if self._try_jump():
                                pass
            self._prev_angle = ang

            # --- metrics updated in _judge_step (oracle side) ---

        self.jump.update(DT)
        if self.jump.state == "idle":
            self._set_grip(False)
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
                # contact evidence: a single-step toe graze on a light floor
                # snake doesn't trip a hopper (it keeps flying and lands
                # upright). A TRIP is contact that is sustained (≥3 steps ≈
                # the rope caught a foot) or that brings the duck down.
                n_contact_steps = sum(1 for t, c in self._contact_hist
                                      if abs(t - v["t_cross"]) <= 0.2 and c)
                window_contact = n_contact_steps >= 3
                upright_after = jd.is_upright(0.45)
                attempted = v["attempted"]
                # airborne: peak feet height within ±0.2 s of the pass (the
                # flight is ~0.6 s long — a tighter window just mis-times)
                air_peak = max((f for t, f in getattr(self, "_feet_hist", [])
                                if abs(t - v["t_cross"]) <= 0.2), default=0.0)
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
                    "attempted": (self.jump.state in ("crouch", "extend", "flight", "land", "jump")
                                  or getattr(self, "_last_jump_t", -9) > self._t - 1.5),
                })
        self._last_ang = ang
