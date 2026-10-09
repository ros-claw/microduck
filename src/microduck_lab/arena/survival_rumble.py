"""Last Duck Alive: roaming tactics, public finale and real physical elimination.

No score-based winner, root pose control, identity tie-break or fake elimination.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import gzip
import hashlib
import json
import math
import time
import mujoco
import numpy as np
from .episode import STATE, sha
from .multiplayer import ContactGraph, observe, GameReferee
from .relay_rumble import (
    RelayConfig,
    RelayTactician,
    ContactTriggeredCollapse,
    RelayRace,
    CHARACTERS,
)
from .survival_world import build_survival_world


@dataclass(frozen=True)
class SurvivalConfig(RelayConfig):
    duration: float = 85.0
    tile_size: float = 0.44
    lifetime_s: float = 12.0
    finale_after_captures: int = 5
    finale_deadline_s: float = 36.0
    return_warning_s: float = 8.0
    inner_delay_s: float = 2.5
    tilt_max_rad: float = 0.14
    tilt_ramp_s: float = 12.0
    winner_hold_s: float = 0.75

    def __post_init__(self):
        super().__post_init__()
        if (
            not 3 <= self.finale_after_captures <= 7
            or not 20 <= self.finale_deadline_s <= 45
            or not 5 <= self.return_warning_s <= 10
            or not 1 <= self.inner_delay_s <= 4
            or not 0 <= self.tilt_max_rad <= 0.30
            or not 6 <= self.tilt_ramp_s <= 20
            or not 0.5 <= self.winner_hold_s <= 2
        ):
            raise ValueError("Invalid physical finale configuration")


class RoamingBeacons(RelayRace):
    """Keep capture statistics, but capturing never selects a champion."""

    def update(self, now, observations, eliminated):
        # The archived score-only outcome is deliberately discarded. Captures
        # still advance the public beacon; only LastDuckVictory can end this game.
        result = super().update(now, observations, eliminated)
        if result is not None:
            self.round += 1
            self.progress = dict.fromkeys(self.progress, 0.0)
            self.events.append(
                dict(time=now, state="BEACON_MOVED", tile=self.goal, round=self.round)
            )
        return None


class LastDuckVictory:
    """Only one non-eliminated body plus upright loaded core support may win.

    A 0.75s stability window must contain >=90% actual foot support and no
    unsupported gap longer than 30ms. Any rival still alive prevents victory.
    """

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.candidate = None
        self.since = None
        self.last_loaded = None
        self.loaded_s = 0.0
        self.max_gap = 0.0

    def update(self, now, observations, eliminated):
        alive = [n for n in observations if n not in eliminated]
        if not alive:
            self.reset()
            return dict(status="DRAW", winner=None, alive=[], rule="LastDuckAlive")
        if len(alive) != 1:
            self.reset()
            return None
        n = alive[0]
        o = observations[n]
        if not o["upright"] or o["z"] < -0.12:
            self.reset()
            return None
        loaded = self.config.island in o["supporting_tiles"]
        if self.candidate != n:
            self.reset()
            if not loaded:
                return None
            self.candidate = n
            self.since = now
            self.last_loaded = now
        if loaded:
            self.loaded_s += self.config.dt
            self.last_loaded = now
        gap = now - self.last_loaded
        self.max_gap = max(self.max_gap, gap)
        if gap > 0.03 + 1e-9:
            self.reset()
            return None
        elapsed = now - self.since
        fraction = self.loaded_s / max(elapsed, self.config.dt)
        if loaded and elapsed + 1e-9 >= self.config.winner_hold_s and fraction >= 0.9:
            return dict(
                status="WINNER",
                winner=n,
                alive=alive,
                rule="LastDuckAlive",
                stable_supported_s=elapsed,
                loaded_fraction=min(1.0, fraction),
                max_unsupported_gap_s=self.max_gap,
                eliminated=sorted(eliminated),
            )
        return None


class SurvivalFinale:
    """A public return warning and two announced rings; releases never target names.

    Early tile damage remains actual load-driven. The disclosed finale is a
    game-rule collapse, followed by torque-limited hinge motion, not duck poses.
    """

    def __init__(self, config):
        self.config = config
        self.started = None
        self.outer_done = False
        self.inner_done = False
        self.hold_angles = None

    def advance(self, d, tiles, now, captures, alive, enabled=True):
        if not enabled:
            return []
        events = []
        if self.started is None and (
            captures >= self.config.finale_after_captures
            or now >= self.config.finale_deadline_s
            or (alive <= 2 and now > 10)
        ):
            self.started = now
            events.append(
                dict(
                    time=now,
                    state="FINALE_WARNING",
                    tile=12,
                    outer_release_s=now + self.config.return_warning_s,
                    inner_release_s=now
                    + self.config.return_warning_s
                    + self.config.inner_delay_s,
                )
            )
            for i, t in tiles.items():
                if i == 12 or t.state not in ("LOCKED", "WARNING"):
                    continue
                if t.state == "LOCKED":
                    t.warn(now)
                events.append(
                    dict(
                        time=now,
                        state="FINAL_WARNING",
                        tile=i,
                        release_at_s=now
                        + self.config.return_warning_s
                        + (
                            0
                            if max(abs(i // 5 - 2), abs(i % 5 - 2)) == 2
                            else self.config.inner_delay_s
                        ),
                    )
                )
        if self.started is None:
            return events
        elapsed = now - self.started
        for radius, delay, flag in [
            (2, self.config.return_warning_s, "outer_done"),
            (1, self.config.return_warning_s + self.config.inner_delay_s, "inner_done"),
        ]:
            if elapsed + 1e-9 >= delay and not getattr(self, flag):
                for i, t in tiles.items():
                    if (
                        i != 12
                        and max(abs(i // 5 - 2), abs(i % 5 - 2)) == radius
                        and t.state == "WARNING"
                    ):
                        t.release(d, now)
                        events.append(
                            dict(
                                time=now,
                                state="RELEASED",
                                tile=i,
                                cause="announced_finale_ring",
                            )
                        )
                setattr(self, flag, True)
                events.append(dict(time=now, state="FINALE_RING", radius=radius))
        return events

    def targets(self, now, angles, alive):
        if self.started is None or not self.inner_done:
            return (0.0, 0.0)
        if alive <= 1:
            if self.hold_angles is None:
                self.hold_angles = tuple(float(v) for v in angles)
            return self.hold_angles
        elapsed = (
            now
            - self.started
            - self.config.return_warning_s
            - self.config.inner_delay_s
        )
        ramp = min(1.0, max(0.0, elapsed / self.config.tilt_ramp_s))
        ramp = ramp * ramp * (3 - 2 * ramp)
        amp = self.config.tilt_max_rad * ramp
        return (
            amp * math.sin(0.92 * elapsed),
            amp * 0.85 * math.sin(0.73 * elapsed + 0.8),
        )


class SurvivalTactician(RelayTactician):
    def choose(self, obs):
        if obs.get("finale_active"):
            # A crowd escape onto a collapsing ring is a bad survival action.
            # Clear that stale route, and use real public centre support to settle.
            self.escape_until = obs["sim_time_s"] + 0.04
            self.escape_target = 12
            r = obs["robot"]
            pos = np.array([r["x"], r["y"]])
            if r["current_tile_id"] == 12:
                if not obs["nearby_ducks"] and np.linalg.norm(pos) < 0.145:
                    return "stand", (0, 0, 0), "BALANCE_FINAL_ISLAND"
                # Distinct holding positions avoid all four targeting one point.
                slots = {
                    "rusher": (0.075, 0.075),
                    "survivor": (-0.075, -0.075),
                    "blocker": (-0.075, 0.075),
                    "chaser": (0.075, -0.075),
                }
                delta = np.array(slots[self.role]) - pos
                dist = float(np.linalg.norm(delta))
                if dist < 0.045:
                    return "stand", (0, 0, 0), "HOLD_FINAL_ISLAND"
                angle = math.atan2(delta[1], delta[0]) - r["yaw"]
                angle = math.atan2(math.sin(angle), math.cos(angle))
                vx = min(0.18, 1.6 * dist) if abs(angle) < 0.6 else 0.0
                return (
                    "walk",
                    (vx, 0, float(np.clip(2.4 * angle, -1.2, 1.2))),
                    "REPOSITION_FINAL",
                )
        p, c, intent = super().choose(obs)
        if obs.get("finale_active") and intent == "CAPTURE_BEACON":
            intent = "HOLD_FINAL_ISLAND"
        elif obs.get("finale_active") and intent in (
            "RUSH_BEACON",
            "CHASE_BEACON",
            "SAFE_DETOUR",
        ):
            intent = "RETURN_TO_FINAL"
        return p, c, intent


def _actor_source_hashes():
    return {
        str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("multiplayer.py"),
            Path(__file__).with_name("relay_world.py"),
            Path(__file__).with_name("relay_rumble.py"),
            Path(__file__).with_name("survival_world.py"),
            Path(__file__).with_name("tiles.py"),
            Path(__file__).with_name("referee.py"),
            Path(__file__).with_name("world.py"),
            Path(__file__).with_name("episode.py"),
            Path(__file__).parent.parent / "sim/runtime.py",
            Path(__file__).parent.parent / "sim/composer.py",
        ]
    }


_LOADED_SOURCE_SNAPSHOT = _actor_source_hashes()


def run_survival(out, seed=31001, config=None, capture=False):
    if _actor_source_hashes() != _LOADED_SOURCE_SNAPSHOT:
        raise RuntimeError("Actor sources changed after import; restart the worker")
    config = config or SurvivalConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m, d, ducks, tiles, spawns = build_survival_world(config, seed)
    source_snapshot = _LOADED_SOURCE_SNAPSHOT.copy()
    policy_snapshot = {
        n: sha(p) for n, p in next(iter(ducks.values())).bank.paths.items()
    }
    graph = ContactGraph(m, ducks, tiles)
    weights = {
        n: float(
            sum(
                m.body_mass[i]
                for i in range(m.nbody)
                if m.body(i).name.startswith(n + "/")
            )
            * 9.81
        )
        for n in ducks
    }
    damage = ContactTriggeredCollapse(config, graph, next(iter(weights.values())))
    referee = GameReferee(tuple(ducks))
    claim = RoamingBeacons(config, ducks)
    finale = SurvivalFinale(config)
    victory = LastDuckVictory(config)
    arena_act = np.array([m.actuator("arena_" + a).id for a in ("roll", "pitch")])
    arena_qpos = np.array(
        [m.jnt_qposadr[m.joint("arena_" + a + "_joint").id] for a in ("roll", "pitch")]
    )
    robot_act = np.concatenate([duck.act_ids for duck in ducks.values()])
    controllers = {n: SurvivalTactician(CHARACTERS[n]["role"]) for n in ducks}
    events, decisions = [], []
    support = {n: [] for n in ducks}
    maxsteps = round(config.duration / config.dt)
    state = np.empty(mujoco.mj_stateSize(m, STATE))
    states = controls = None
    if capture:
        mujoco.mj_saveModel(m, str(out / "scene.mjb"), None)
        states = np.lib.format.open_memmap(
            out / "states.npy", mode="w+", dtype="float64", shape=(maxsteps, len(state))
        )
        controls = np.lib.format.open_memmap(
            out / "ctrl.npy", mode="w+", dtype="float64", shape=(maxsteps, m.nu)
        )
    contact_file = gzip.open(out / "contacts.jsonl.gz", "wt") if capture else None
    maxdepth = maxmotor = maxarena = 0.0
    finite, reset, external_zero = True, False, True
    contact_steps = 0
    contact_active = False
    last_contact_event = -1.0
    terminal = terminal_at = None
    start = time.monotonic()
    try:
        for step in range(maxsteps):
            t = step * config.dt
            if step % 80 == 0:
                arena_targets = finale.targets(
                    t, d.qpos[arena_qpos], len(ducks) - len(referee.eliminated)
                )
                d.ctrl[arena_act] = arena_targets
                events.append(
                    dict(
                        time=t,
                        state="PLATFORM_MOTOR",
                        targets=list(arena_targets),
                        angles=d.qpos[arena_qpos].tolist(),
                    )
                )
                events.append(
                    dict(
                        time=t,
                        state="SCOREBOARD",
                        scores=claim.scores.copy(),
                        progress=claim.progress.copy(),
                        tile=12 if finale.started is not None else claim.goal,
                        finale_active=finale.started is not None,
                    )
                )
                for name, duck in ducks.items():
                    obs = observe(
                        name, ducks, tiles, support, config, t, referee.eliminated
                    )
                    for tile in obs["visible_tiles"]:
                        tile["damage"] = damage.damage[tile["id"]]
                        tile["warning_remaining_s"] = None
                        if finale.started is not None and tile["id"] != 12:
                            radius = max(
                                abs(tile["id"] // 5 - 2), abs(tile["id"] % 5 - 2)
                            )
                            deadline = (
                                finale.started
                                + config.return_warning_s
                                + (config.inner_delay_s if radius == 1 else 0)
                            )
                            tile["warning_remaining_s"] = max(0.0, deadline - t)
                    obs.update(
                        objective="LastDuckAlive",
                        finale_active=finale.started is not None,
                        island=12 if finale.started is not None else claim.goal,
                        final_active=True,
                        passive=config.passive,
                        claim_radius_m=config.claim_radius_m,
                        beacon_scores=claim.scores.copy(),
                        beacon_progress=claim.progress.copy(),
                    )
                    if name in referee.eliminated:
                        policy, command, intent = "joint_hold", (0, 0, 0), "ELIMINATED"
                    elif terminal:
                        policy, command, intent = "stand", (0, 0, 0), "MATCH_COMPLETE"
                    else:
                        policy, command, intent = controllers[name].choose(obs)
                    duck.set_command(twist=command)
                    if policy == "joint_hold":
                        d.ctrl[duck.act_ids] = d.qpos[duck.joint_qpos_idx]
                    else:
                        duck.active_policy = policy
                        duck.step()
                    obs["beacon_scores"] = claim.scores.copy()
                    decisions.append(
                        dict(
                            time=t,
                            body_id=name,
                            input=obs,
                            policy=policy,
                            command=command,
                            intent=intent,
                            target=controllers[name].target,
                        )
                    )
            if capture:
                mujoco.mj_getState(m, d, state, STATE)
                states[step], controls[step] = state, d.ctrl
            mujoco.mj_step(m, d)
            pairs, support, encounters = graph.read(m, d, tiles)
            if contact_file:
                contact_file.write(
                    json.dumps(dict(step=step, time=t, contacts=pairs, support=support))
                    + "\n"
                )
            if encounters:
                contact_steps += 1
                if not contact_active and t - last_contact_event >= 0.25:
                    last_contact_event = t
                    events.append(
                        dict(
                            time=t + config.dt,
                            state="BODY_CONTACT",
                            contacts=encounters,
                        )
                    )
            contact_active = bool(encounters)
            events.extend(
                damage.advance(d, tiles, pairs, t + config.dt, terminal is None)
            )
            events.extend(
                finale.advance(
                    d,
                    tiles,
                    t + config.dt,
                    sum(claim.scores.values()),
                    len(ducks) - len(referee.eliminated),
                    terminal is None,
                )
            )
            events.extend(
                e for tile in tiles.values() for e in tile.update(d, t + config.dt)
            )
            observations = {
                n: dict(
                    x=float(o.trunk_pos()[0]),
                    y=float(o.trunk_pos()[1]),
                    z=float(o.trunk_pos()[2]),
                    vz=float(o.trunk_linvel()[2]),
                    supporting_tiles=support[n],
                    contact_step=step,
                    upright=bool(d.xmat[o.trunk_body_id].reshape(3, 3)[2, 2] > 0.65),
                )
                for n, o in ducks.items()
            }
            events.extend(referee.update(t + config.dt, observations))
            if terminal is None and finale.started is None:
                claim.update(t + config.dt, observations, referee.eliminated)
            events.extend(claim.events)
            claim.events.clear()
            outcome = (
                victory.update(t + config.dt, observations, referee.eliminated)
                if terminal is None
                else None
            )
            if terminal is None and outcome:
                terminal, terminal_at = outcome, t + config.dt
                events.append(dict(time=terminal_at, state="TERMINAL", **terminal))
            maxdepth = max(maxdepth, max((max(0, -p[2]) for p in pairs), default=0))
            maxmotor = max(maxmotor, float(np.max(np.abs(d.actuator_force[robot_act]))))
            maxarena = max(maxarena, float(np.max(np.abs(d.actuator_force[arena_act]))))
            finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            reset |= abs(d.time - (t + config.dt)) > 1e-7
            external_zero &= not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            if terminal_at is not None and t + config.dt >= terminal_at + 1.0:
                break
    finally:
        if contact_file:
            contact_file.close()
    steps = step + 1
    if capture:
        states.flush()
        controls.flush()
        np.savez_compressed(
            out / "trajectory.npz", states=states[:steps], ctrl=controls[:steps]
        )
        del states, controls
        (out / "states.npy").unlink()
        (out / "ctrl.npy").unlink()
    with gzip.open(out / "decisions.jsonl.gz", "wt") as f:
        for row in decisions:
            f.write(json.dumps(row) + "\n")
    mujoco.mj_getState(m, d, state, STATE)
    if _actor_source_hashes() != _LOADED_SOURCE_SNAPSHOT:
        raise RuntimeError(
            "Actor sources changed during the run; evidence is not sealed"
        )
    audit = dict(
        schema="microduck.survival-rumble.development.v1",
        seed=seed,
        config=asdict(config),
        capture=capture,
        steps=steps,
        duration=steps * config.dt,
        mujoco_version=mujoco.__version__,
        spawns=spawns,
        outcome=terminal
        or dict(
            status="TIME_LIMIT",
            winner=None,
            alive=[n for n in ducks if n not in referee.eliminated],
            rule="LastDuckAlive",
        ),
        events=events,
        damage=damage.damage,
        body_weight_N=weights,
        body_contact_steps=contact_steps,
        penetration_max_m=maxdepth,
        motor_force_max_Nm=maxmotor,
        arena_motor_force_max_Nm=maxarena,
        final_observations=observations,
        final_alive=[n for n in ducks if n not in referee.eliminated],
        finale_started_s=finale.started,
        finite=finite,
        time_reset=reset,
        external_forces_zero=bool(external_zero),
        solver_warnings=[int(w.number) for w in d.warning],
        final_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
        stationary_fraction=sum(
            r["input"]["robot"]["speed_m_s"] < 0.03
            for r in decisions
            if r["intent"] not in ("ELIMINATED", "MATCH_COMPLETE")
        )
        / max(
            1,
            sum(r["intent"] not in ("ELIMINATED", "MATCH_COMPLETE") for r in decisions),
        ),
        stationary_metric="active nonterminal decisions, including initial settle",
        no_safe_option_fraction=sum(r["intent"] == "NO_SAFE_OPTION" for r in decisions)
        / len(decisions),
        roles={n: c.role for n, c in controllers.items()},
        characters=CHARACTERS,
        travelled_m={n: c.travel for n, c in controllers.items()},
        objective_replans={n: c.replans for n, c in controllers.items()},
        contested_s=claim.contested_steps * config.dt,
        points=claim.scores.copy(),
        policy_sha256=policy_snapshot,
        source_sha256=source_snapshot,
        performance=dict(
            real_time_factor=steps * config.dt / (time.monotonic() - start)
        ),
        hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()},
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
