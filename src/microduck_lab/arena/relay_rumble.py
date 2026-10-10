"""Relay Rumble development rules: moving objectives, fixed character tactics.

Separate ruleset from the immutable Last Duck Standing release. No pose helpers.
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
from .multiplayer import GameConfig, ContactGraph, observe, GameReferee
from .episode import STATE, sha
from .relay_world import build_relay_world


@dataclass(frozen=True)
class RelayConfig(GameConfig):
    receiver_half_extent_m: float = 3.0
    players: int = 4
    grid: int = 5
    tile_size: float = 0.40
    gap: float = 0.008
    duration: float = 65.0
    lifetime_s: float = 8.0
    filter_s: float = 0.12
    final_at_s: float = 0.8
    claim_s: float = 0.75
    claim_radius_m: float = 0.105
    points_to_win: int = 3
    passive: bool = False

    def __post_init__(self):
        if (
            self.receiver_half_extent_m != 3.0
            or self.players != 4
            or self.grid != 5
            or self.layout != "square"
            or not 0.36 <= self.tile_size <= 0.44
            or not 0.005 <= self.gap <= 0.012
            or self.dt != 0.00025
            or self.contact_ref_s != 0.0005
            or self.solver_iterations != 80
            or not self.external_envelopes
            or self.ccd != "libccd"
            or not 20 <= self.duration <= 90
            or not 3 <= self.lifetime_s <= 12
            or not 0.02 <= self.filter_s <= 0.5
            or not 0.5 <= self.claim_s <= 2
            or not 0.07 <= self.claim_radius_m <= 0.12
            or self.points_to_win not in (2, 3)
            or not 0 <= self.final_at_s <= 2
        ):
            raise ValueError("Unsupported relay rules or physical calibration")

    @property
    def stations(self):
        # Equal-distance cardinal outposts, joined by contact-consumed bridges.
        return (12, 2, 14, 22, 10)

    @property
    def island(self):
        return 12


CHARACTERS = {
    "lavender": {"name": "张三", "role": "rusher", "trait": "莽夫抢点"},
    "cream": {"name": "二呆", "role": "survivor", "trait": "谨慎绕行"},
    "sky": {"name": "老六", "role": "blocker", "trait": "预判截路"},
    "graphite": {"name": "卷王", "role": "chaser", "trait": "追分抢位"},
}


class ContactTriggeredCollapse:
    """Low-pass vertical *actual foot loads*, integrated in body-weight seconds.

    No occupancy oracle. Total peak clipped at 3 body weights before filtering.
    Two equally loaded ducks accumulate damage twice as fast. Five permanent outposts exempt; connecting tiles consume real load.
    """

    def __init__(self, config, graph, body_weight):
        self.config, self.graph, self.weight = config, graph, body_weight
        self.filtered = {i: 0.0 for i in config.ids}
        self.damage = {i: 0.0 for i in config.ids}
        self.loaded_once = set()
        self.last_emit = {i: 0 for i in config.ids}

    def advance(self, d, tiles, pairs, now, enabled=True):
        if not enabled:
            return []
        loads = {i: 0.0 for i in tiles}
        owners = {i: set() for i in tiles}
        for p in pairs:
            a, b = p[:2]
            i = self.graph.tiles.get(a, self.graph.tiles.get(b))
            owner = self.graph.feet.get(a, self.graph.feet.get(b))
            if i is None or owner is None or not d.eq_active[tiles[i].eq_id]:
                continue
            up = p[12 + 2] * (1 if b in self.graph.feet else -1)
            if p[3] > 0.05 and up > 0.5:
                loads[i] += p[3] * up
                owners[i].add(owner)
        return self.integrate(d, tiles, loads, owners, now)

    def integrate(self, d, tiles, loads, owners, now):
        events = []
        alpha = -math.expm1(-self.config.dt / self.config.filter_s)
        for i, tile in tiles.items():
            if i in self.config.stations or not d.eq_active[tile.eq_id]:
                continue
            load = min(3 * self.weight, max(0, loads[i]))
            self.filtered[i] += alpha * (load - self.filtered[i])
            if load > 0.05 and i not in self.loaded_once:
                self.loaded_once.add(i)
                events.append(
                    dict(time=now, tile=i, state="LOADED", owners=sorted(owners[i]))
                )
            self.damage[i] += (
                self.filtered[i] / self.weight * self.config.dt / self.config.lifetime_s
            )
            if self.damage[i] >= 0.02 and tile.state == "LOCKED":
                tile.warn(now)
                events.append(
                    dict(time=now, tile=i, state="CRACKING", damage=self.damage[i])
                )
            level = min(10, int(self.damage[i] * 10))
            if level > self.last_emit[i]:
                self.last_emit[i] = level
                events.append(
                    dict(
                        time=now,
                        tile=i,
                        state="DAMAGE",
                        damage=self.damage[i],
                        load_N=self.filtered[i],
                        owners=sorted(owners[i]),
                    )
                )
            if self.damage[i] >= 1 and tile.state == "WARNING":
                tile.release(d, now)
                events.append(
                    dict(time=now, tile=i, state="RELEASED", damage=self.damage[i])
                )
        return events


class RelayRace:
    """A captured beacon immediately moves; no timeout ranking or forced winner.

    Exclusive upright loaded occupancy earns progress, contested occupancy earns
    nothing. Progress belongs to the current beacon and resets after capture.
    Station order is public and geometry-only, never selected under a winner.
    """

    def __init__(self, config, names):
        self.config = config
        self.scores = dict.fromkeys(names, 0)
        self.progress = dict.fromkeys(names, 0.0)
        self.round = 0
        self.events = []
        self.contested_steps = 0

    @property
    def goal(self):
        return self.config.stations[self.round % len(self.config.stations)]

    def update(self, now, observations, eliminated):
        if now <= self.config.final_at_s:
            return None
        c = self.config.centre(self.goal)
        occupants = [
            n
            for n, o in observations.items()
            if n not in eliminated
            and o["z"] >= -0.12
            and math.hypot(o["x"] - c[0], o["y"] - c[1]) <= self.config.claim_radius_m
        ]
        if len(occupants) > 1:
            self.contested_steps += 1
        if len(occupants) != 1:
            return None
        n = occupants[0]
        o = observations[n]
        if not o["upright"] or self.goal not in o["supporting_tiles"]:
            return None
        self.progress[n] += self.config.dt
        if self.progress[n] + 1e-9 < self.config.claim_s:
            return None
        self.scores[n] += 1
        self.events.append(
            dict(
                time=now,
                state="BEACON_CAPTURE",
                body_id=n,
                tile=self.goal,
                scores=self.scores.copy(),
                round=self.round,
            )
        )
        if self.scores[n] >= self.config.points_to_win:
            return dict(
                status="WINNER",
                winner=n,
                alive=[k for k in observations if k not in eliminated],
                rule="RelayRace",
                points=self.scores.copy(),
                required_points=self.config.points_to_win,
            )
        self.round += 1
        self.progress = dict.fromkeys(self.progress, 0.0)
        self.events.append(
            dict(time=now, state="BEACON_MOVED", tile=self.goal, round=self.round)
        )
        return None


class RelayTactician:
    """Weighted public-state routing plus stalled-crowd escape; no pose assistance."""

    def __init__(self, role="rusher"):
        self.role = role
        self.target = None
        self.last_goal = None
        self.last_pos = None
        self.stuck_since = None
        self.escape_until = 0.0
        self.escape_target = None
        self.replans = 0
        self.travel = 0.0
        self.progress_anchor = None
        self.progress_time = 0.0

    def choose(self, obs):
        import heapq

        r, now = obs["robot"], obs["sim_time_s"]
        pos = np.array([r["x"], r["y"]])
        goal = obs["island"]
        n = obs["grid"]
        ts = {t["id"]: t for t in obs["visible_tiles"]}
        if self.last_pos is not None:
            self.travel += float(np.linalg.norm(pos - self.last_pos))
        self.last_pos = pos.copy()
        if obs["passive"] or now < 0.8:
            return "stand", (0, 0, 0), "PASSIVE" if obs["passive"] else "SETTLE"
        safe = {
            i
            for i, t in ts.items()
            if t["state"] in ("LOCKED", "WARNING") and t["damage"] < 0.85
        }
        safe.add(goal)
        anchor = r["current_tile_id"]
        if anchor is None:
            anchor = min(ts, key=lambda i: np.linalg.norm(pos - ts[i]["centre_xy"]))
        rivals = obs["nearby_ducks"]
        crowded = any(np.linalg.norm(pos - [o["x"], o["y"]]) < 0.25 for o in rivals)
        if (
            self.progress_anchor is None
            or np.linalg.norm(pos - self.progress_anchor) > 0.09
            or self.last_goal != goal
        ):
            self.progress_anchor = pos.copy()
            self.progress_time = now
        if crowded and (r["speed_m_s"] < 0.045 or now - self.progress_time > 1.25):
            if self.stuck_since is None:
                self.stuck_since = now
        else:
            self.stuck_since = None
        if (
            self.stuck_since is not None
            and now - self.stuck_since > 1.0
            and now > self.escape_until
        ):
            options = [
                i
                for i in safe
                if abs(i // n - anchor // n) + abs(i % n - anchor % n) == 1
            ]
            if options:
                self.escape_target = max(
                    options,
                    key=lambda i: (
                        min(
                            (
                                np.linalg.norm(
                                    np.array(ts[i]["centre_xy"]) - [o["x"], o["y"]]
                                )
                                for o in rivals
                            ),
                            default=1,
                        )
                        - ts[i]["damage"]
                    ),
                )
                self.escape_until = now + 1.2
                self.stuck_since = None
        dest = self.escape_target if now < self.escape_until else goal
        queue = [(0, anchor, [])]
        best = {anchor: 0.0}
        path = None
        while queue:
            cost, node, route = heapq.heappop(queue)
            if node == dest:
                path = route
                break
            for j in sorted(safe):
                if abs(j // n - node // n) + abs(j % n - node % n) != 1:
                    continue
                c = np.array(ts[j]["centre_xy"])
                crowd = sum(
                    max(0, 0.36 - np.linalg.norm(c - [o["x"], o["y"]])) / 0.36
                    for o in rivals
                )
                weight = 1 + ts[j]["damage"] * (5 if self.role == "survivor" else 2)
                weight += crowd * (
                    2.2
                    if self.role == "survivor"
                    else 0.35
                    if self.role == "rusher"
                    else 0.7
                )
                if cost + weight < best.get(j, math.inf):
                    best[j] = cost + weight
                    heapq.heappush(queue, (cost + weight, j, route + [j]))
        if path is None:
            self.target = None
            return "stand", (0, 0, 0), "NO_SAFE_OPTION"
        self.target = path[0] if path else dest
        # Skip the current tile once within a conservative centre corridor.
        if len(path) > 1 and np.linalg.norm(pos - ts[self.target]["centre_xy"]) < 0.10:
            self.target = path[1]
        if self.last_goal != goal:
            self.replans += 1
            self.last_goal = goal
        target = np.array(ts[self.target]["centre_xy"])
        intent = "RUSH_BEACON"
        if now < self.escape_until:
            intent = "ESCAPE_CROWD"
        elif self.role == "survivor":
            intent = "SAFE_DETOUR"
        elif self.role == "chaser":
            intent = "CHASE_BEACON"
        elif self.role == "blocker" and rivals:
            # Intercept a rival's public projected position near the objective.
            c = np.array(ts[goal]["centre_xy"])
            rival = min(rivals, key=lambda o: np.linalg.norm(c - [o["x"], o["y"]]))
            predicted = np.array(
                [rival["x"] + 0.45 * rival["vx"], rival["y"] + 0.45 * rival["vy"]]
            )
            if (
                self.target == goal
                and np.linalg.norm(predicted - c) < 0.34
                and np.linalg.norm(pos - c) > 0.10
            ):
                v = predicted - c
                target = c + v / max(np.linalg.norm(v), 0.001) * 0.085
                intent = "INTERCEPT_APPROACH"
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if (
            self.target == goal
            and dist < 0.055
            and anchor == goal
            and now >= self.escape_until
        ):
            return "stand", (0, 0, 0), "CAPTURE_BEACON"
        angle = math.atan2(delta[1], delta[0]) - r["yaw"]
        angle = math.atan2(math.sin(angle), math.cos(angle))
        vx = float(np.clip(2.4 * dist, 0.08, 0.48)) if abs(angle) < 0.60 else 0.0
        return "walk", (vx, 0, float(np.clip(3 * angle, -1.8, 1.8))), intent


def _actor_source_hashes():
    return {
        str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("multiplayer.py"),
            Path(__file__).with_name("relay_world.py"),
            Path(__file__).with_name("tiles.py"),
            Path(__file__).with_name("referee.py"),
            Path(__file__).with_name("world.py"),
            Path(__file__).with_name("episode.py"),
            Path(__file__).parent.parent / "sim/runtime.py",
            Path(__file__).parent.parent / "sim/composer.py",
        ]
    }


_LOADED_SOURCE_SNAPSHOT = _actor_source_hashes()


def run_relay(out, seed=31001, config=None, capture=False):
    if _actor_source_hashes() != _LOADED_SOURCE_SNAPSHOT:
        raise RuntimeError("Actor sources changed after import; restart the worker")
    config = config or RelayConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    m, d, ducks, tiles, spawns = build_relay_world(config, seed)
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
    claim = RelayRace(config, ducks)
    controllers = {n: RelayTactician(CHARACTERS[n]["role"]) for n in ducks}
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
    maxdepth = maxmotor = 0.0
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
                events.append(
                    dict(
                        time=t,
                        state="SCOREBOARD",
                        scores=claim.scores.copy(),
                        progress=claim.progress.copy(),
                        tile=claim.goal,
                    )
                )
                for name, duck in ducks.items():
                    obs = observe(
                        name, ducks, tiles, support, config, t, referee.eliminated
                    )
                    for tile in obs["visible_tiles"]:
                        tile["damage"] = damage.damage[tile["id"]]
                        # No fixed countdown exists in this mode.
                        tile["warning_remaining_s"] = None
                    obs.update(
                        objective="RelayRace",
                        island=claim.goal,
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
            outcome = (
                claim.update(t + config.dt, observations, referee.eliminated)
                if terminal is None
                else None
            )
            events.extend(claim.events)
            claim.events.clear()
            survival = referee.outcome()
            if not outcome and survival["status"] == "DRAW":
                outcome = dict(**survival, rule="LastSupportedSurvivor")
            if terminal is None and outcome:
                terminal, terminal_at = outcome, t + config.dt
                events.append(dict(time=terminal_at, state="TERMINAL", **terminal))
            maxdepth = max(maxdepth, max((max(0, -p[2]) for p in pairs), default=0))
            maxmotor = max(maxmotor, float(np.max(np.abs(d.actuator_force))))
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
        schema="microduck.relay-rumble.development.v1",
        seed=seed,
        config=asdict(config),
        capture=capture,
        steps=steps,
        duration=steps * config.dt,
        mujoco_version=mujoco.__version__,
        spawns=spawns,
        outcome=terminal
        or dict(
            **referee.outcome(timeout=True),
            rule="RelayRace",
            points=claim.scores.copy(),
        ),
        events=events,
        damage=damage.damage,
        body_weight_N=weights,
        body_contact_steps=contact_steps,
        penetration_max_m=maxdepth,
        motor_force_max_Nm=maxmotor,
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
