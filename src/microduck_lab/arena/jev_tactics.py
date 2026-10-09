"""Live batched Jev tactics; deterministic routing and a local recovery reflex.

Jev selects goals, never joint positions. The raw typed answers are retained.
No privileged scheduler state, future trajectories, or credentials enter logs.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import json
import math
import time
from urllib.request import Request, urlopen

import numpy as np

from ..jev.client import ENDPOINT, JevClient
from .relay_rumble import CHARACTERS


def public_route(obs, goal):
    r, ts = obs["robot"], {t["id"]: t for t in obs["visible_tiles"]}
    safe = {i for i, t in ts.items() if t["state"] in ("LOCKED", "WARNING")
            and (t["warning_remaining_s"] is None or t["warning_remaining_s"] > 1.0)}
    anchor = r["current_tile_id"]
    if anchor is None:
        anchor = min(ts, key=lambda i: math.hypot(r["x"]-ts[i]["centre_xy"][0], r["y"]-ts[i]["centre_xy"][1]))
    if goal not in safe or anchor not in safe:
        # A warned departure tile may still be used as the route's start.
        if goal not in safe or ts[anchor]["state"] not in ("LOCKED", "WARNING"):
            return None
    queue = [(0., anchor, [])]
    seen = {anchor: 0.}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == goal:
            return path
        for j in sorted(safe):
            if abs(j//5-node//5)+abs(j%5-node%5) != 1:
                continue
            crowd = sum(max(0., .32-math.hypot(o["x"]-ts[j]["centre_xy"][0], o["y"]-ts[j]["centre_xy"][1])) for o in obs["nearby_ducks"])
            new = cost + 1 + ts[j]["damage"] + crowd
            if new < seen.get(j, math.inf):
                seen[j] = new
                heapq.heappush(queue, (new, j, path+[j]))
    return None


def candidates(obs):
    r, ts = obs["robot"], {t["id"]: t for t in obs["visible_tiles"]}
    reachable = [i for i in ts if public_route(obs, i) is not None]
    result = {}
    if r["current_tile_id"] in reachable and ts[r["current_tile_id"]]["warning_remaining_s"] is None:
        result["HOLD"] = dict(tile=r["current_tile_id"], description="Hold this safe tile briefly; yield initiative, keep balance.")
    if obs["island"] in reachable:
        result["CAPTURE"] = dict(tile=obs["island"], description="Approach the moving public beacon and contest it; captures are statistics, NOT victory.")
    if 12 in reachable:
        result["RETREAT_CORE"] = dict(tile=12, description="Return to the permanent centre via existing tiles; rivals may already occupy it.")
    if reachable:
        safe = min(reachable, key=lambda i: sum(max(0., .55-math.hypot(o["x"]-ts[i]["centre_xy"][0], o["y"]-ts[i]["centre_xy"][1])) for o in obs["nearby_ducks"])
                   + .1*math.hypot(r["x"]-ts[i]["centre_xy"][0], r["y"]-ts[i]["centre_xy"][1]) + ts[i]["damage"])
        result["EVADE"] = dict(tile=safe, description="Move toward this reachable less crowded tile, making room for balance/recovery.")
        for rival in sorted(obs["nearby_ducks"], key=lambda o: math.hypot(r["x"]-o["x"], r["y"]-o["y"]))[:1]:
            target = min(reachable, key=lambda i: math.hypot(ts[i]["centre_xy"][0]-rival["x"]-.35*rival["vx"], ts[i]["centre_xy"][1]-rival["y"]-.35*rival["vy"]))
            result["INTERCEPT"] = dict(tile=target, rival=rival["body_id"], description="Approach the named rival's projected route on a reachable tile; contact is physical, success is not guaranteed.")
    if not result:
        result["BRAKE"] = dict(tile=None, description="No reachable destination; stop translation and balance.")
    return result


def make_request(observations, options, model="jev-latest"):
    # The board is shared, rather than repeated four times. Round sensor values
    # only in the network payload; the motor loop retains full precision.
    def rounded(value):
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [rounded(v) for v in value]
        return value

    first = next(iter(observations.values()))
    compact = {n: {k: v for k, v in o.items() if k not in ("visible_tiles", "schema", "grid", "objective", "beacon_scores", "beacon_progress")}
               for n, o in observations.items()}
    return dict(model=model, state=dict(
        game="Four ducks, one physical survivor. Falling below the arena eliminates; no score tiebreak. Tiles break one at a time after a public countdown. No future schedule is visible.",
        horizon_s=2.0, observations=rounded(compact), candidates=options,
        shared_board=rounded(first["visible_tiles"]), beacon_scores=first.get("beacon_scores", {}),
        personalities={n: CHARACTERS[n] for n in observations},
        local_motor_reflex="A tilted duck stops pursuit and runs its stand/get-up policy until stably upright. This does not teleport it or prevent real contacts."),
        questions={n: dict(type="choice", instructions=f"Choose the next tactical action for duck '{n}' ({CHARACTERS[n]['name']}, personality {CHARACTERS[n]['trait']}) using its observation and candidates in state. Seek survival with active territorial competition: approach/contest/intercept when safe, evade warned tiles early; avoid pointless permanent camping. Predict only from current visible motion. HOLD is a brief tactical pause. Each other question controls a different duck.",
                          criteria={k: f"{v['description']} Destination tile {v['tile']}." for k, v in opts.items()})
                   for n, opts in options.items()})


def validate(raw, options):
    if not isinstance(raw.get("model"), str) or set(raw["answers"]) != set(options):
        raise ValueError("Unexpected model/answer set")
    plans = {}
    for name, opts in options.items():
        answer = raw["answers"][name]
        probs = answer["probabilities"]
        confidence = float(answer["confidence"])
        values = [float(v) for v in probs.values()]
        if (answer.get("type") != "choice" or answer["choice"] not in opts
                or set(probs) != set(opts) or not math.isfinite(confidence) or not 0 <= confidence <= 1
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                or abs(sum(values)-1) > .005*len(values)+1e-8):
            raise ValueError("Invalid Jev tactical distribution")
        plans[name] = dict(action=answer["choice"], confidence=confidence,
                           **opts[answer["choice"]])
    return plans


class LiveTactics:
    """One shared asynchronous request, four typed choices; not four LLM agents."""

    def __init__(self, path, interval=2.0, client=None):
        self.client = client or JevClient(timeout=4.)
        self.interval = interval
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.last_submit = -math.inf
        self.rows = []
        self.plans = {}
        self.path = path

    def _call(self, body):
        start = time.monotonic()
        request = Request(ENDPOINT, data=json.dumps(body, allow_nan=False).encode(),
                          headers={"Authorization": "Bearer "+self.client.key, "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.client.timeout) as response:
            raw = json.load(response)
        return raw, (time.monotonic()-start)*1000

    def update(self, now, observations):
        if self.pending is not None and self.pending.done():
            row = self.rows[-1]
            row.update(delivered_s=now, physical_age_s=now-row["submitted_s"])
            try:
                raw, latency = self.pending.result()
                row.update(raw=raw, latency_ms=latency)
                plans = validate(raw, row["request"]["state"]["candidates"])
                row["validated_plans"] = plans
                row["accepted"] = {}
                for n, plan in plans.items():
                    accepted = (n in observations and now-row["submitted_s"] <= 3.0
                                and (plan["tile"] is None or public_route(observations[n], plan["tile"]) is not None))
                    row["accepted"][n] = accepted
                    if accepted:
                        self.plans[n] = dict(**plan, request_id=row["id"], delivered_s=now, expires_s=now+3.0)
            except Exception as exc:
                row["error"] = type(exc).__name__  # Never log HTTP authorization/request headers.
            self.pending = None
            self.flush()
        if observations and self.pending is None and now-self.last_submit >= self.interval:
            opts = {n: candidates(o) for n, o in observations.items()}
            body = make_request(observations, opts, self.client.model)
            self.rows.append(dict(id=len(self.rows), submitted_s=now, request=body,
                                  request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()))
            self.last_submit = now
            self.pending = self.pool.submit(self._call, body)
        return self.plans

    def flush(self):
        self.path.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2)+"\n")

    def close(self):
        self.pool.shutdown(wait=True)
        # A response arriving after simulation end must not be labelled applied.
        if self.pending is not None:
            row = self.rows[-1]
            try:
                raw, latency = self.pending.result()
                row.update(raw=raw, latency_ms=latency, status="arrived_after_match_not_applied")
            except Exception as exc:
                row["error"] = type(exc).__name__
        self.flush()


class RecordedTactics:
    """Replay the recorded network delivery tape, NEVER a new online Jev run.

    Regenerate request states and acceptance checks before applying raw answers.
    A different simulation state or delivery sequence fails closed.
    """
    def __init__(self, tape, path):
        self.rows = json.loads(tape.read_text())
        self.path = path
        self.plans = {}
        self.submitted = set()
        self.delivered = set()

    def update(self, now, observations):
        for row in self.rows:
            if row.get("delivered_s") == now:
                if row["id"] not in self.submitted:
                    raise ValueError("Delivery precedes regenerated request")
                accepted_map = {}
                if "raw" in row and "error" not in row:
                    opts = row["request"]["state"]["candidates"]
                    plans = validate(row["raw"], opts)
                    if plans != row["validated_plans"]:
                        raise ValueError("Validated Jev plans differ")
                    for n, plan in plans.items():
                        accepted = (n in observations and now-row["submitted_s"] <= 3.0
                                        and (plan["tile"] is None or public_route(observations[n], plan["tile"]) is not None))
                        accepted_map[n] = accepted
                        if accepted:
                            self.plans[n] = dict(**plan, request_id=row["id"], delivered_s=now, expires_s=now+3.0)
                    if accepted_map != row["accepted"]:
                        raise ValueError("Regenerated acceptance differs")
                self.delivered.add(row["id"])
            if row["submitted_s"] == now:
                opts = {n: candidates(o) for n, o in observations.items()}
                body = make_request(observations, opts, row["request"]["model"])
                if body != row["request"]:
                    raise ValueError(f"Regenerated public request differs: {row['id']}")
                self.submitted.add(row["id"])
        return self.plans

    def close(self):
        if self.submitted != {r["id"] for r in self.rows} or self.delivered != {r["id"] for r in self.rows if "delivered_s" in r}:
            raise ValueError("Recorded response timeline was not consumed")
        self.path.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2)+"\n")


class JevMotor:
    def __init__(self, role):
        self.role = role
        self.target = None
        self.travel = 0.
        self.last_pos = None
        self.replans = 0
        self.recovering = False
        self.stable_since = None
        self.recovery_since = None
        self.events = []

    def choose(self, obs, plan):
        r, now = obs["robot"], obs["sim_time_s"]
        pos = np.array([r["x"], r["y"]])
        if self.last_pos is not None:
            self.travel += float(np.linalg.norm(pos-self.last_pos))
        self.last_pos = pos.copy()
        if r["up_cos"] < .72 and not self.recovering:
            self.recovering = True
            self.recovery_since = now
            self.events.append(dict(time=now, state="RECOVERY_STARTED", body_id=r["body_id"]))
        if self.recovering:
            good = r["up_cos"] > .94 and r["current_tile_id"] is not None and r["speed_m_s"] < .16
            if not good:
                self.stable_since = None
            elif self.stable_since is None:
                self.stable_since = now
            if self.stable_since is not None and now-self.stable_since >= .18-1e-9:
                self.recovering = False
                self.stable_since = None
                self.events.append(dict(time=now, state="RECOVERY_COMPLETED", body_id=r["body_id"], duration_s=now-self.recovery_since))
            else:
                return "stand", (0, 0, 0), "LOCAL_RECOVERY"
        if now < .8:
            return "stand", (0, 0, 0), "SETTLE"
        if plan is None or now > plan["expires_s"]:
            return "stand", (0, 0, 0), "AWAIT_JEV"
        goal = plan["tile"]
        path = public_route(obs, goal) if goal is not None else None
        if path is None:
            return "stand", (0, 0, 0), "JEV_PLAN_INVALID_BRAKE"
        if self.target != goal:
            self.replans += 1
        self.target = goal
        ts = {t["id"]: t for t in obs["visible_tiles"]}
        waypoint = path[0] if path else goal
        if len(path) > 1 and np.linalg.norm(pos-ts[waypoint]["centre_xy"]) < .09:
            waypoint = path[1]
        target = np.array(ts[waypoint]["centre_xy"])
        if plan["action"] == "INTERCEPT" and waypoint == goal:
            rival = next((o for o in obs["nearby_ducks"] if o["body_id"] == plan.get("rival")), None)
            if rival:
                # Bounded within the tile footprint, never follow a rival into a hole.
                offset = np.array([rival["x"]+.25*rival["vx"], rival["y"]+.25*rival["vy"]])-target
                target += np.clip(offset, -.075, .075)
        delta = target-pos
        dist = float(np.linalg.norm(delta))
        if dist < .055 and r["current_tile_id"] == goal:
            return "stand", (0, 0, 0), "JEV_"+plan["action"]
        angle = math.atan2(delta[1], delta[0])-r["yaw"]
        angle = math.atan2(math.sin(angle), math.cos(angle))
        vx = min(.38, 2.0*dist) * max(0., math.cos(angle)) if abs(angle) < .70 else 0.
        return "walk", (vx, 0, float(np.clip(2.5*angle, -1.3, 1.3))), "JEV_"+plan["action"]
