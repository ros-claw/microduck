"""Jev encounter decisions with single-flight prefetch and execution-time checks.

The physical executor supplies only certified candidates and a predicted end
snapshot. A tactical timeout does not interrupt a committed motor skill.
"""

import hashlib, json, math, time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
from ..jev.client import JevClient, ENDPOINT

DESCRIPTIONS = {
    "RUN": "Continue forward through the verified clear corridor.",
    "ROLL_CENTER": "Approach the next low crossbar and perform a full forward roll; local feedback triggers it.",
    "TAKE_LEFT_ROUTE": "Take the physically reachable route to higher world y, then align forward.",
    "TAKE_RIGHT_ROUTE": "Take the physically reachable route to lower world y, then align forward.",
    "JUMP_CENTER": "Execute a certified long jump across the next real gap; local feedback triggers takeoff.",
    "PUSH_BALL": "Approach the lightweight bowling ball, push through real contact, then brake and observe the pins. A verified strike unlocks the exit.",
    "BRAKE_AND_WAIT": "Stop in a verified safe waiting region until a passage window opens.",
}


def make_request(state, candidates, model="jev-latest", profile="choice"):
    if not candidates or any(c not in DESCRIPTIONS for c in candidates):
        raise ValueError("Invalid maneuvers")
    questions = {
        "next_action": dict(
            type="choice",
            instructions="Choose the best maneuver for the next encounter, over a 2–4 second tactical horizon. "
            "The 50 Hz physical executor controls exact timing. Use the mission risk budget and observed "
            "motion forecasts. Prefer forward progress when feasible; never assume unobserved future events. "
            "Only listed candidates have passed the current physical feasibility filter.",
            criteria={c: DESCRIPTIONS[c] for c in candidates},
        )
    }
    if profile in ("choice_noul", "four"):
        questions["uncertain"] = dict(
            type="noul",
            instructions="Is the observed encounter too ambiguous to choose a safe maneuver?",
        )
    if profile == "four":
        questions["risk"] = dict(
            type="score",
            instructions="Rate collision risk over the encounter.",
            criteria=["safe", "mild", "high", "critical"],
        )
        questions["act_now"] = dict(
            type="noul",
            instructions="Should the executor change maneuver at the next handoff?",
        )
    if profile not in ("choice", "choice_noul", "four"):
        raise ValueError(profile)
    return dict(model=model, state=state, questions=questions)


class TacticalClient(JevClient):
    def __init__(self, *args, profile="choice_noul", **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = profile

    def decide(self, state, candidates):
        body = make_request(state, candidates, self.model, self.profile)
        request = Request(
            ENDPOINT,
            data=json.dumps(body, allow_nan=False).encode(),
            headers={
                "Authorization": "Bearer " + self.key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.monotonic()
        with urlopen(request, timeout=self.timeout) as response:
            raw = json.load(response)
        choice = raw["answers"]["next_action"]
        prob = choice["probabilities"]
        values = list(map(float, prob.values()))
        if (
            choice.get("type") != "choice"
            or choice["choice"] not in candidates
            or set(prob) != set(candidates)
        ):
            raise ValueError("Invalid action")
        if (
            not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            or abs(sum(values) - 1) > 0.005 * len(values) + 1e-8
        ):
            raise ValueError("Invalid distribution")
        confidence = float(choice["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        uncertain = (
            float(raw["answers"]["uncertain"]["noul"])
            if self.profile != "choice"
            else None
        )
        if uncertain is not None and (
            not math.isfinite(uncertain) or not 0 <= uncertain <= 1
        ):
            raise ValueError("Invalid uncertainty")
        return dict(
            action=choice["choice"],
            probabilities=prob,
            confidence=confidence,
            uncertain=uncertain,
            latency_ms=1000 * (time.monotonic() - start),
            raw=raw,
        )


class RollingHorizon:
    def __init__(self, client, deadline=1.5, handoff_grace=0.35):
        self.client = client
        self.deadline = deadline
        self.handoff_grace = handoff_grace
        self.deferred = False
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.ready = None
        self.records = []

    def prefetch(self, encounter_id, predicted_end_state, candidates, remaining):
        if remaining > 1.0 or self.future is not None or self.ready is not None:
            return False
        self.encounter_id = encounter_id
        self.sent = time.monotonic()
        self.snapshot = json.loads(json.dumps(predicted_end_state, allow_nan=False))
        self.candidates = list(candidates)
        self.future = self.pool.submit(
            self.client.decide, self.snapshot, self.candidates
        )
        return True

    def poll(self, encounter_id, legal, completed=False):
        now = time.monotonic()
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                status = "ready"
                if now - self.sent > self.deadline:
                    status = "deadline"
                elif self.encounter_id != encounter_id:
                    status = "encounter_changed"
                elif result.get("uncertain") is not None and result["uncertain"] > 0.7:
                    status = "uncertain"
                if status == "ready":
                    self.ready = (result, self.encounter_id, now)
                    self.deferred = False
                self.records.append(
                    dict(
                        status=status,
                        encounter=self.encounter_id,
                        result=result,
                        state_hash=hashlib.sha256(
                            json.dumps(self.snapshot, sort_keys=True).encode()
                        ).hexdigest(),
                    )
                )
            except Exception as exc:
                self.records.append(dict(status="api_error", error=type(exc).__name__))
            self.future = None
        if self.ready is None or not completed:
            return None
        result, requested_id, received = self.ready
        # A gait transition can briefly leave a measured entry envelope. Keep
        # the existing answer for a bounded window, rechecking every handoff;
        # never execute it while illegal or extend the freshness deadline.
        if (
            requested_id == encounter_id
            and result["action"] not in legal
            and now - received < self.handoff_grace
        ):
            if not self.deferred:
                self.records.append(
                    dict(
                        status="deferred_at_handoff",
                        encounter=encounter_id,
                        action=result["action"],
                    )
                )
                self.deferred = True
            return None
        self.ready = None
        valid = (
            requested_id == encounter_id
            and result["action"] in legal
            and now - received < 1.5
        )
        self.records.append(
            dict(
                status="accepted" if valid else "invalid_at_handoff",
                encounter=encounter_id,
                action=result["action"],
            )
        )
        return result["action"] if valid else None

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
