"""Encounter-level Jev bridge; conservative route envelopes and explicit fallback.

The course is mapped, while moving object state is measured each request. The
jump uses a separately measured launch window; no forecast invents a jump over
an unsupported policy region. Revalidation remains mandatory at the motor gate.
"""

import json
from pathlib import Path
import numpy as np
from .prediction import observe_hazards, conflicts
from .tactics import RollingHorizon, TacticalClient


class EncounterBrain:
    def __init__(self, m, d, r, track, level, driver, mode="rule"):
        self.m, self.d, self.r, self.track, self.level, self.driver = (
            m,
            d,
            r,
            track,
            level,
            driver,
        )
        self.loop = RollingHorizon(TacticalClient()) if mode == "jev" else None
        self.selected = {}
        self.events = []
        self.requests = []
        self.mode = mode

    def snapshot(self, encounter, remaining, predicted_position=None):
        m, d, r = self.m, self.d, self.r
        states, public = observe_hazards(
            m,
            d,
            self.level.hazards,
            self.driver,
            self.track.lanes,
            float(r.trunk_pos()[0]),
            float(r.trunk_linvel()[0]),
        )
        pos = r.trunk_pos().copy()
        pos[0] += max(0.0, remaining) * max(0.0, r.trunk_linvel()[0])
        if predicted_position is not None:
            pos = np.asarray(predicted_position).copy()
        legal = []
        checks = {}
        if encounter == "bar":
            legal = ["ROLL_CENTER", "BRAKE_AND_WAIT"]
            checks["ROLL_CENTER"] = {
                "method": "measured moving-roll envelope",
                "trigger_distance_m": 0.37,
                "bar_height_m": next(
                    h.z for h in self.level.hazards if h.kind == "push_bar"
                ),
            }
        elif encounter == "crate":
            # Predict the whole 1.3 s lane transition. The falling crate is
            # handled by its measured release ETA and gravity model.
            for action, side in [("TAKE_LEFT_ROUTE", 1), ("TAKE_RIGHT_ROUTE", -1)]:
                measured = json.loads(
                    Path(__file__).with_name("route_envelopes.json").read_text()
                )[action]
                path = [
                    (
                        t,
                        (pos + np.asarray(low)).tolist(),
                        (pos + np.asarray(high)).tolist(),
                    )
                    for t, low, high in measured["path"]
                ]
                relevant = [
                    s for s in states if s.kind in ("crate", "boulder", "sweeper")
                ]
                hits = conflicts(path, relevant, time_offset=remaining)
                checks[action] = {
                    "conflicts": hits,
                    "duration_s": 1.3,
                    "model": "10-seed measured swept envelope + transfer margin, rechecked at handoff",
                }
                if not hits:
                    legal.append(action)
            legal.append("BRAKE_AND_WAIT")
        elif encounter == "gap":
            width = self.level.gaps[0][1] - self.level.gaps[0][0]
            if 0.12 <= width <= 0.20:
                legal.append("JUMP_CENTER")
            legal.append("BRAKE_AND_WAIT")
            checks["JUMP_CENTER"] = {
                "gap_width_m": width,
                "launch_distance_m": [0.06, 0.14],
                "requires_upright": True,
                "max_entry_speed_mps": 0.08,
                "local_trigger_required": True,
            }
        state = dict(
            encounter=encounter,
            simulation_time_s=float(d.time),
            prediction_horizon_s=3.0,
            mission=dict(
                goal="escape", lives=3, style="high", speed="high", risk_budget="medium"
            ),
            robot=dict(
                position=r.trunk_pos().tolist(),
                velocity=r.trunk_linvel().tolist(),
                upright=float(d.xmat[r.trunk_body_id, 8]),
            ),
            predicted_handoff=dict(
                position=pos.tolist(),
                seconds_from_now=remaining,
                expected_upright=0.98,
                committed_maneuver="ROLL_CENTER"
                if remaining > 0 and encounter == "crate"
                else None,
                uncertainty_m=0.03 + 0.025 * remaining**2,
            ),
            hazards=public,
            feasibility=checks,
            candidates=legal,
            guidance="Prefer completing the next encounter without waiting when a listed maneuver is feasible. A brake forfeits progress and lets the boulder close.",
        )
        return state, legal

    def prepare(self, encounter, remaining=0.0, handoff=False, predicted_position=None):
        if encounter in self.selected:
            return self.selected[encounter]
        state, legal = self.snapshot(encounter, remaining, predicted_position)
        default = {
            "bar": "ROLL_CENTER",
            "crate": "TAKE_LEFT_ROUTE",
            "gap": "JUMP_CENTER",
        }[encounter]
        if encounter == "crate" and self.r.trunk_pos()[1] < -0.03:
            default = "TAKE_RIGHT_ROUTE"
        if self.loop is None:
            if handoff:
                self.selected[encounter] = default
            return default if handoff else None
        if (
            legal == ["BRAKE_AND_WAIT"]
            and self.loop.future is None
            and self.loop.ready is None
        ):
            if handoff:
                self.events.append(
                    dict(
                        type="PHYSICAL_WAIT", t=float(self.d.time), encounter=encounter
                    )
                )
                return "BRAKE_AND_WAIT"
            return None
        accepted = self.loop.poll(encounter, legal, completed=handoff)
        if accepted:
            self.selected[encounter] = accepted
            self.events.append(
                dict(
                    type="JEV_DECISION",
                    t=float(self.d.time),
                    encounter=encounter,
                    action=accepted,
                )
            )
            return accepted
        if self.loop.prefetch(encounter, state, legal, remaining):
            self.requests.append(
                dict(
                    t=float(self.d.time),
                    encounter=encounter,
                    state=state,
                    candidates=legal,
                )
            )
        return None

    def close(self):
        if self.loop:
            self.loop.close()
