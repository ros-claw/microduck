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
        envelopes = json.loads(
            Path(__file__).with_name("route_envelopes.json").read_text()
        )
        profile = envelopes.get("_profile", {})
        if mode == "jev":
            matched = abs(profile.get("lane_width_m", -1.0) - track.lane_width) < 1e-5
            foot = m.geom("duck/left_foot_collision").id
            matched &= np.allclose(
                m.geom_solref[foot], [profile.get("duck_self_reference_s", -1.0), 1.0]
            )
            ground_pairs = [
                i
                for i in range(m.npair)
                if foot in (m.pair_geom1[i], m.pair_geom2[i])
                and any(
                    m.geom(g).name.startswith("floor/")
                    for g in (m.pair_geom1[i], m.pair_geom2[i])
                )
            ]
            matched &= bool(ground_pairs) and all(
                np.allclose(
                    m.pair_solref[i], [profile.get("duck_floor_reference_s", -1.0), 1.0]
                )
                and abs(m.pair_margin[i] - profile.get("ground_margin_m", -1.0)) < 1e-8
                for i in ground_pairs
            )
            if not matched:
                raise ValueError(
                    "Route envelopes do not match this physical profile. Requalify before live Jev; the current envelopes require --hard-contacts."
                )
        self.route_envelopes = envelopes
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
        if encounter in ("bar", "exit_bar"):
            legal = ["ROLL_CENTER", "BRAKE_AND_WAIT"]
            checks["ROLL_CENTER"] = {
                "method": "measured moving-roll envelope",
                "trigger_distance_m": 0.37,
                "bar_height_m": next(
                    h.z for h in self.level.hazards if h.name == encounter
                ),
            }
        elif encounter == "crate":
            # Predict the whole 1.3 s lane transition. The falling crate is
            # handled by its measured release ETA and gravity model.
            for action, side in [("TAKE_LEFT_ROUTE", 1), ("TAKE_RIGHT_ROUTE", -1)]:
                entries = self.route_envelopes[action]["entries"]
                expected_vx = 0.05 if remaining > 0 else float(r.trunk_linvel()[0])
                eligible = [
                    e
                    for e in entries
                    if abs(pos[1] - e["entry_y_m"]) <= e["entry_y_tolerance_m"]
                    and e["entry_vx_range_mps"][0]
                    <= expected_vx
                    <= e["entry_vx_range_mps"][1]
                ]
                if not eligible:
                    checks[action] = {
                        "rejected": "entry state outside measured route envelope"
                    }
                    continue
                measured = min(
                    eligible,
                    key=lambda e: (
                        abs(pos[1] - e["entry_y_m"]),
                        abs(expected_vx - e.get("entry_vx_mps", expected_vx)),
                    ),
                )
                path = [
                    (
                        t,
                        (pos + np.asarray(low)).tolist(),
                        (pos + np.asarray(high)).tolist(),
                    )
                    for t, low, high in measured["path"]
                ]
                support_path = [
                    (
                        t,
                        (pos + np.asarray(low)).tolist(),
                        (pos + np.asarray(high)).tolist(),
                    )
                    for t, low, high in measured.get("support_path", measured["path"])
                ]
                # The encounter continues after lane-change completion. Check
                # holding that lane until the robot has passed the falling box,
                # rather than declaring a lane safe while still upstream of it.
                crate = next(h for h in self.level.hazards if h.kind == "crate")
                end_t, end_low, end_high = path[-1]
                low, high = np.asarray(end_low), np.asarray(end_high)
                center_x = float((low[0] + high[0]) / 2)
                follow = min(2.5, max(0.0, (crate.x + 0.30 - center_x) / 0.55))
                support_low, support_high = map(np.asarray, support_path[-1][1:])
                for dt in np.arange(0.02, follow + 0.02001, 0.02):
                    shift = np.array([0.55 * dt, 0.0, 0.0])
                    padding = np.array([0.025 * dt, 0.003 * dt, 0.0])
                    path.append(
                        (
                            float(end_t + dt),
                            (low + shift - padding).tolist(),
                            (high + shift + padding).tolist(),
                        )
                    )
                    support_path.append(
                        (
                            float(end_t + dt),
                            (support_low + shift - padding).tolist(),
                            (support_high + shift + padding).tolist(),
                        )
                    )
                relevant = [
                    s for s in states if s.kind in ("crate", "boulder", "sweeper")
                ]
                hits = conflicts(path, relevant, time_offset=remaining)
                within_track = all(
                    low[1] >= -self.track.width / 2 and high[1] <= self.track.width / 2
                    for _, low, high in support_path
                )
                checks[action] = {
                    "conflicts": hits,
                    "duration_s": measured["duration_s"],
                    "encounter_forecast_s": path[-1][0],
                    "within_track": within_track,
                    "support_model": "measured foot envelope"
                    if "support_path" in measured
                    else "conservative full-body fallback",
                    "model": "hard-contact, entry-state-conditioned measured swept envelope; rechecked at handoff",
                }
                if not hits and within_track:
                    legal.append(action)
            legal.append("BRAKE_AND_WAIT")
        elif encounter == "bowling":
            ball = d.body("playball").xpos
            ready = bool(
                d.xmat[r.trunk_body_id, 8] > 0.9
                and ball[0] > r.trunk_pos()[0] + 0.15
                and abs(ball[1]) < 0.12
                and not self.driver.bowling.boss_ball_interference
            )
            if ready:
                legal.append("PUSH_BALL")
            legal.append("BRAKE_AND_WAIT")
            checks["PUSH_BALL"] = dict(
                ready=ready,
                ball_position=ball.tolist(),
                method="upright approach to a reachable lightweight ball in the physical guided alley; strike is verified only after contact",
            )
        elif encounter == "gap":
            width = self.level.gaps[0][1] - self.level.gaps[0][0]
            if 0.12 <= width <= 0.18:
                legal.append("JUMP_CENTER")
            legal.append("BRAKE_AND_WAIT")
            checks["JUMP_CENTER"] = {
                "gap_width_m": width,
                "launch_distance_m": [0.04, 0.08],
                "requires_upright": True,
                "max_entry_speed_mps": 0.08,
                "local_trigger_required": True,
            }
        state = dict(
            encounter=encounter,
            simulation_time_s=float(d.time),
            prediction_horizon_s=3.0,
            mission=dict(
                goal="strike_to_unlock_then_escape"
                if hasattr(self.driver, "bowling")
                else "escape",
                lives=3,
                style="high",
                speed="high",
                risk_budget="medium",
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
            "bowling": "PUSH_BALL",
            "exit_bar": "ROLL_CENTER",
        }[encounter]
        if encounter == "crate" and self.r.trunk_pos()[1] < -0.03:
            default = "TAKE_RIGHT_ROUTE"
        if self.loop is None:
            if default not in legal:
                default = next(
                    (a for a in legal if a != "BRAKE_AND_WAIT"), "BRAKE_AND_WAIT"
                )
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
