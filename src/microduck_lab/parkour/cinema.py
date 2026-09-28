"""Event-driven cuts over an immutable recorded physical trajectory."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Clip:
    start: float
    stop: float
    speed: float = 1.0
    shot: str = "auto"
    title: str = ""
    hold: float = 0.0

    @property
    def duration(self):
        return self.hold or (self.stop - self.start) / self.speed


class CinematicEventTimeline:
    def __init__(self, report, end):
        self.report = report
        self.events = report["events"]
        self.end = end
        self.finish = next(e["t"] for e in self.events if e["type"] == "FINISH")
        self.door = next(e["t"] for e in self.events if e["type"] == "BOSS_GATE_IMPACT")
        self.stop = min(end, self.door + 1.4)
        self.skills = {}
        for e in self.events:
            if e["type"] == "SKILL_START":
                self.skills.setdefault(e["skill"], [e["t"], e["t"]])
            elif (
                e["type"] == "SKILL_RESULT"
                and e["skill"] in self.skills
                and e["status"] == "SUCCESS"
            ):
                self.skills[e["skill"]][1] = e["t"]
        self.flight = next(f for f in report["gap_flights"] if f["crossed"])
        self.fall = next(
            (e["t"] for e in self.events if e["type"] == "FALL"), float("inf")
        )
        self.recover = next(
            (e["t"] for e in self.events if e["type"] == "RECOVER"), float("inf")
        )
        self.route = next(
            (e["t"] for e in self.events if e["type"] == "ROUTE_PREVIEW"),
            self.skills["JUMP_CENTER"][1],
        )

    def hero(self):
        a, b = self.skills["ROLL_CENTER"]
        f = self.flight
        windows = [
            (a + 0.2, min(b, a + 0.75), 0.5, "roll"),
            (
                max(self.skills["JUMP_CENTER"][0], f["start"] - 0.12),
                f["end"] + 0.16,
                0.4,
                "jump",
            ),
            (self.door - 0.10, self.door + 0.30, 0.4, "finish"),
        ]
        clips = []
        cursor = 0.0002
        for start, stop, speed, shot in windows:
            if start > cursor:
                clips.append(Clip(cursor, start))
            clips.append(Clip(start, stop, speed, shot))
            cursor = stop
        if cursor < self.stop:
            clips.append(Clip(cursor, self.stop))
        clips.append(
            Clip(
                self.stop,
                self.stop,
                title="ESCAPED.",
                shot="finish",
                hold=max(1.5, 25 - sum(c.duration for c in clips)),
            )
        )
        return clips

    def technical(self):
        a, b = self.skills["ROLL_CENTER"]
        ja, jb = self.skills["JUMP_CENTER"]
        da, db = (
            self.skills["TAKE_LEFT_ROUTE"]
            if "TAKE_LEFT_ROUTE" in self.skills
            else self.skills["TAKE_RIGHT_ROUTE"]
        )
        bar = next(
            e["t"]
            for e in self.events
            if e["type"] == "BOSS_HIT_PROP" and e["hazard"] == "bar"
        )
        clips = [
            Clip(
                2.0,
                2.0,
                shot="wide",
                title=(
                    "A TACTICAL MODEL. A PHYSICAL BODY."
                    if self.report["brain"] == "jev"
                    else "PHYSICAL BASELINE / RULE-BASED TACTICS"
                ),
                hold=4.0,
            ),
            Clip(0.0002, self.stop, title="CONTINUOUS RECORDED RUN"),
            Clip(
                a - 0.15,
                b + 0.2,
                0.35,
                "roll",
                "MOVING ROLL / completion from rotation + foot support",
            ),
            Clip(
                da - 0.1,
                db + 0.15,
                1.0,
                "dodge",
                "LANE CHANGE / feedback steering, no root translation",
            ),
            Clip(ja - 0.2, jb + 0.2, 0.35, "jump", "LONG JUMP / a real 15 cm void"),
            Clip(
                self.fall - 0.5 if np.isfinite(self.fall) else self.route,
                self.recover + 0.4
                if np.isfinite(self.recover)
                else min(self.route + 3.5, self.finish),
                1.0,
                "impact" if np.isfinite(self.fall) else "dodge",
                "IMPACT → RECOVER / physical contact remains enabled"
                if np.isfinite(self.fall)
                else "LOCAL PHYSICS PREVIEW / choose a route around the moving arm",
            ),
            Clip(
                bar - 0.3,
                bar + 0.8,
                1.0,
                "boss",
                "BOSS / opens a hinged obstacle with contact force",
            ),
            Clip(
                self.door - 0.5,
                self.door + 0.8,
                0.4,
                "finish",
                "CLOSED DOOR / sphere stopped by a real contact",
            ),
            Clip(
                self.stop,
                self.stop,
                shot="finish",
                title="REPRODUCIBLE INPUT REPLAY",
                hold=5.0,
            ),
        ]
        if sum(c.duration for c in clips) < 60:
            clips[-1] = Clip(
                self.stop,
                self.stop,
                shot="finish",
                title="REPRODUCIBLE INPUT REPLAY",
                hold=5 + 60 - sum(c.duration for c in clips),
            )
        return clips


class ShotDirector:
    def __init__(self, timeline):
        self.timeline = timeline
        self.look = None
        self.velocity = np.zeros(3)
        self.distance = None
        self.azimuth = None
        self.elevation = None

    def choose(self, t):
        tl = self.timeline
        if t < 0.8:
            return "spawn"
        if t >= tl.finish:
            return "finish"
        if tl.fall - 0.5 <= t <= tl.recover + 0.5:
            return "impact"
        for name, shot in [
            ("ROLL_CENTER", "roll"),
            ("JUMP_CENTER", "jump"),
            ("TAKE_LEFT_ROUTE", "dodge"),
            ("TAKE_RIGHT_ROUTE", "dodge"),
        ]:
            if name in tl.skills:
                a, b = tl.skills[name]
                if a - 0.2 <= t <= b + 0.2:
                    return shot
        for e in tl.events:
            if (
                e["type"] == "BOSS_HIT_PROP"
                and e["hazard"] == "bar"
                and e["t"] <= t <= e["t"] + 0.85
            ):
                return "boss"
        return "chase"

    def target(self, t, d, shot="auto"):
        if shot == "auto":
            shot = self.choose(t)
        duck = d.body("duck/trunk_base").xpos.copy()
        boss = d.body("boss").xpos.copy()
        params = {
            "chase": (duck + np.array([0.72, 0, 0.08]), 1.85, 145.0, -23.0),
            "wide": (duck + np.array([0.65, 0, 0.10]), 2.05, 130.0, -25.0),
            "roll": (duck + np.array([0.10, 0, 0.09]), 1.10, 115.0, -15.0),
            "jump": (duck + np.array([0.08, 0, 0.09]), 1.05, 88.0, -17.0),
            "dodge": (duck + np.array([0.26, 0, 0.07]), 1.30, 130.0, -30.0),
            "impact": (duck + np.array([0.03, 0, 0.08]), 1.05, 70.0, -22.0),
            "boss": (boss + np.array([0.25, 0, 0.05]), 1.40, 110.0, -17.0),
            "spawn": (boss + np.array([0.20, 0, -0.03]), 1.50, 120.0, -17.0),
            "finish": (np.array([5.55, 0, 0.24]), 1.60, 105.0, -18.0),
        }
        return params[shot], shot

    def camera(self, t, d, dt, shot="auto", reset=False):
        import mujoco

        (look, distance, azimuth, elevation), kind = self.target(t, d, shot)
        if reset or self.look is None:
            self.look = look.copy()
            self.velocity[:] = 0.0
            self.distance = distance
            self.azimuth = azimuth
            self.elevation = elevation
        else:
            # Critically damped camera spring in output time; exact scene states
            # are never interpolated or altered by the camera.
            omega = 16.0
            self.velocity += (
                omega**2 * (look - self.look) - 2 * omega * self.velocity
            ) * dt
            self.look += self.velocity * dt
            alpha = 1 - np.exp(-10 * dt)
            self.distance += alpha * (distance - self.distance)
            self.azimuth += alpha * (azimuth - self.azimuth)
            self.elevation += alpha * (elevation - self.elevation)
        camera = mujoco.MjvCamera()
        camera.lookat = self.look
        camera.distance = self.distance
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation
        return camera, kind
