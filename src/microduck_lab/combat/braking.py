"""Measured exit using the skill's trained zero command, without filtering."""


class SkillBrake:
    def __init__(self):
        self.started = None
        self.stable_since = None
        self.complete = False
        self.previous_head = None
        self.previous_time = None

    def reset(self):
        self.started, self.stable_since, self.complete = None, None, False
        self.previous_head, self.previous_time = None, None

    def ready(self, now, body):
        if self.complete:
            return True
        if self.started is None:
            self.started = now
        head = (body.neck_pitch_delta, body.head_pitch_delta)
        speed = (
            (
                max(abs(a - b) for a, b in zip(head, self.previous_head))
                / max(1e-9, now - self.previous_time)
            )
            if self.previous_head is not None
            else float("inf")
        )
        self.previous_head, self.previous_time = head, now
        # The measured R3 zero-command equilibrium has ~0.33 rad head-pitch
        # offset under gravity. Require a settled feasible pose, not an exact
        # HOME pose that this exported skill never reaches.
        good = (
            body.upright > 0.94
            and body.foot_support
            and body.speed < 0.16
            and abs(body.neck_pitch_delta) < 0.50
            and abs(body.head_pitch_delta) < 0.50
            and speed < 2.0
        )
        self.stable_since = (
            (self.stable_since if self.stable_since is not None else now)
            if good
            else None
        )
        self.complete = (
            now - self.started >= 0.24 - 1e-9
            and self.stable_since is not None
            and now - self.stable_since >= 0.12 - 1e-9
        )
        return self.complete
