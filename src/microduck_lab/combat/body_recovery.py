"""Opt-in contact recovery; a bounded 61D motor selected from public state."""

from .recovery import RecoveryGuard


class BodyRecoveryGuard:
    def __init__(self, name, learned=False):
        self.name, self.learned = name, learned
        self.active = False
        self.started = self.stable_since = self.last_touch = None
        self.settle_until = 0.0
        self.learned_seconds = 0.0
        self.last_time = None
        self.last_policy = None
        self.fallback = None
        self.fallback_cursor = 0
        self.events = []

    def event(self, now, state, **kw):
        self.events.append(dict(time_s=now, state=state, actor=self.name, **kw))

    def choose(self, now, me):
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        if self.active and self.last_policy == "recovery_candidate":
            self.learned_seconds += dt
        self.last_time = now
        if me.touching_opponent:
            self.last_touch = now
        if not self.active and me.upright < 0.80:
            self.active, self.started, self.stable_since = True, now, None
            self.learned_seconds, self.last_policy = 0.0, None
            self.fallback, self.fallback_cursor = None, 0
            self.event(now, "RECOVERY_STARTED", upright=me.upright)
        if not self.active:
            return "stand" if now < self.settle_until else None
        good = (
            me.upright > 0.94
            and me.z > 0.105
            and me.foot_support
            and not me.body_ground
            and me.speed < 0.16
        )
        self.stable_since = (
            (now if self.stable_since is None else self.stable_since) if good else None
        )
        if self.stable_since is not None and now - self.stable_since >= 0.30 - 1e-9:
            self.active, self.settle_until = False, now + 0.30
            self.event(
                now,
                "RECOVERED",
                duration_s=now - self.started,
                learned_motor_s=self.learned_seconds,
            )
            self.last_policy = "stand"
            return "stand"
        policy = "stand"
        if now - self.started >= 3.0 - 1e-9:
            if self.fallback is None:
                self.fallback = RecoveryGuard(self.name, allow_roll=False)
                self.event(
                    now,
                    "RECOVERY_LEGACY_FALLBACK",
                    learned_motor_s=self.learned_seconds,
                )
            policy = self.fallback.choose(now, me) or "stand"
            self.events.extend(
                e
                for e in self.fallback.events[self.fallback_cursor :]
                if e["state"] == "RECOVERY_MOTOR_STARTED"
            )
            self.fallback_cursor = len(self.fallback.events)
        elif (
            self.learned
            and self.last_touch is not None
            and now - self.last_touch <= 0.10 + 1e-9
            and me.upright < 0.94
            and me.z > 0.025
            and self.learned_seconds < 2.0 - 1e-9
        ):
            policy = "recovery_candidate"
        if policy != self.last_policy:
            self.event(
                now,
                "RECOVERY_POLICY_SELECTED",
                policy=policy,
                recent_measured_contact=self.last_touch is not None
                and now - self.last_touch <= 0.10 + 1e-9,
                learned_motor_s=self.learned_seconds,
            )
        self.last_policy = policy
        return policy
