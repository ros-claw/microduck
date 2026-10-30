"""Persistent, public-state recovery guard adapted from the reviewed get-up work.

Every selected policy is a normal 61D motor. Timers bound attempts; measured
upright, speed and real foot support determine completion. No pose/force writes.
"""

from collections import deque


class RecoveryGuard:
    def __init__(self, name):
        self.name = name
        self.active = False
        self.started = None
        self.stable_since = None
        self.delay = 0.8
        self.initial_upright = 1.0
        self.attempt = 0
        self.burst_since = None
        self.burst_policy = None
        self.next_attempt = 0.0
        self.settle_until = 0.0
        self.postures = deque()
        self.events = []

    def event(self, now, state, **kw):
        self.events.append(dict(time_s=now, state=state, actor=self.name, **kw))

    def choose(self, now, me):
        if not self.active and me.upright < .80:
            self.active, self.started, self.stable_since = True, now, None
            self.initial_upright = me.upright
            self.delay = .8 if me.upright < .3 else 2.0
            self.attempt, self.burst_since, self.burst_policy = 0, None, None
            self.next_attempt = now + self.delay
            self.postures.clear()
            self.event(now, "RECOVERY_STARTED", upright=me.upright)
        if not self.active:
            return "stand" if now < self.settle_until else None
        good = me.upright > .94 and me.foot_support and me.speed < .16
        self.stable_since = (self.stable_since if self.stable_since is not None else now) if good else None
        if self.stable_since is not None and now - self.stable_since >= .3 - 1e-9:
            self.active, self.settle_until = False, now + .3
            self.event(now, "RECOVERED", duration_s=now-self.started, attempts=self.attempt)
            return "stand"
        self.postures.append((now, me.upright))
        while self.postures and now - self.postures[0][0] > .4 + 1e-9:
            self.postures.popleft()
        if self.burst_since is not None:
            elapsed = now - self.burst_since
            duration = .5 if self.burst_policy == "roulade" else .8
            if self.burst_policy == "roulade" and elapsed >= .1 and me.upright > .7:
                elapsed = duration
            if elapsed < duration - 1e-9:
                return self.burst_policy
            self.burst_since = None
        improving = len(self.postures) > 1 and me.upright - self.postures[0][1] > .18
        if now >= self.next_attempt - 1e-9 and me.upright < .7 and me.z > -.08 and not improving:
            first_dynamic_roll = self.attempt == 0 and self.initial_upright >= .3 and me.upright < .35
            self.burst_policy = "roulade" if first_dynamic_roll or self.attempt % 2 else "sitstand"
            self.attempt += 1
            self.burst_since = now
            duration = .5 if self.burst_policy == "roulade" else .8
            self.next_attempt = now + duration + 1.6
            self.event(now, "RECOVERY_MOTOR_STARTED", policy=self.burst_policy,
                       maximum_duration_s=duration, attempt=self.attempt, upright=me.upright)
            return self.burst_policy
        return "stand"
