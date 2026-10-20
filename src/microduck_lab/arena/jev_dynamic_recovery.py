"""Persistent motor-only recovery after the archived one-shot attempt stalls.

No pose writes, collision exclusions, external forces or enlarged torque limits.
Stand/get-up/roll policies all use the ordinary proprioceptive action contract.
"""

from collections import deque

from .jev_recovery import RecoveryMotor


class DynamicRecoveryMotor(RecoveryMotor):
    def __init__(self, role):
        super().__init__(role)
        self.retry_count = 0
        self.retry_start = None
        self.retry_policy = None
        self.retry_after = None
        self.postures = deque()
        self.initial_skill = None
        self.initial_roll_done = False
        self.settle_until = None

    def choose(self, obs, plan):
        was_recovering = self.recovering
        result = super().choose(obs, plan)
        now = obs["sim_time_s"]
        if was_recovering and not self.recovering:
            self.settle_until = now + 0.3
        if not self.recovering and self.settle_until is not None:
            if now < self.settle_until:
                return "stand", (0, 0, 0), "LOCAL_RECOVERY_SETTLE"
            if now < self.settle_until + 0.5 and result[0] == "walk":
                factor = max(0.2, (now - self.settle_until) / 0.5)
                return result[0], tuple(v * factor for v in result[1]), result[2]
        return result

    def control(self, obs, plan):
        result = super().control(obs, plan)
        now, r = obs["sim_time_s"], obs["robot"]
        if not self.recovering:
            self.retry_count = 0
            self.retry_start = self.retry_policy = self.retry_after = None
            self.postures.clear()
            self.initial_skill = None
            self.initial_roll_done = False
            return result
        if result[2] == "LOCAL_RECOVERY_SKILL" and self.retry_count == 0:
            if self.initial_skill is None:
                self.initial_skill = (
                    "roulade"
                    if self.first_attempt_s > 1.0 and r["up_cos"] < 0.35
                    else "sitstand"
                )
                if self.initial_skill == "roulade":
                    self.events.append(
                        dict(
                            time=now,
                            state="RECOVERY_ROLL_STARTED",
                            body_id=r["body_id"],
                            maximum_duration_s=0.5,
                            phase="first_moving_fall_attempt",
                        )
                    )
            if self.initial_skill == "roulade":
                elapsed = now - self.skill_start
                if elapsed >= 0.1 and r["up_cos"] > 0.7:
                    self.initial_roll_done = True
                if elapsed < 0.5 - 1e-9 and not self.initial_roll_done:
                    return "roulade", (0, 0, 0), "LOCAL_RECOVERY_ROLL"
                return "stand", (0, 0, 0), "LOCAL_RECOVERY"
        self.postures.append((now, r["up_cos"]))
        while self.postures and now - self.postures[0][0] > 0.4 + 1e-9:
            self.postures.popleft()
        # Preserve natural standing and the original calibrated first burst.
        if self.retry_after is None:
            self.retry_after = self.recovery_since + self.first_attempt_s + 0.8 + 1.2
        if self.retry_start is not None:
            duration = 0.5 if self.retry_policy == "roulade" else 0.8
            if now - self.retry_start < duration - 1e-9:
                return (
                    self.retry_policy,
                    (0, 0, 0),
                    "LOCAL_RECOVERY_ROLL"
                    if self.retry_policy == "roulade"
                    else "LOCAL_RECOVERY_SKILL",
                )
            self.retry_start = None
        # Do not interrupt a body already moving toward upright; wait for the
        # support/stability criterion in the ordinary recovery controller.
        improving = len(self.postures) > 1 and r["up_cos"] - self.postures[0][1] > 0.18
        if (
            now >= self.retry_after - 1e-9
            and r["up_cos"] < 0.70
            and r["z"] > -0.08
            and not improving
        ):
            self.retry_policy = "sitstand" if self.retry_count % 2 == 0 else "roulade"
            self.retry_count += 1
            self.retry_start = now
            duration = 0.5 if self.retry_policy == "roulade" else 0.8
            self.retry_after = now + duration + 1.6
            self.events.append(
                dict(
                    time=now,
                    state="RECOVERY_RETRY",
                    body_id=r["body_id"],
                    policy=self.retry_policy,
                    attempt=self.retry_count,
                    maximum_duration_s=duration,
                    observed_up_cos=r["up_cos"],
                )
            )
            return (
                self.retry_policy,
                (0, 0, 0),
                "LOCAL_RECOVERY_ROLL"
                if self.retry_policy == "roulade"
                else "LOCAL_RECOVERY_SKILL",
            )
        return result
