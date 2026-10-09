"""Recovery skill transition tied to recovery phase, rather than speed jitter."""

from .jev_fairness import FairMotor
from .jev_tactics import JevMotor


class RecoveryMotor(FairMotor):
    def __init__(self, role):
        super().__init__(role)
        self.first_attempt_s = None

    def control(self, obs, plan):
        if self.recovering and self.first_attempt_s is None:
            # A duck already lying flat uses the calibrated short transition.
            # A moving fall first gets two seconds to finish natural standing;
            # interrupting it at 0.8 s regressed recorded-impact experiments.
            self.first_attempt_s = .8 if obs["robot"]["up_cos"] < .3 else 2.0
        result = JevMotor.choose(self, obs, plan)
        r, now = obs["robot"], obs["sim_time_s"]
        if result[2] == "LOCAL_RECOVERY":
            # Still not stably upright after the initial stand attempt. A fixed
            # phase avoids missing the tested transition due to momentary lean
            # or low-speed threshold crossings. Never repeat within one fall.
            if self.skill_start is None and now-self.recovery_since >= self.first_attempt_s-1e-9:
                self.skill_start = now
                self.events.append(dict(time=now, state="RECOVERY_SKILL_STARTED", body_id=r["body_id"], policy="sitstand", maximum_duration_s=.8))
            if self.skill_start is not None and now-self.skill_start < .8-1e-9:
                return "sitstand", (0, 0, 0), "LOCAL_RECOVERY_SKILL"
        else:
            self.skill_start = None
            self.first_attempt_s = None
        return result
