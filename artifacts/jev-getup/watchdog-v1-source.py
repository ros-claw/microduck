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

    def control(self, obs, plan):
        result = super().control(obs, plan)
        now, r = obs['sim_time_s'], obs['robot']
        if not self.recovering:
            self.retry_count = 0
            self.retry_start = self.retry_policy = self.retry_after = None
            self.postures.clear()
            return result
        self.postures.append((now, r['up_cos']))
        while self.postures and now-self.postures[0][0] > .4+1e-9:
            self.postures.popleft()
        # Preserve natural standing and the original calibrated first burst.
        if self.retry_after is None:
            self.retry_after = self.recovery_since+self.first_attempt_s+.8+1.2
        if self.retry_start is not None:
            duration = .5 if self.retry_policy == 'roulade' else .8
            if now-self.retry_start < duration-1e-9:
                return self.retry_policy, (0, 0, 0), 'LOCAL_RECOVERY_ROLL' if self.retry_policy == 'roulade' else 'LOCAL_RECOVERY_SKILL'
            self.retry_start = None
        # Do not interrupt a body already moving toward upright; wait for the
        # support/stability criterion in the ordinary recovery controller.
        improving = len(self.postures)>1 and r['up_cos']-self.postures[0][1] > .18
        if (now >= self.retry_after-1e-9 and r['up_cos'] < .70
                and r['z'] > -.08 and not improving):
            self.retry_policy = 'sitstand' if self.retry_count % 2 == 0 else 'roulade'
            self.retry_count += 1
            self.retry_start = now
            duration = .5 if self.retry_policy == 'roulade' else .8
            self.retry_after = now+duration+1.6
            self.events.append(dict(time=now, state='RECOVERY_RETRY', body_id=r['body_id'],
                                    policy=self.retry_policy, attempt=self.retry_count,
                                    maximum_duration_s=duration, observed_up_cos=r['up_cos']))
            return self.retry_policy, (0, 0, 0), 'LOCAL_RECOVERY_ROLL' if self.retry_policy == 'roulade' else 'LOCAL_RECOVERY_SKILL'
        return result
