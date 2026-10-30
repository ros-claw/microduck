"""One measured response event per separated, committed attack.

Contact impulses prove interaction; displacement/stagger marks an observable
response. This is not a counterfactual causal estimator or a health system.
Friendly contacts stay physical and never receive an enemy attack reward.
"""

import numpy as np


class StrikeAudit:
    def __init__(self, teams):
        self.teams = teams
        self.baselines, self.claimed, self.events = {}, set(), []

    def observe(self, now, episodes, controllers, before, after):
        new = []
        for episode in episodes:
            a, b = episode["actor_pair"]
            if self.teams[a] == self.teams[b]:
                continue
            for actor, target in ((a,b), (b,a)):
                pilot = controllers[actor]
                attack = pilot.active_attack
                if attack is None or pilot.target != target or attack not in episode["attacks"] or attack in self.claimed:
                    continue
                key = (episode["event_id"], attack)
                if key not in self.baselines:
                    delta = np.array(before[target].xy)-before[actor].xy
                    direction = delta/max(1e-9,np.linalg.norm(delta))
                    self.baselines[key] = (np.array(before[target].xy), before[target].upright, direction)
                origin, up, direction = self.baselines[key]
                impulse = np.array(episode["normal_impulse_on_second_Ns"][:2]) * (1 if target == b else -1)
                transmitted = float(impulse @ direction)
                displacement = float((np.array(after[target].xy)-origin) @ direction)
                stagger = up-after[target].upright >= .08 and float(np.array(after[target].velocity) @ direction) > .04
                if transmitted >= .01 and (displacement >= .01 or stagger) and after[actor].upright > .85:
                    event = dict(time_s=now, state="VALID_STRIKE", actor=actor, target=target, attack_id=attack,
                                 contact_episode_id=episode["event_id"], forward_impulse_Ns=transmitted,
                                 response_displacement_m=displacement, upright_drop=up-after[target].upright,
                                 response_kind="STAGGER" if stagger else "DISPLACEMENT")
                    self.claimed.add(attack)
                    self.events.append(event)
                    new.append(event)
        return new
