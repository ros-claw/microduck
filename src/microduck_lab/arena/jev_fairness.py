"""Second calibration: warned-floor evacuation and earlier posture recovery.

These are explicit local reflexes, not claims about Jev's intelligence.
The original live Jev capture and its exact replay remain reproducible.
"""

import math

from .jev_tactics import JevMotor, public_route
from .progressive_collapse import ProgressiveCollapse, neighbours


class FairCollapse(ProgressiveCollapse):
    warning_s = 4.0


class FairMotor(JevMotor):
    def __init__(self, role):
        super().__init__(role)
        self.evacuating = False
        self.skill_start = None

    def choose(self, obs, plan):
        now, r = obs["sim_time_s"], obs["robot"]
        if r["up_cos"] < .82 and not self.recovering:
            self.recovering = True
            self.recovery_since = now
            self.skill_start = None
            self.events.append(dict(time=now, state="RECOVERY_STARTED", body_id=r["body_id"], trigger="early_posture_reflex"))
        ts = {t["id"]: t for t in obs["visible_tiles"]}
        anchor = r["current_tile_id"]
        if anchor is None:
            anchor = min(ts, key=lambda i: math.hypot(r["x"]-ts[i]["centre_xy"][0], r["y"]-ts[i]["centre_xy"][1]))
        self.evacuating = False
        warned = ts[anchor]["warning_remaining_s"] is not None
        departing = (plan is not None and plan["expires_s"] >= now and plan["tile"] != anchor
                     and plan["tile"] is not None and public_route(obs, plan["tile"]) is not None)
        if warned and not departing:
            destinations = [i for i in neighbours(anchor, ts) if ts[i]["state"] == "LOCKED"
                            and public_route(obs, i) is not None]
            if destinations:
                dest = min(destinations, key=lambda i: (
                    sum(max(0., .4-math.hypot(o["x"]-ts[i]["centre_xy"][0], o["y"]-ts[i]["centre_xy"][1])) for o in obs["nearby_ducks"])
                    + .15*math.hypot(*ts[i]["centre_xy"]), i))
                emergency = dict(action="EVADE", tile=dest, expires_s=now+.04)
                policy, command, intent = self.control(obs, emergency)
                if intent == "JEV_EVADE":
                    self.evacuating = True
                    return policy, command, "LOCAL_EVACUATE"
                return policy, command, intent
        return self.control(obs, plan)

    def control(self, obs, plan):
        result = super().choose(obs, plan)
        r, now = obs["robot"], obs["sim_time_s"]
        if result[2] == "LOCAL_RECOVERY":
            if (self.skill_start is None and now-self.recovery_since >= .8
                    and r["up_cos"] < .3 and r["speed_m_s"] < .08 and r["z"] > .015):
                self.skill_start = now
                self.events.append(dict(time=now, state="RECOVERY_SKILL_STARTED", body_id=r["body_id"], policy="sitstand", maximum_duration_s=.8))
            if self.skill_start is not None and now-self.skill_start < .8:
                return "sitstand", (0, 0, 0), "LOCAL_RECOVERY_SKILL"
        else:
            self.skill_start = None
        return result
