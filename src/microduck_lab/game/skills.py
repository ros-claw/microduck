"""Bounded motor-policy transitions; no root-pose or velocity writes."""
import math
import numpy as np
from .state import LANES

class SkillRuntime:
    def __init__(self,duck):
        self.duck=duck;self.action='BRAKE';self.target_lane=1;self.until=0.;self.roll_ready_at=0.;self.events=[]
    def locked(self,t):return t<self.until
    def apply(self,action,t):
        if self.locked(t):return False
        if action=='BRAKE' and self.action!='BRAKE':self.until=t+.6
        if action=='ROULADE':self.until=t+2.8;self.roll_ready_at=t+5.;self.roll_started=t
        if action in ('DODGE_LEFT','DODGE_RIGHT'):
            self.target_lane=int(np.clip(self.target_lane+(1 if action=='DODGE_LEFT' else -1),0,2));self.until=t+1.0
        if action!=self.action:self.events.append(dict(t=t,action=action,lane=self.target_lane))
        self.action=action;return True
    def step(self,t):
        r=self.duck;r.command[:]=0
        if self.action=='ROULADE':
            if t<self.roll_started+2:r.active_policy='roulade'
            elif t<self.until:r.active_policy='stand'
            else:self.action='BRAKE';r.active_policy='stand'
        elif self.action in ('KEEP_RUNNING','DODGE_LEFT','DODGE_RIGHT'):
            r.active_policy='run';delta=LANES[self.target_lane]-r.trunk_pos()[1]
            desired=math.atan2(delta,.6);error=math.atan2(math.sin(desired-r.trunk_yaw()),math.cos(desired-r.trunk_yaw()))
            r.command[0]=.5 if abs(error)<.65 else .28
            r.command[2]=np.clip(3*error,-1.5,1.5)
        else:
            yaw=r.trunk_yaw()
            r.active_policy='run' if np.linalg.norm(r.trunk_linvel()[:2])>.03 or abs(yaw)>.07 else 'stand'
            r.command[2]=np.clip(-3*yaw,-1.5,1.5)
        r.step()
