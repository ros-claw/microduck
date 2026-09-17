"""Relay state transitions consume physical evidence, never animation timers."""
from dataclasses import dataclass,field
from .mission import RelayRequest


@dataclass
class RelayGate:
    request: RelayRequest = field(default_factory=RelayRequest)
    stage: str = 'sky_solo'
    count: int = 0
    events: list = field(default_factory=list)
    failure: str | None = None
    last_cycle_end: float = -1.
    stage_started: float = 9.

    def transition(self,stage,t):
        self.events.append(dict(t=float(t),event='transition',from_stage=self.stage,to_stage=stage))
        self.stage=stage;self.count=0;self.stage_started=t

    def fault(self,reason,t):
        if self.failure is None:
            self.failure=reason;self.events.append(dict(t=float(t),event='failure',stage=self.stage,reason=reason))

    def consume(self,sky,graphite,*,t,graphite_inside,sky_outside,finale_supported=False):
        if self.failure or self.stage=='complete':return
        if self.stage=='finale':
            if finale_supported:self.transition('complete',t)
            return
        if self.stage=='exit' and sky_outside:
            self.transition('graphite_solo',t)
            return
        source=graphite if self.stage=='graphite_solo' else sky
        if source is None or source['end']<=self.last_cycle_end:return
        self.last_cycle_end=source['end']
        if source['start']<self.stage_started:return
        matching=graphite is not None and abs(graphite['end']-source['end'])<.002
        if self.stage=='entry':
            if graphite_inside and source['clean'] and matching and graphite['clean']:
                self.transition('duo',t)
            return
        if self.stage=='exit':return
        clean=source['clean'] and (self.stage!='duo' or (matching and graphite['clean']))
        self.count=self.count+1 if clean else 0
        self.events.append(dict(t=float(t),event='cycle',stage=self.stage,clean=bool(clean),consecutive=self.count))
        required={'sky_solo':self.request.sky_cycles,'duo':self.request.duo_cycles,'graphite_solo':self.request.graphite_cycles}[self.stage]
        if self.count>=required:
            self.transition({'sky_solo':'entry','duo':'exit','graphite_solo':'finale'}[self.stage],t)

    def result(self):
        return dict(passed=self.stage=='complete' and self.failure is None,stage=self.stage,
                    failure=self.failure,events=self.events,remaining_consecutive=self.count)


def joint_cycle_result(sky,graphite):
    """Two individually good runs are insufficient: the same cycles must pass."""
    pairs=[]
    for a in sky['cycles']:
        match=next((b for b in graphite['cycles'] if abs(a['start']-b['start'])<.002 and abs(a['end']-b['end'])<.002),None)
        pairs.append(dict(start=a['start'],end=a['end'],clean=bool(a['clean'] and match and match['clean'])))
    clean=sum(p['clean'] for p in pairs);count=len(pairs)
    return dict(clean_cycles=clean,full_cycles=count,success_rate=clean/count if count else None,
                passed=bool(sky['passed'] and graphite['passed'] and count>=20 and clean/count>=.8),cycles=pairs)
