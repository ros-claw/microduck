"""TypeSafe HTTP contract, checked against https://docs.typesafe.ai/api ."""
import json,os,time,math
from pathlib import Path
from urllib.request import Request,urlopen

ENDPOINT='https://api.typesafe.ai/v1/systemone'

def make_request(state,candidates,model='jev-latest'):
    return dict(model=model,state=state,questions={
        'next_action':dict(type='choice',instructions='Choose one legal skill for the NEXT 0.8 seconds to make progress toward the finish without collisions. Standing still forever loses the game. Use observed clear lanes to bypass obstacles early. The robot moves only about 0.27 metres per second, so a hazard 1 metre away is not an immediate collision. Respect skill prerequisites; do not assume unobserved future events.',criteria={c:{'KEEP_RUNNING':'Start or continue walking forward in the target lane at about 0.27 m/s. Feedback holds the lane.', 'BRAKE':'Stop; use only when motion is unsafe or preparing a roll. This does not advance to the finish.', 'DODGE_LEFT':'Turn and walk to the adjacent LEFT lane (higher world y), while progressing forward. Candidate filtering found that destination clear.', 'DODGE_RIGHT':'Turn and walk to the adjacent RIGHT lane (lower world y), while progressing forward. Candidate filtering found that destination clear.', 'ROULADE':'Execute a verified 2-second forward roll, travelling about 0.59 m, then stand up. This candidate is offered only near a suitable high bar.'}.get(c,c) for c in candidates}),
        'situation_risk':dict(type='score',instructions='Rate immediate collision or fall risk.',criteria=['safe','mild','high','critical']),
        'act_now':dict(type='noul',instructions='Should the robot change action now rather than continue its current skill?'),
        'need_system_two':dict(type='noul',instructions='Is this situation too ambiguous for a reliable immediate tactical decision?'),
    })

def validate_response(raw,candidates):
    a=raw['answers'];choice=a['next_action'];p=choice['probabilities']
    if choice.get('type')!='choice' or choice['choice'] not in candidates or set(p)!=set(candidates):raise ValueError('Invalid action distribution')
    # Live API rounds probabilities to 2 decimals; keep raw values, tolerate
    # only the maximum accumulated rounding error, not arbitrary distributions.
    values=[float(v) for v in p.values()]
    if not all(math.isfinite(v) and 0<=v<=1 for v in values) or abs(sum(values)-1)>.005*len(values)+1e-8:raise ValueError('Invalid probabilities')
    confidence=float(choice['confidence'])
    if not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid confidence')
    for key in ('act_now','need_system_two'):
        v=float(a[key]['noul'])
        if not math.isfinite(v) or not 0<=v<=1:raise ValueError('Invalid noul')
    risk=float(a['situation_risk']['score'])
    if not math.isfinite(risk) or not 0<=risk<=3:raise ValueError('Invalid risk')
    return dict(action=choice['choice'],probabilities=p,confidence=confidence,act_now=float(a['act_now']['noul']),need_system_two=float(a['need_system_two']['noul']),risk=risk,model=raw['model'],usage=raw.get('usage',{}))

class JevClient:
    def __init__(self,model='jev-latest',timeout=2.):
        self.model=model;self.timeout=timeout
        self.key=os.environ.get('TYPESAFE_API_KEY')
        path=Path.home()/'.config/microduck/typesafe.key'
        if not self.key and path.exists():self.key=path.read_text().strip()
        if not self.key:raise RuntimeError('Set TYPESAFE_API_KEY or ~/.config/microduck/typesafe.key')
    def decide(self,state,candidates):
        body=make_request(state,candidates,self.model)
        request=Request(ENDPOINT,data=json.dumps(body,allow_nan=False).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},method='POST')
        started=time.monotonic()
        with urlopen(request,timeout=self.timeout) as response:raw=json.load(response)
        result=validate_response(raw,candidates)
        return dict(**result,latency_ms=(time.monotonic()-started)*1000,raw=raw)
