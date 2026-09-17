"""Small, declared command grammar; not an unrestricted language-model planner."""
from dataclasses import dataclass,asdict
import math
import re


@dataclass(frozen=True)
class RelayRequest:
    sky_cycles: int = 5
    duo_cycles: int = 3
    graphite_cycles: int = 5

    def __post_init__(self):
        if any(type(n) is not int or not 1 <= n <= 100 for n in asdict(self).values()):
            raise ValueError('Cycle counts must be integers in [1, 100]')

    def plan(self):
        specs=[('sky_solo','rope_skip',self.sky_cycles,'REHEARSE'),
               ('entry','enter_moving_rope',None,'PRACTICE'),
               ('duo','synchronized_jump',self.duo_cycles,'PRACTICE'),
               ('exit','exit_moving_rope',None,'PRACTICE'),
               ('graphite_solo','rope_skip',self.graphite_cycles,'REHEARSE'),
               ('finale','team_bow',None,'REHEARSE')]
        return dict(kind='relay',request=asdict(self),status='NEEDS_PRACTICE',
                    planner='bounded bilingual command grammar',
                    stages=[dict(id=i,skill=s,cycles=n,status=status,depends_on=[] if k==0 else [specs[k-1][0]])
                            for k,(i,s,n,status) in enumerate(specs)],
                    promotion='Independent full-mission physical verification on disjoint holdout seeds')


def parse_request(text):
    """Reject unsupported/ambiguous goals instead of inventing capabilities."""
    speed=re.search(r'(?:加快|加速|快|faster\s*(?:by)?|speed\s*up\s*(?:by)?)\s*(\d+(?:\.\d+)?)\s*%',text,re.I)
    if speed:
        percent=float(speed[1])
        if not 0 < percent <= 100:raise ValueError('Speed increase must be in (0, 100]%')
        return dict(kind='speed',increase_fraction=percent/100,status='NEEDS_PRACTICE',
                    planner='bounded bilingual command grammar',
                    stages=['measure_baseline','curriculum_trials','holdout_gate','rehearse'])
    if not re.search('Sky',text,re.I) or not re.search('Graphite',text,re.I):
        raise ValueError('Supported requests: Sky/Graphite 5-3-5 relay, or faster by N%. Other circus skills are not implemented.')
    # The canonical relay has three explicitly numbered jump sections.
    counts=[int(n) for n in re.findall(r'(\d+)\s*(?:下|次|圈|jumps?|skips?)',text,re.I)]
    if len(counts)!=3 or not re.search(r'一起|同时|together',text,re.I):
        raise ValueError('Specify three jump counts: Sky solo, both together, Graphite solo.')
    return RelayRequest(*counts).plan()


def speed_curriculum(measured_hz,increase_fraction,steps=4):
    if not math.isfinite(measured_hz) or measured_hz<=0 or not 0<increase_fraction<=1 or steps<1:
        raise ValueError('Invalid measured cadence/curriculum')
    return [measured_hz*(1+increase_fraction*i/steps) for i in range(steps+1)]
