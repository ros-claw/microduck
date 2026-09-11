"""A full revolution alone, or an airborne duck next to the rope, is not a skip."""
import math
import pathlib
import sys

import pytest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from microduck_lab.sim.skip_metrics import SkipCycles


def run_cycle_case(case='clean'):
    scorer = SkipCycles(warmup=0.)
    for k in range(401):
        f = k % 100
        crossings=[]
        if f==60 and case!='no_overhead':
            crossings=[(0,-.3,True)]
        if case!='beside_duck':
            if f==95:
                crossings=[(0,.02,False)]
            if f==98:
                crossings=[(1,-.001 if case=='clips_foot' else .02,False)]
        phase=2*math.pi*k/100
        if case=='swinging':
            phase=math.sin(phase)
        scorer.update(t=k/100,phase=phase,crossings=crossings,
            supported=10<f<40 and case!='no_landing',upright=case!='fallen',
            body_contact=case=='head_landing' and f==20,
            rope_contact=case=='rope_contact' and f==97,
            attachment_gap=.02 if case=='loose_rope' else .002,
            collisions_enabled=case!='ghost_rope',
            floor_penetration=.01 if case=='underground' else 0.)
    return scorer.result()


def test_geometric_passes_with_clean_landing():
    result=run_cycle_case()
    assert result['full_revolutions']==3
    assert result['clean_skips']==3


@pytest.mark.parametrize('case',['beside_duck','clips_foot','no_landing','head_landing',
                                 'rope_contact','fallen','no_overhead','loose_rope','ghost_rope','underground'])
def test_invalid_revolutions_never_count(case):
    result=run_cycle_case(case)
    assert result['full_revolutions']==3
    assert result['clean_skips']==0


def test_oscillation_is_not_a_full_revolution():
    result=run_cycle_case('swinging')
    assert result['full_revolutions']==0
    assert result['physical_success_rate'] is None
    assert not result['passed']
