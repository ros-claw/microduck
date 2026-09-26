import dataclasses,json,time,threading
import numpy as np
import pytest
from microduck_lab.game.chunks import generate
from microduck_lab.game.state import candidates,lane_clear
from microduck_lab.jev.client import make_request,validate_response
from microduck_lab.jev.decision_loop import DecisionLoop

def state():
    return dict(robot=dict(upright=1.,target_lane=1,airborne=False,speed_m_s=0.,y_m=0.,yaw_rad=0.),hazards=[])

def response(probabilities=None,choice='BRAKE',confidence=.8):
    return dict(model='test',answers=dict(next_action=dict(type='choice',choice=choice,probabilities=probabilities or {'BRAKE':.8,'KEEP_RUNNING':.2},confidence=confidence),act_now=dict(noul=.9),need_system_two=dict(noul=.1),situation_risk=dict(score=1.)),usage={})

def test_level_seed_changes_physical_layout():
    a=[dataclasses.asdict(x) for x in generate(101)]
    assert a==[dataclasses.asdict(x) for x in generate(101)]
    b=[dataclasses.asdict(x) for x in generate(102)]
    assert [(x['kind'],x['y'],x['phase']) for x in a]!=[(x['kind'],x['y'],x['phase']) for x in b]

def test_illegal_skills_removed_before_request():
    s=state();s['robot']['airborne']=True
    legal=candidates(s)
    assert 'JUMP' not in legal and 'ROULADE' not in legal and 'DODGE_LEFT' not in legal
    assert candidates(s,locked=True)==['CONTINUE']
    body=make_request(s,legal)
    assert set(body['questions']['next_action']['criteria'])==set(legal)

def test_roll_requires_geometry_pose_and_stopped_body():
    s=state();s['hazards']=[dict(kind='high_bar',distance_m=.31,lateral_m=0.,half_width_m=.85,height_m=.29)]
    assert 'ROULADE' in candidates(s)
    s['robot']['speed_m_s']=.2
    assert 'ROULADE' not in candidates(s)
    s['robot']['speed_m_s']=0.;s['hazards'][0]['height_m']=.24
    assert 'ROULADE' not in candidates(s)
    assert not lane_clear(s,0)

def test_probability_rounding_and_invalid_outputs():
    a=response({'BRAKE':.49,'KEEP_RUNNING':.50},choice='KEEP_RUNNING')
    assert validate_response(a,['BRAKE','KEEP_RUNNING'])['probabilities']==a['answers']['next_action']['probabilities']
    for bad in [response({'BRAKE':.1,'KEEP_RUNNING':.2}),response(choice='FLY'),response(confidence=float('nan'))]:
        with pytest.raises(ValueError):validate_response(bad,['BRAKE','KEEP_RUNNING'])

def decision(action='KEEP_RUNNING',confidence=.9):
    return dict(action=action,confidence=confidence,need_system_two=.1,latency_ms=1.,probabilities={action:1.})

class DelayedClient:
    def __init__(self):self.release=threading.Event()
    def decide(self,*args):self.release.wait(1);return decision()

def test_network_does_not_block_physics_and_late_result_is_discarded():
    client=DelayedClient();loop=DecisionLoop(client,deadline=.015)
    try:
        start=time.monotonic();assert loop.request(state(),['BRAKE','KEEP_RUNNING'],0)
        assert time.monotonic()-start<.1
        assert not loop.request(state(),['BRAKE'],.01)
        time.sleep(.025);assert loop.poll(['BRAKE','KEEP_RUNNING'],.02)=='BRAKE'
        client.release.set();loop.future.result(timeout=1)
        assert loop.poll(['BRAKE','KEEP_RUNNING'],.04)=='BRAKE'
        assert loop.records[-1]['status']=='stale'
    finally:client.release.set();loop.close()

def test_result_revalidated_and_skill_commitment_preserved():
    class Immediate:
        def decide(self,*args):return decision()
    loop=DecisionLoop(Immediate())
    try:
        loop.request(state(),['BRAKE','KEEP_RUNNING'],0);loop.future.result(timeout=1)
        assert loop.poll(['BRAKE'],.1)=='BRAKE'
        assert loop.records[-1]['status']=='no_longer_legal'
        loop.request(state(),['BRAKE','KEEP_RUNNING'],.2);loop.future.result(timeout=1)
        assert loop.poll(['CONTINUE'],.3,locked=True) is None
        assert loop.records[-1]['status']=='skill_committed'
    finally:loop.close()

def test_contact_invalidates_a_crossing_and_timer_is_not_a_roll():
    import mujoco
    from microduck_lab.game.chunks import Chunk
    from microduck_lab.game.scoring import GameAudit
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1" pos="0 0 -1"/>
    <body name="duck/trunk_base"><geom type="box" size=".1 .1 .1"/></body>
    <body name="bar" pos="0 0 .105"><freejoint/><geom type="sphere" size=".01" mass=".1"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    class Duck:
        trunk_body_id=int(m.body('duck/trunk_base').id)
        def trunk_pos(self):return np.array([1.,0.,0.])
    audit=GameAudit(m,[Chunk('bar','high_bar',0,0)])
    audit.begin_roll(0);d.time=3.;audit.physics(m,d,Duck());audit.checkpoints(Duck())
    result=audit.result()
    assert result['hit']==['bar'] and result['cleared']==[] and result['score']==0
    assert result['max_hazard_penetration_m']>.004
    assert result['rolls'][0]['clean'] is False

def test_roll_has_a_hysteresis_window_after_braking():
    s=state();s['hazards']=[dict(kind='high_bar',distance_m=.38,lateral_m=0.,half_width_m=.85,height_m=.29)]
    legal=candidates(s)
    assert 'ROULADE' in legal and 'KEEP_RUNNING' not in legal
