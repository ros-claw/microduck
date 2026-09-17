import hashlib
import math
from pathlib import Path
import pytest
import mujoco
import numpy as np
from microduck_lab.circus.mission import parse_request,RelayRequest,speed_curriculum
from microduck_lab.circus.relay import RelayGate
from microduck_lab.circus.promotion import promotion_gate
from microduck_lab.sim.classic_rope import build_classic_world
from microduck_lab.sim.composer import DuckSpec
from microduck_lab.sim.skip_metrics import PhysicalSkipAudit


def cycle(t,clean=True):return dict(start=t-.32,end=t,clean=clean)


def test_relay_language_preserves_requested_counts_and_marks_missing():
    a=parse_request('Sky 先跳 5 下，Graphite 进去，两只一起跳 3 下，Sky 退出，Graphite 再跳 5 下。')
    b=parse_request('Sky 5 jumps, Graphite enters, together 3 jumps, Sky exits, Graphite 5 jumps.')
    assert a==b
    assert [s['cycles'] for s in a['stages']]==[5,None,3,None,5,None]
    assert a['status']=='NEEDS_PRACTICE'
    assert a['stages'][1]['status']=='PRACTICE'


@pytest.mark.parametrize('text',['两只鸭子双摇','Sky and Graphite jump together','Sky 0 jumps together Graphite 3 jumps then 5 jumps'])
def test_unknown_or_invalid_program_fails_closed(text):
    with pytest.raises(ValueError):parse_request(text)


def test_acceleration_is_a_curriculum_from_measured_rate():
    assert parse_request('速度加快 30%')['increase_fraction']==.3
    assert parse_request('faster by 30%')['increase_fraction']==.3
    assert speed_curriculum(3.1,.3)[-1]==pytest.approx(4.03)
    with pytest.raises(ValueError):speed_curriculum(float('nan'),.3)


def test_no_timer_or_single_duck_counter_can_complete_relay():
    g=RelayGate(RelayRequest(1,1,1))
    g.consume(None,None,t=100,graphite_inside=True,sky_outside=True)
    assert g.stage=='sky_solo'
    g.consume(cycle(10),None,t=10,graphite_inside=False,sky_outside=False)
    assert g.stage=='entry'
    g.consume(cycle(11),cycle(11,False),t=11,graphite_inside=True,sky_outside=False)
    assert g.stage=='entry'
    g.consume(cycle(12),cycle(12),t=12,graphite_inside=True,sky_outside=False)
    assert g.stage=='duo'
    g.consume(cycle(13),cycle(12.9),t=13,graphite_inside=True,sky_outside=False)
    assert g.stage=='duo'
    g.consume(cycle(14),cycle(14),t=14,graphite_inside=True,sky_outside=False)
    assert g.stage=='exit'
    g.consume(cycle(15),cycle(15),t=15,graphite_inside=True,sky_outside=False)
    assert g.stage=='exit'
    g.consume(cycle(16),cycle(16),t=16,graphite_inside=True,sky_outside=True)
    assert g.stage=='graphite_solo'
    g.consume(cycle(17),cycle(17),t=17,graphite_inside=True,sky_outside=True)
    assert g.stage=='finale' and not g.result()['passed']
    g.consume(None,None,t=18,graphite_inside=True,sky_outside=True,finale_supported=True)
    assert g.result()['passed']


def test_failed_cycle_resets_consecutive_count_and_fault_blocks_promotion():
    g=RelayGate(RelayRequest(2,1,1))
    for t,ok in [(10,True),(11,False),(12,True)]:
        g.consume(cycle(t,ok),None,t=t,graphite_inside=False,sky_outside=False)
    assert g.stage=='sky_solo' and g.count==1
    g.fault('rope_contact',12.1)
    g.consume(cycle(13),cycle(13),t=13,graphite_inside=True,sky_outside=True,finale_supported=True)
    assert not g.result()['passed']


def test_darwin_rejects_overlap_partial_tasks_and_missing_evidence():
    results=[dict(kind='relay',seed=s,passed=True,error=None,evidence_sha256='a'*64,protocol_sha256='b'*64) for s in [701,702,703,704]]
    assert promotion_gate([0,1],results,'relay')['promoted']
    assert not promotion_gate([701],results,'relay')['promoted']
    assert not promotion_gate([0],results,'entry')['promoted']
    results[0]['passed']=False
    assert not promotion_gate([0],results,'relay')['promoted']
    results[0]['passed']=True;results[0]['evidence_sha256']=''
    assert not promotion_gate([0],results,'relay')['promoted']


def test_fourth_duck_has_real_rope_contacts_and_separate_audit():
    ducks=[DuckSpec('lavender',(-.224,0),0),DuckSpec('cream',(.224,0),math.pi),
           DuckSpec('sky',(0,0),math.pi/2),DuckSpec('graphite',(0,-.4),math.pi/2)]
    m,d,info=build_classic_world(ducks=ducks,rope_kind='triple',connect_to='handles',rope_contacts='full')
    rope=m.geom('rope/rope_s12').id;foot=m.geom('graphite/left_foot_collision').id
    assert m.geom_contype[rope]&m.geom_conaffinity[foot]
    assert m.body_conaffinity[m.geom_bodyid[foot]]&8
    assert m.nmocap==0
    audit=PhysicalSkipAudit('graphite');audit.setup(m,d,info)
    assert audit.trunk==m.body('graphite/trunk_base').id
    assert audit.enabled
    # A real forced overlap must generate a solver contact, not just compatible masks.
    rt=m.body('rope/ropewrap').jntadr[0];adr=m.jnt_qposadr[rt]
    mujoco.mj_forward(m,d)
    d.qpos[adr:adr+3]+=d.geom_xpos[foot]-d.geom_xpos[rope]
    mujoco.mj_forward(m,d)
    assert any(c.efc_address>=0 and ((audit.rope[c.geom1] and audit.sky[c.geom2]) or (audit.rope[c.geom2] and audit.sky[c.geom1])) for c in d.contact)


def test_entry_boundary_checks_whole_robot_not_only_root():
    from microduck_lab.circus.geometry import outside_rope_region
    m,d,info=build_classic_world(rope_kind='triple',connect_to='handles',rope_contacts='full',rope_length=.73)
    audit=PhysicalSkipAudit();audit.setup(m,d,info)
    joint=m.body('sky/trunk_base').jntadr[0];q=m.jnt_qposadr[joint]
    d.qpos[q+1]=.40;mujoco.mj_forward(m,d)
    # Root exceeds L/2, but robot geometry still intersects the sweep envelope.
    assert not outside_rope_region(m,d,audit,.73,1)
    d.qpos[q+1]=.60;mujoco.mj_forward(m,d)
    assert outside_rope_region(m,d,audit,.73,1)
    d.qpos[q+1]=-.60;mujoco.mj_forward(m,d)
    assert outside_rope_region(m,d,audit,.73,-1)


def test_duo_requires_success_on_matching_cycles_not_separate_averages():
    from microduck_lab.circus.relay import joint_cycle_result
    sky=dict(passed=True,cycles=[cycle(i, i>=4) for i in range(20)])
    graphite=dict(passed=True,cycles=[cycle(i, i<16) for i in range(20)])
    result=joint_cycle_result(sky,graphite)
    assert result['success_rate']==.6 and not result['passed']
    assert joint_cycle_result(sky,sky)['passed']


def test_integrated_turner_clock_does_not_rephase_when_cadence_changes():
    from microduck_lab.sim.runtime import TurnerDuckRuntime
    r=object.__new__(TurnerDuckRuntime)
    r.clock_t=10.;r.turn_frequency=3.;r.phase_offset_s=0.;r.continuous_turn_phase=True
    first=r.drive_phase()
    r.turn_frequency=2.8
    assert r.drive_phase()==pytest.approx(first)  # no retroactive change to 10 seconds
    r.clock_t+=.02
    assert r.drive_phase()-first==pytest.approx(2*math.pi*2.8*.02)
    r.phase_offset_s=.01
    assert r.drive_phase()-first==pytest.approx(2*math.pi*2.8*.03)
    r.clock_t=0.
    assert r.drive_phase()==pytest.approx(2*math.pi*2.8*.01)


def test_diagnostics_are_read_only_and_record_solver_contact():
    from microduck_lab.circus.diagnostics import FormationDiagnostics
    ducks=[DuckSpec('lavender',(-.3,0),0),DuckSpec('cream',(.3,0),math.pi),
           DuckSpec('sky',(0,0),math.pi/2),DuckSpec('graphite',(.02,0),math.pi/2)]
    m,d,info=build_classic_world(ducks=ducks,rope_kind='triple',connect_to='handles',rope_contacts='full')
    mujoco.mj_forward(m,d)
    observer=FormationDiagnostics();observer.setup(m,d,info)
    before=[v.copy() for v in (d.qpos,d.qvel,d.ctrl)]
    observer.physics(m,d)
    assert any(e['pair']=='graphite/sky' for e in observer.events)
    for a,b in zip(before,(d.qpos,d.qvel,d.ctrl)):np.testing.assert_array_equal(a,b)


def test_diagnostics_do_not_change_rollout():
    from dataclasses import replace
    from microduck_lab.circus.trial import TrialConfig,run_trial
    base=TrialConfig(kind='duo',seconds=.06,span=.8,rope_length=.93,dx=.14)
    a=run_trial(base);b=run_trial(replace(base,diagnostics=True))
    assert a['error'] is None and b['error'] is None
    assert a['positions']==b['positions']
    assert a['jumpers']==b['jumpers']
    assert a['inter_robot_contacts']==b['inter_robot_contacts']
    assert a['diagnostics'] is None and b['diagnostics']['samples']


def test_local_phase_distinguishes_twisted_rope_at_two_jumper_slots():
    from types import SimpleNamespace
    from microduck_lab.circus.geometry import local_rope_phase
    rotation=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    m=SimpleNamespace(geom_size=np.array([[.001,.3,0]]*3))
    d=SimpleNamespace(geom_xmat=np.tile(rotation.reshape(1,9),(5,1)),
        geom_xpos=np.array([[-.6,.2,.1],[0.,0.,.1],[.6,-.2,.1],[-.6,0.,0.],[-.6,0.,0.]]),
        site_xpos=np.array([[-1.,0.,.3],[1.,0.,.3]]))
    audit=SimpleNamespace(rope_ids=np.array([0,1,2]),feet=[3,4],handles=[0,1])
    assert local_rope_phase(m,d,audit)==pytest.approx(math.pi/4)
    d.geom_xpos[3:,0]=.6
    assert local_rope_phase(m,d,audit)==pytest.approx(-math.pi/4)


def test_saved_slow_motion_states_preserve_physical_trial(tmp_path):
    from microduck_lab.circus.trial import TrialConfig,run_trial
    cfg=TrialConfig(kind='duo',seconds=.06,span=.8,rope_length=.86,dx=.14)
    original=run_trial(cfg);captured=run_trial(cfg,capture_dir=tmp_path)
    assert original['positions']==captured['positions']
    assert original['jumpers']==captured['jumpers']
    states=np.load(tmp_path/'trajectory.npz')
    assert len(states['time'])==12
    np.testing.assert_allclose(np.diff(states['time']),.005,atol=1e-10)
    model=mujoco.MjModel.from_binary_path(str(tmp_path/'scene.mjb'))
    assert states['qpos'].shape==(12,model.nq)
    assert states['ctrl'].shape==(12,model.nu)
