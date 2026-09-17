"""An exhaustive observer must measure contact forces without changing dynamics."""
import importlib.util
from pathlib import Path
import mujoco
import numpy as np

spec=importlib.util.spec_from_file_location('audit_circus_contacts',Path(__file__).resolve().parents[1]/'scripts/audit_circus_contacts.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_contact_observer_measures_penetration_and_is_read_only():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.0002"/><worldbody>
    <body name="sky/trunk_base"><geom name="sky/box" type="box" size=".1 .1 .1"/></body>
    <body name="rope/segment" pos="0 0 .105"><freejoint/><geom name="rope/rope_s0" type="sphere" size=".01" mass=".1"/></body>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    audit=module.FullContactAudit();audit.setup(m,d,{'rope_body_ids':[]})
    before=(d.qpos.copy(),d.qvel.copy(),d.qacc.copy(),d.efc_force.copy())
    audit.physics(m,d)
    for actual,expected in zip((d.qpos,d.qvel,d.qacc,d.efc_force),before):np.testing.assert_array_equal(actual,expected)
    s=audit.stats['rope/sky']
    assert abs(s['max_penetration_m']-.005)<1e-10
    assert s['max_normal_force_n']>0 and s['contact_samples']==1
    assert s['normal_impulse_ns']==s['max_normal_force_n']*m.opt.timestep
