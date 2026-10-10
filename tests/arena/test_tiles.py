"""Physical regressions use real MuJoCo and delivered motor policies."""

import mujoco
import numpy as np
import pytest
from microduck_lab.arena.world import build_world


def test_locked_load_and_gravity_release_without_ghost_floor():
    m, d, duck = build_world()
    assert not np.any(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert m.nmocap == 0 and len(duck.act_ids) == 14
    eid = m.equality("tile_1_support").id
    jid = m.joint("tile_1_free").id
    q = int(m.jnt_qposadr[jid])
    for i in range(2000):
        if i % 40 == 0:
            duck.step()
        mujoco.mj_step(m, d)
    assert abs(d.qpos[q + 2] + 0.025) < 0.001
    assert duck.trunk_pos()[2] > 0.10
    before = d.qpos.copy()
    vel = d.qvel.copy()
    d.eq_active[eid] = 0
    np.testing.assert_array_equal(d.qpos, before)
    np.testing.assert_array_equal(d.qvel, vel)
    for i in range(1000):
        if i % 40 == 0:
            duck.step()
        mujoco.mj_step(m, d)
    assert d.qpos[q + 2] < -0.4
    assert duck.trunk_pos()[2] < -0.2
    assert not any(w.number for w in d.warning)
    assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
    assert np.max(np.abs(d.actuator_force)) <= 0.640500001


def test_unoccupied_tile_acceleration_and_independent_welds():
    m, d, _ = build_world()
    eid = m.equality("tile_8_support").id
    jid = m.joint("tile_8_free").id
    q = int(m.jnt_qposadr[jid])
    v = int(m.jnt_dofadr[jid])
    d.eq_active[eid] = 0
    mujoco.mj_step(m, d, 40)
    assert d.qvel[v + 2] == pytest.approx(-9.81 * 0.02, abs=0.003)
    assert d.qpos[q + 2] < -0.026
    assert sum(d.eq_active) == 8


def test_invalid_motor_period_rejected():
    with pytest.raises(ValueError):
        build_world(dt=0.003)
