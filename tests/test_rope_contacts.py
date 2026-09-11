"""Collision routing must include physics geoms, never visual meshes."""
import pathlib
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from microduck_lab.sim.classic_rope import build_classic_world


@pytest.mark.parametrize("mode", ["off", "floor", "jumper", "full"])
def test_contact_routing(mode):
    m, _, _ = build_classic_world(rope_kind="triple", connect_to="handles",
                                  rope_length=0.5, rope_contacts=mode)
    geom = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    rope, foot, floor = map(geom, ["rope/rope_s12", "sky/left_foot_collision", "floor"])

    def allowed(a, b):
        return bool((m.geom_contype[a] & m.geom_conaffinity[b]) or
                    (m.geom_contype[b] & m.geom_conaffinity[a]))

    assert allowed(rope, floor) == (mode in ("floor", "full"))
    assert allowed(rope, foot) == (mode in ("jumper", "full"))
    assert allowed(foot, floor)
    assert not allowed(rope, geom("rope/rope_s10"))
    assert m.nmocap == 0
    for g in range(m.ngeom):
        if m.geom_group[g] == 2:  # robot visual meshes
            assert not allowed(rope, g)
    # Broadphase caches must include the chosen masks (postcompile geom-only
    # edits can otherwise silently leave contacts disabled).
    assert m.body_contype[m.geom_bodyid[rope]] & 8 == (8 if mode != "off" else 0)


def test_invalid_contact_mode_fails():
    with pytest.raises(ValueError, match="rope_contacts"):
        build_classic_world(rope_contacts="typo")
    with pytest.raises(ValueError, match="triple"):
        build_classic_world(rope_kind="cable", rope_contacts="full")


def test_visible_rope_starts_at_near_attachment():
    m, d, _ = build_classic_world(rope_kind="triple", connect_to="handles",
                                  rope_length=.5, timestep=.001, connect_timeconst=.004)
    mujoco.mj_forward(m, d)
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rope/ropewrap")
    first = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rope/seg_0")
    np.testing.assert_allclose(d.xpos[first], d.xpos[root], atol=1e-12)
    geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "rope/rope_s0")
    # Capsule local z is its long axis. Its actual first endpoint must be
    # the attached point, with no invisible one-segment extension.
    half = d.geom_xmat[geom].reshape(3,3)[:,2] * m.geom_size[geom,1]
    ends = [d.geom_xpos[geom] - half, d.geom_xpos[geom] + half]
    assert min(np.linalg.norm(end - d.xpos[root]) for end in ends) < 1e-12


def test_legacy_root_offset_is_explicit():
    m, d, _ = build_classic_world(rope_kind="triple", connect_to="handles",
                                  rope_length=.5, legacy_rope_offset=True)
    mujoco.mj_forward(m, d)
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rope/ropewrap")
    first = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rope/seg_0")
    assert np.linalg.norm(d.xpos[first] - d.xpos[root]) == pytest.approx(.5/24, abs=4e-6)


def test_floor_stiffness_changes_only_explicit_rope_floor_pairs():
    kw = dict(rope_kind="triple",connect_to="handles",rope_length=.5,
              timestep=.001,rope_contacts="full",connect_timeconst=.004)
    base,_,_ = build_classic_world(**kw)
    m,_,_ = build_classic_world(**kw,rope_floor_timeconst=.002)
    assert m.npair == 24
    np.testing.assert_array_equal(m.geom_solref,base.geom_solref)
    floor = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"floor")
    for i in range(m.npair):
        assert floor in (m.pair_geom1[i],m.pair_geom2[i])
        other = m.pair_geom2[i] if m.pair_geom1[i] == floor else m.pair_geom1[i]
        assert mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,other).startswith("rope/rope_s")
        assert m.pair_solref[i,0] == pytest.approx(.002)
    with pytest.raises(ValueError):
        build_classic_world(**kw,rope_floor_timeconst=.0001)


def test_ball_joint_variant_preserves_visible_rope_and_contact_pairs():
    kw = dict(rope_kind="triple",connect_to="handles",rope_length=.5,
              timestep=.001,rope_contacts="full",connect_timeconst=.004,
              rope_floor_timeconst=.002)
    hinge,dh,_ = build_classic_world(**kw)
    ball,db,_ = build_classic_world(**kw,rope_joint_type="ball")
    assert hinge.nv == ball.nv
    assert ball.npair == 24
    mujoco.mj_forward(hinge,dh)
    mujoco.mj_forward(ball,db)
    for i in range(24):
        j = mujoco.mj_name2id(ball,mujoco.mjtObj.mjOBJ_JOINT,f"rope/rball{i}")
        assert ball.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL
        gh = mujoco.mj_name2id(hinge,mujoco.mjtObj.mjOBJ_GEOM,f"rope/rope_s{i}")
        gb = mujoco.mj_name2id(ball,mujoco.mjtObj.mjOBJ_GEOM,f"rope/rope_s{i}")
        np.testing.assert_allclose(dh.geom_xpos[gh],db.geom_xpos[gb],atol=1e-12)
        assert ball.geom_contype[gb] == 8


def test_initial_rope_phase_rotates_loop_without_moving_endpoints():
    kw = dict(rope_kind="triple",connect_to="handles",rope_length=.58,
              timestep=.0005,rope_contacts="full",rope_joint_type="ball")
    ends=[]
    centers=[]
    for phase in (0.,np.pi/2):
        m,d,_ = build_classic_world(**kw,rope_initial_phase=phase)
        mujoco.mj_forward(m,d)
        root = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"rope/ropewrap")
        last = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"rope/seg_23")
        end = d.xpos[last]+d.xmat[last].reshape(3,3)@np.array([.58/24,0,0])
        ends.append(np.stack([d.xpos[root],end]))
        mid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"rope/rope_s12")
        centers.append(d.geom_xpos[mid].copy())
    np.testing.assert_allclose(ends[0],ends[1],atol=1e-5)
    assert np.linalg.norm(centers[0]-centers[1]) > .1
