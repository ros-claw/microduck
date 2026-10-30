from dataclasses import replace
import numpy as np
import mujoco
import pytest
from microduck_lab.combat.config import Config, Physics, strict_load
from microduck_lab.combat.world import build
from microduck_lab.combat.observations import Body, snapshot
from microduck_lab.combat.maneuver import Maneuver
from microduck_lab.combat.contacts import ContactAudit
from microduck_lab.combat.referee import Referee
from microduck_lab.combat.episode import run


def body(name="a", **kwargs):
    b = Body(name, (0.0, 0.0), 0.12, (0.0, 0.0), 0.0, 1.0, True, False, 1.0)
    return replace(b, **kwargs)


def test_strict_configuration_rejects_unknown_and_type_coercion(tmp_path):
    p = tmp_path / "bad.yaml"
    for contents in [
        "unknown: true",
        'swap: "false"',
        "duration_s: .nan",
        "duration_s: true",
        "evade_sign: 1.0",
    ]:
        p.write_text(contents)
        with pytest.raises(ValueError):
            strict_load(p, Config)
    for changes in [
        dict(schema="other"),
        dict(mode="REAL"),
        dict(duration_s=float("nan")),
        dict(defense="learned"),
    ]:
        with pytest.raises(ValueError):
            Config(**changes)


def test_clock_uses_actual_dt_not_archived_substeps():
    assert Physics().substeps == 80
    assert Physics(dt=0.000125).substeps == 160
    with pytest.raises(ValueError):
        Physics(dt=0.005)
    with pytest.raises(ValueError):
        Physics(torque_limit_Nm=0.96)


def test_two_ducks_have_independent_history_and_native_inertia():
    m, d, ducks, _ = build(Config(), Physics(), 101)
    a, b = ducks.values()
    assert len(set(a.act_ids) & set(b.act_ids)) == 0
    assert len(set(a.joint_qpos_idx) & set(b.joint_qpos_idx)) == 0
    a.last_action[0] = 1
    assert b.last_action[0] == 0
    assert np.max(m.actuator_forcerange) == pytest.approx(0.6405)
    masses = [
        sum(
            m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith(n + "/")
        )
        for n in ducks
    ]
    assert masses == pytest.approx([0.73724318, 0.73724318])
    assert m.neq == 0
    assert not any("external_envelope" in m.geom(i).name for i in range(m.ngeom))
    assert m.geom("stage").id != m.geom("receiver").id
    assert np.all(d.xfrc_applied == 0) and np.all(d.qfrc_applied == 0)
    assert a.get_obs().shape == (61,)
    public = snapshot(ducks, ContactAudit(m, ducks), Config())
    assert all(
        not hasattr(v, "phase") and not hasattr(v, "intent") for v in public.values()
    )
    with pytest.raises(Exception):
        public["lavender"].yaw = 3


def test_attack_commits_to_launch_heading_and_recovery_preempts():
    c = Config()
    actor = Maneuver("a", "coach_attacker", c)
    me = body()
    actor.choose(1.0, me, body("b", xy=(0.3, 0)))
    assert actor.phase == "WINDUP"
    actor.choose(1.14, me, body("b", xy=(0.3, 0)))
    assert actor.phase == "ACTIVE"
    command = actor.choose(1.16, me, body("b", xy=(0.3, 0.4)))
    assert command.owner == "maneuver" and command.twist[2] == 0
    assert actor.active_attack == "a:1"
    command = actor.choose(1.18, replace(me, upright=0.5), body("b", xy=(0.3, 0.4)))
    assert command.owner == "recovery" and actor.active_attack is None


def test_dodge_uses_visible_closing_motion_not_opponent_intent():
    me = body()
    opponent = body("b", xy=(0.3, 0), velocity=(-0.25, 0))
    actor = Maneuver("a", "counter", Config())
    cmd = actor.choose(1.0, me, opponent)
    assert actor.skill == "evade_step" and abs(cmd.twist[1]) == 0.3
    actor = Maneuver("a", "counter", Config())
    actor.choose(1.0, me, replace(opponent, velocity=(0.25, 0)))
    assert actor.skill != "evade_step"


@pytest.mark.parametrize("names", [("a", "b"), ("b", "a")])
def test_simultaneous_ringout_is_draw_and_receiver_does_not_save(names):
    r = Referee(names, 10)
    states = {n: body(n, z=-0.3, foot_support=False) for n in names}
    r.update(0.0, states)
    r.update(0.14, states)
    assert r.result["kind"] == "DRAW"
    assert set(r.eliminated) == set(names)


def test_airborne_step_and_roll_label_cannot_bypass_down_count():
    r = Referee(("a", "b"), 10)
    states = {"a": body("a", z=0.25, foot_support=False), "b": body("b")}
    r.update(0.0, states)
    r.update(0.5, states)
    assert not r.eliminated
    states["a"] = body("a", upright=-0.5, body_ground=True, foot_support=False)
    r.update(1.0, states)
    r.update(11.02, states)
    assert r.eliminated == {"a": "DOWN_COUNT"}
    assert r.result["winner"] == "b"


def test_physics_fault_never_awards_winner():
    r = Referee(("a", "b"), 10)
    r.update(1.0, {"a": body("a"), "b": body("b")}, True)
    assert r.result["kind"] == "NO_CONTEST_PHYSICS" and r.result["winner"] is None


def test_no_contact_no_strike_and_no_initialization_overwrite(tmp_path):
    out = tmp_path / "run"
    result = run(out, Config(duration_s=2.0, defense="absent"), Physics(), 101)
    assert result["physical"]["external_forces_zero"]
    assert result["physical"]["model_unchanged"]
    assert result["physical"]["clock_ok"]
    assert result["physical"]["finite"]
    assert "VALID_STRIKE" not in (out / "runtime_events.jsonl").read_text()
    assert "CONTACTED_ATTACK" not in (out / "runtime_events.jsonl").read_text()
    with pytest.raises(FileExistsError):
        run(out, Config(), Physics(), 101)


def test_contact_normal_vector_sign_matches_two_free_body_response():
    s = mujoco.MjSpec()
    s.option.gravity = [0, 0, 0]
    s.option.timestep = 0.00025
    s.worldbody.add_geom(
        name="stage",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1, 1, 0.1],
        pos=[0, 0, -10],
    )
    s.worldbody.add_geom(
        name="receiver",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1, 1, 0.1],
        pos=[0, 0, -20],
    )
    for name, x in [("lavender", -0.095), ("cream", 0.095)]:
        b = s.worldbody.add_body(name=name + "/trunk_base", pos=[x, 0, 0])
        b.add_freejoint()
        b.add_geom(
            name=name + "/left_foot_collision",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.1, 0, 0],
            mass=1.0,
        )
        b.add_geom(
            name=name + "/right_foot_collision",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.001, 0, 0],
            pos=[0, 0, 1],
            mass=0,
            contype=0,
            conaffinity=0,
        )
    m = s.compile()
    d = mujoco.MjData(m)
    ducks = dict.fromkeys(["lavender", "cream"])
    audit = ContactAudit(m, ducks)
    for k in range(20):
        mujoco.mj_step(m, d)
        audit.read(m, d, k, 0.00025, {})
    assert d.qvel[0] < 0 and d.qvel[6] > 0
    assert len(audit.episodes) == 1
    # Canonical second body is lavender, whose world impulse is negative X.
    assert audit.episodes[0]["normal_impulse_on_second_Ns"][0] < 0
    assert audit.episodes[0]["attacks"] == []
    assert not audit.support["lavender"]


def test_launch_distance_is_a_bounded_physical_calibration_parameter():
    me = body()
    opponent = body("b", xy=(0.35, 0))
    near = Maneuver("a", "coach_attacker", Config(dash_start_distance_m=0.33))
    far = Maneuver("a", "coach_attacker", Config(dash_start_distance_m=0.4))
    near.choose(1.0, me, opponent)
    far.choose(1.0, me, opponent)
    assert near.phase == "READY" and far.phase == "WINDUP"
    with pytest.raises(ValueError):
        Config(dash_start_distance_m=0.1)


def test_unavailable_learned_brain_never_creates_a_fake_rule_run(tmp_path):
    import subprocess
    import sys

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    out = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_combat.py"),
            "--config",
            "unused.yaml",
            "--brain",
            "learned",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "no fallback" in result.stderr
    assert not out.exists()
