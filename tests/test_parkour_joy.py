"""Continuous learned finish must preserve physics and emphasize contact details."""

import numpy as np
from microduck_lab.parkour.joy import JoyDance, joy_pose_commands
from microduck_lab.parkour.level import build_level
from microduck_lab.parkour.cinema import CinematicEventTimeline


def test_continuous_controller_only_sets_commands_and_uses_one_policy():
    m, d, r, _, _ = build_level(16, difficulty="arcade", hard_contacts=True)
    joy = JoyDance(0.0, finished_at=0.0)
    q, v, f = d.qpos.copy(), d.qvel.copy(), d.xfrc_applied.copy()
    commands = []
    for t in np.arange(1.6, 6.0, 0.02):
        d.time = float(t)
        joy.update(m, d, r)
        assert r.active_policy == "joy"
        assert r.head_override is None and r.leg_override is None
        assert np.array_equal(q, d.qpos) and np.array_equal(v, d.qvel)
        assert np.array_equal(f, d.xfrc_applied)
        commands.append(r.command.copy())
    assert np.max(np.abs(np.diff(commands, axis=0))) < 0.04
    assert not joy.report(m, d, r)["passed"]


def test_details_cut_uses_verified_impact_and_three_meaningful_slow_sections():
    events = [
        dict(type="SKILL_START", skill="ROLL_CENTER", t=0.1),
        dict(type="SKILL_RESULT", skill="ROLL_CENTER", status="SUCCESS", t=0.5),
        dict(type="SKILL_START", skill="JUMP_CENTER", t=1.5),
        dict(type="SKILL_RESULT", skill="JUMP_CENTER", status="SUCCESS", t=2.2),
        dict(type="BALL_PUSH", t=5.0),
        dict(type="BOWLING_STRIKE", t=5.8),
        dict(type="SKILL_START", skill="ROLL_CENTER", t=6.0),
        dict(type="SKILL_RESULT", skill="ROLL_CENTER", status="SUCCESS", t=6.8),
        dict(type="FINISH", t=7.5),
        dict(type="BOSS_GATE_IMPACT", t=8.0),
    ]
    r = dict(
        events=events,
        difficulty="arcade",
        brain="jev",
        detail_focus=True,
        sweeper_detail=dict(peak_force_time=3.7),
        gap_flights=[dict(crossed=True, start=1.8, end=2.0)],
        victory=dict(passed=True),
    )
    clips = CinematicEventTimeline(r, 13.5).technical()
    assert clips[0].stop == 13.5
    slow = [c for c in clips if c.speed < 1 and not c.hold]
    assert [c.shot for c in slow] == ["landing", "sweeper_top", "bowling"]
    contact = next(c for c in clips if c.title.startswith("CONTACT FRAME"))
    assert contact.start == 3.7 and contact.hold > 0
    assert sum(c.duration for c in clips) < 120
    assert not any("VICTORY" in c.title for c in clips)


def test_target_conditioner_matches_training_equation_and_preserves_raw_actions():
    from types import SimpleNamespace

    rng = np.random.default_rng(25)
    initial = rng.normal(0, 0.1, 14)
    d = SimpleNamespace(ctrl=initial.copy())
    r = SimpleNamespace(
        act_ids=np.arange(14),
        last_action=np.zeros(14),
        bank=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                get_modelmeta=lambda: SimpleNamespace(
                    custom_metadata_map={
                        "joy_target_conditioner": "v1",
                        "joy_target_alpha": ".25",
                        "joy_target_max_step_rad": ".08",
                    }
                )
            )
        ),
    )
    joy = JoyDance(0.0, anchor=np.zeros(3), last_targets=initial.copy())
    previous = initial.copy()
    for raw in rng.normal(0, 1, (50, 14)):
        r.last_action = raw.copy()
        d.ctrl[:] = raw
        expected = previous + np.clip(0.25 * (raw - previous), -0.08, 0.08)
        joy.condition_targets(d, r)
        np.testing.assert_allclose(d.ctrl, expected, atol=1e-12)
        np.testing.assert_array_equal(r.last_action, raw)
        assert max(abs(d.ctrl - previous)) <= 0.08000001
        previous = expected
        joy.last_targets = d.ctrl.copy()
