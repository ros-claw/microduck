"""Victory presentation must retain physical control and the entire ending."""

import numpy as np
from microduck_lab.parkour.level import build_level
from microduck_lab.parkour.victory import VictoryDance
from microduck_lab.parkour.cinema import CinematicEventTimeline, ShotDirector


def test_finish_controller_never_changes_root_or_external_forces():
    m, d, r, _, _ = build_level(16, difficulty="arcade", hard_contacts=True)
    victory = VictoryDance(0.0)
    qpos, qvel, forces = d.qpos.copy(), d.qvel.copy(), d.xfrc_applied.copy()
    victory.update(m, d, r)
    assert victory.finished_at is None and r.active_policy == "run"
    assert np.array_equal(qpos, d.qpos) and np.array_equal(qvel, d.qvel)
    assert np.array_equal(forces, d.xfrc_applied)
    # An explicitly scheduled celebration still writes controller targets only.
    victory.finished_at = 0.0
    for t in [0.2, 1.0, 1.9, 2.5, 4.1, 5.3, 6.5]:
        d.time = t
        victory.update(m, d, r)
        assert np.array_equal(qpos, d.qpos) and np.array_equal(qvel, d.qvel)
        assert np.array_equal(forces, d.xfrc_applied)
        if r.head_override is not None:
            assert np.max(np.abs(r.head_override)) < 0.6
    assert not victory.report(m, d, r)[
        "passed"
    ]  # no simulated airborne/landing evidence


def test_celebration_is_not_cut_short_by_boss_impact():
    events = [
        dict(type="SKILL_START", skill="ROLL_CENTER", t=0.1),
        dict(type="SKILL_RESULT", skill="ROLL_CENTER", status="SUCCESS", t=0.5),
        dict(type="SKILL_START", skill="JUMP_CENTER", t=1.5),
        dict(type="SKILL_RESULT", skill="JUMP_CENTER", status="SUCCESS", t=2.2),
        dict(type="BALL_PUSH", t=2.3),
        dict(type="BOWLING_STRIKE", t=2.8),
        dict(type="SKILL_START", skill="ROLL_CENTER", t=3.2),
        dict(type="SKILL_RESULT", skill="ROLL_CENTER", status="SUCCESS", t=4.0),
        dict(type="FINISH", t=5.5),
        dict(type="BOSS_GATE_IMPACT", t=6.0),
    ]
    report = dict(
        events=events,
        difficulty="arcade",
        brain="jev",
        gap_flights=[dict(crossed=True, start=1.8, end=2.0)],
        victory=dict(passed=True),
    )
    tl = CinematicEventTimeline(report, 12.5)
    assert tl.stop == 12.5
    clips = tl.technical()
    assert clips[0].stop == 12.5
    assert all(c.duration > 0 for c in clips)
    assert len([c for c in clips if c.speed < 1 and not c.hold]) == 3
    assert sum(c.duration for c in clips) < 120
    assert any(c.shot == "victory" and c.stop >= 11.8 for c in clips)
    assert ShotDirector(tl).choose(9.0) == "victory"
