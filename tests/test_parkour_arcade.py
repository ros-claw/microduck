import mujoco
import numpy as np
from microduck_lab.parkour.hazards import Hazard
from microduck_lab.parkour.arcade import BowlingLock, ArcadeDriver
from microduck_lab.parkour.world import build_world
from microduck_lab.parkour.skills import lane_command
from microduck_lab.parkour.audit import Audit


def bowling_trial(off_axis=False):
    hazards = [Hazard("playball", "bowling_ball", 0.55, z=0.085)] + [
        Hazard(
            "pin" + str(i),
            "pin",
            1.25 + i * 0.055,
            y=0.25 if off_axis and i == 2 else 0,
            z=0.065,
        )
        for i in range(3)
    ]
    hazards += [
        Hazard("gutter_left", "guide", 0.87, y=0.14, z=0.075, half_width=0.52),
        Hazard("gutter_right", "guide", 0.87, y=-0.14, z=0.075, half_width=0.52),
    ]
    m, d, r, _ = build_world(hazards=hazards, end=2, hard_contacts=True)
    lock = BowlingLock()
    max_depth = 0
    for k in range(300):
        r.active_policy = "stand" if k < 50 else "run"
        r.set_command()
        if k >= 50 and lock.duck_push_impulse <= 0.025:
            r.command[:3] = lane_command(r, 0, 0.55)
        r.step()
        for _ in range(100):
            mujoco.mj_step(m, d)
            lock.sample(m, d)
            for i, c in enumerate(d.contact):
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, force)
                if force[0] > 1e-7:
                    max_depth = max(max_depth, -float(c.dist))
        if k == 49:
            assert not lock.unlocked and not lock.struck
        if lock.unlocked:
            break
    assert d.xmat[r.trunk_body_id, 8] > 0.9 and max_depth < 0.0015
    assert not np.any(d.xfrc_applied)
    assert all("playball" not in m.actuator(i).name for i in range(m.nu))
    return lock, m, d


def test_real_push_and_pin_relay_unlock_only_after_all_three_topple():
    lock, _, _ = bowling_trial()
    assert lock.unlocked and len(lock.toppled) == 3 and not lock.contaminated
    assert lock.parents == {"pin0": "playball", "pin1": "pin0", "pin2": "pin1"}
    assert lock.duck_push_impulse > 0.001


def test_a_target_outside_the_contact_chain_keeps_exit_locked():
    lock, _, d = bowling_trial(off_axis=True)
    assert not lock.unlocked and "pin2" not in lock.struck
    assert d.body("pin2").xmat[8] > 0.9
    assert lock.toppled and not lock.contaminated


def test_unpushed_ball_does_not_unlock_and_gate_uses_bounded_actuator():
    hazards = [
        Hazard("playball", "bowling_ball", 0.55, z=0.085),
        Hazard("portal", "finish_gate", 1.6, z=0.905),
    ] + [
        Hazard(
            "pin" + str(i),
            "pin",
            0.95 + (i > 0) * 0.065,
            y=[0, -0.035, 0.035][i],
            z=0.065,
        )
        for i in range(3)
    ]
    m, d, r, _ = build_world(hazards=hazards, end=2, hard_contacts=True)
    driver = ArcadeDriver(hazards)
    for _ in range(60):
        r.active_policy = "stand"
        r.set_command()
        driver.step(m, d, 0)
        r.step()
        for _ in range(100):
            mujoco.mj_step(m, d)
            driver.sample(m, d)
    assert not driver.bowling.unlocked and not driver.bowling.struck
    motor = m.actuator("portal/motor").id
    assert d.ctrl[motor] == -0.6 and m.actuator_forcelimited[motor]
    assert np.array_equal(m.actuator_forcerange[motor], [-6, 6])
    assert not np.any(d.xfrc_applied)


def test_intentional_prop_touch_keeps_contact_and_depth_evidence_without_damage():
    audit = Audit(interactive_props=("playball",))
    audit.contacts["playball"] = dict(samples=3, max_penetration_m=0.002)
    audit.depths = [0.002]
    audit._pending["playball"] = dict(first=0.1, last=0.2, impulse=0.1, geom=0)
    report = audit.report()
    assert report["hp"] == 3 and report["max_penetration_m"] == 0.002
    assert report["contacts"]["playball"]["samples"] == 3
    assert report["events"][0]["type"] == "PROP_TOUCH"
    audit = Audit()
    audit._pending["boss"] = dict(first=0.1, last=0.2, impulse=0.1, geom=0)
    assert audit.report()["hp"] == 2


def test_repeated_rolls_keep_separate_cinematic_windows():
    from microduck_lab.parkour.cinema import CinematicEventTimeline

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
    )
    timeline = CinematicEventTimeline(report, 6.5)
    assert timeline.skills["ROLL_CENTER"] == [0.1, 0.5]
    assert timeline.instances["ROLL_CENTER"] == [[0.1, 0.5], [3.2, 4.0]]
    for clips in [timeline.hero(), timeline.technical()]:
        assert len([c for c in clips if c.speed < 1 and not c.hold]) == 3
        assert sum(c.duration for c in clips) < 120


def test_arcade_first_roll_cannot_bypass_the_combination_planner(tmp_path):
    from microduck_lab.demos.parkour_rehearsal import run
    from microduck_lab.game.world import REPO

    report = run(
        str(REPO / "policies/parkour_long_jump_v2.onnx"),
        seed=16,
        forecast_entry_roll=True,
        seconds=6,
        output=tmp_path / "entry",
        hard_contacts=True,
        predictive=True,
        difficulty="arcade",
        sweeper_mass=0.4,
        sweeper_torque=0.2,
        sweeper_phase=0.9,
    )
    assert report["combination_planning"]
    entries = [f for f in report["forecasts"] if f["kind"] == "entry_roll"]
    assert len(entries) == 1
    assert all(c.get("self_contact") is not None for c in entries[0]["candidates"])
    if not entries[0]["accepted"]:
        assert report["stage"] == "ROLL_ABORT"
        assert not any(s["stage"] == "ROLL" for s in report["skills"])
