"""V2 contracts that must hold before any hero-run claim."""

import numpy as np
import mujoco
from microduck_lab.parkour.world import build_world
from microduck_lab.parkour.hazards import Hazard, HazardDriver
from microduck_lab.parkour.skills import Maneuver


def test_gap_is_a_void_not_a_painted_floor():
    m, d, r, track = build_world(gaps=((0.2, 0.4),))
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    def down(x):
        hit = np.array([-1], dtype=np.int32)
        return mujoco.mj_ray(
            m, d, np.array([x, 0, 1.0]), np.array([0.0, 0.0, -1.0]), group, 1, -1, hit
        )

    assert down(0.3) == -1
    assert abs(down(0.1) - 1) < 1e-8
    assert abs(down(0.5) - 1) < 1e-8
    assert 1.3 <= track.lane_width / track.duck_width <= 1.5
    assert np.isclose(track.width, 3 * track.lane_width)


def test_crate_is_free_after_warned_release():
    h = Hazard("crate", "crate", 1.0, z=0.6)
    m, d, r, _ = build_world(hazards=[h])
    driver = HazardDriver([h])
    assert m.jnt_type[m.joint("crate/free").id] == mujoco.mjtJoint.mjJNT_FREE
    driver.step(m, d, 0.5)
    assert d.eq_active[m.equality("crate/hold").id]
    for _ in range(50):
        driver.step(m, d, 0.5)
        r.step()
        mujoco.mj_step(m, d, 100)
    assert not d.eq_active[m.equality("crate/hold").id]
    assert d.xpos[m.body("crate").id, 2] < 0.6
    assert driver.events[1]["t"] - driver.events[0]["t"] >= 0.85


def test_roll_timeout_is_failure_not_success():
    m, d, r, _ = build_world()
    roll = Maneuver("ROLL_CENTER", started=-10.0)
    assert roll.update(m, d, r) == "FAILED"
    assert roll.rotation == 0
    assert not roll.inverted


def test_jump_requires_the_audited_jump_executor():
    import pytest

    m, d, r, _ = build_world()
    with pytest.raises(ValueError, match="requires CandidateJump"):
        Maneuver("JUMP_CENTER", 0).update(m, d, r)


def test_sweeper_has_finite_torque_not_position_windup():
    h = Hazard("sweep", "sweeper", 1.0, z=0.1, speed=100.0)
    m, d, r, _ = build_world(hazards=[h])
    driver = HazardDriver([h])
    driver.step(m, d, 0)
    aid = m.actuator("sweep/motor").id
    assert abs(d.ctrl[aid]) <= 0.025
    assert np.all(m.actuator_biasprm[aid] == 0)
    assert np.allclose(m.actuator_forcerange[aid], [-0.025, 0.025])


def test_prediction_rejects_obstacle_clear_now_but_crossing_later():
    from microduck_lab.parkour.prediction import HazardState, conflicts

    moving = HazardState(
        "wall", "moving_wall", (0.5, 0.5, 0.1), (0.0, -1.0, 0.0), (0.05, 0.05, 0.1)
    )
    # Robot and wall are initially separated; their paths intersect after .5 s.
    path = [
        (float(t), [t - 0.06, -0.06, 0.0], [t + 0.06, 0.06, 0.25])
        for t in np.linspace(0, 1, 26)
    ]
    assert conflicts(path, [moving]) == ["wall"]
    assert (
        conflicts(
            path,
            [
                HazardState(
                    "away",
                    "moving_wall",
                    (0.5, 0.5, 0.1),
                    (0.0, 1.0, 0.0),
                    (0.05, 0.05, 0.1),
                )
            ],
        )
        == []
    )


def test_prefetch_holds_answer_until_completion_and_revalidates():
    from concurrent.futures import Future
    from microduck_lab.parkour.tactics import RollingHorizon
    import time

    loop = RollingHorizon(None)
    loop.encounter_id = "crate1"
    loop.sent = time.monotonic()
    loop.snapshot = {}
    loop.candidates = ["TAKE_LEFT_ROUTE"]
    loop.future = Future()
    loop.future.set_result({"action": "TAKE_LEFT_ROUTE", "uncertain": 0.0})
    assert loop.poll("crate1", ["TAKE_LEFT_ROUTE"], completed=False) is None
    # The route became obstructed while the current skill was still executing.
    assert loop.poll("crate1", ["BRAKE_AND_WAIT"], completed=True) is None
    assert loop.records[-1]["status"] == "deferred_at_handoff"
    assert loop.poll("crate1", ["TAKE_LEFT_ROUTE"], completed=True) == "TAKE_LEFT_ROUTE"
    assert loop.records[-1]["status"] == "accepted"
    loop.close()


def test_hazard_proxies_preserve_inertias_and_native_floor_support():
    h = Hazard("sweep", "sweeper", 1.0, z=0.08)
    base, _, _, _ = build_world(hazards=[h], hazard_proxies=False)
    m, _, _, _ = build_world(hazards=[h])
    for bid in range(m.nbody):
        if m.body(bid).name.startswith("duck/"):
            old = base.body(m.body(bid).name).id
            assert np.allclose(m.body_mass[bid], base.body_mass[old])
            assert np.allclose(m.body_inertia[bid], base.body_inertia[old])
    for g in range(m.ngeom):
        if m.geom(g).name.endswith("/hazard_proxy"):
            assert m.geom_contype[g] == 8 and m.geom_conaffinity[g] == 4
            assert m.geom_group[g] == 4
    assert m.geom_conaffinity[m.geom("floor/0").id] & 8 == 0


def test_low_sweeper_never_starts_inside_rails():
    for height in [0.05, 0.08, 0.12]:
        h = Hazard("sweep", "sweeper", 1.0, z=height)
        m, d, _, _ = build_world(hazards=[h], rails=True)
        mujoco.mj_forward(m, d)
        for c in d.contact:
            names = [m.geom(g).name for g in (c.geom1, c.geom2)]
            assert not (
                any(n.startswith("rail/") for n in names)
                and "sweep/geom" in names
                and c.dist < 0
            )


def test_boulder_ttc_includes_its_physical_radius():
    from microduck_lab.parkour.prediction import observe_hazards

    h = Hazard("boss", "boulder", -1.0, z=0.25)
    m, d, r, tr = build_world(hazards=[h])
    driver = HazardDriver([h])
    d.qvel[m.jnt_dofadr[m.joint("boss/free").id]] = 0.5
    mujoco.mj_forward(m, d)
    _, public = observe_hazards(m, d, [h], driver, tr.lanes, 0.0, 0.0)
    assert np.isclose(public[0]["distance"], 0.75)
    assert np.isclose(public[0]["ttc"], 1.5)


def test_publication_rejects_ground_penetration_even_when_hazards_pass():
    from microduck_lab.parkour.evidence import contact_gate

    rows = {
        name: dict(samples=100, max_penetration_m=0.0, p99_penetration_m=0.0)
        for name in ("duck_floor", "duck_hazard", "duck_self")
    }
    assert contact_gate(rows, [0])["passed"]
    rows["duck_floor"]["max_penetration_m"] = 0.0177
    assert not contact_gate(rows, [0])["passed"]
    rows["duck_floor"]["max_penetration_m"] = 0.0
    rows["duck_self"]["p99_penetration_m"] = 0.0035
    assert not contact_gate(rows, [0])["passed"]
    assert not contact_gate({}, [0])["passed"]


def test_hard_contact_deployment_has_explicit_ground_and_self_limits():
    m, d, r, track = build_world(hard_contacts=True)
    native = (m.geom_contype & 1) != 0
    native &= np.array(
        [m.body(m.geom_bodyid[g]).name.startswith("duck/") for g in range(m.ngeom)]
    )
    assert np.allclose(m.geom_solref[native], [0.0012, 1.0])
    assert np.allclose(m.geom_margin[native], 0.0)
    for pair in range(m.npair):
        names = [m.geom(g).name for g in (m.pair_geom1[pair], m.pair_geom2[pair])]
        if any(n.startswith("duck/") for n in names) and any(
            n.startswith("floor/") for n in names
        ):
            assert np.allclose(m.pair_solref[pair], [0.002, 1.0])
            assert m.pair_margin[pair] == 0.00020


def test_live_envelopes_reject_native_soft_ground():
    import pytest
    from microduck_lab.parkour.level import build_level
    from microduck_lab.parkour.encounters import EncounterBrain

    m, d, r, track, level = build_level()
    driver = HazardDriver(level.hazards)
    with pytest.raises(ValueError, match="physical profile"):
        EncounterBrain(m, d, r, track, level, driver, mode="jev")


def test_deferred_tactic_expires_and_cannot_cross_encounters():
    import time
    from microduck_lab.parkour.tactics import RollingHorizon

    loop = RollingHorizon(None)
    result = {"action": "TAKE_LEFT_ROUTE"}
    loop.ready = (result, "crate", time.monotonic() - 0.5)
    assert loop.poll("crate", ["BRAKE_AND_WAIT"], completed=True) is None
    assert loop.ready is None
    assert loop.records[-1]["status"] == "invalid_at_handoff"
    loop.ready = (result, "crate", time.monotonic())
    assert loop.poll("gap", ["TAKE_LEFT_ROUTE"], completed=True) is None
    assert loop.ready is None
    loop.close()
