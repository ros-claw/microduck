import json
import numpy as np
import mujoco
from microduck_lab.parkour.world import build_world
from microduck_lab.parkour.preview import preview_jump, GapSequence
from microduck_lab.parkour.jump import GapAudit
from microduck_lab.game.world import REPO


def test_gap_prediction_preserves_live_state_and_matches_execution():
    m, d, r, _ = build_world(gaps=((0.06, 0.21),), hard_contacts=True)
    r.bank.paths["jump"] = str(REPO / "policies/parkour_long_jump_v2.onnx")
    for _ in range(100):
        r.active_policy = "stand"
        r.set_command()
        r.step()
        mujoco.mj_step(m, d, 100)
    before = {
        name: getattr(d, name).copy()
        for name in ["qpos", "qvel", "ctrl", "qacc_warmstart"]
    }
    action = r.last_action.copy()
    command = r.command.copy()
    t = float(d.time)
    result = preview_jump(m, d, r, 0.21)
    json.dumps(result)
    for name, expected in before.items():
        assert np.array_equal(getattr(d, name), expected), name
    assert np.array_equal(r.last_action, action)
    assert np.array_equal(r.command, command)
    assert d.time == t
    assert result["passed"]
    sequence = GapSequence(t, 0.21, result, r.trunk_pos())
    audit = GapAudit()
    audit.sample(m, d, r)
    for _ in range(150):
        sequence.update(m, d, r, audit)
        r.step()
        for _ in range(100):
            mujoco.mj_step(m, d)
            audit.sample(m, d, r)
        if sequence.status != "RUNNING":
            break
    assert sequence.status == "SUCCESS" and audit.crossed
    assert np.linalg.norm(r.trunk_pos() - result["final_position"]) < 0.02


def test_clean_escape_does_not_require_a_collision_but_keeps_contact_limits():
    from microduck_lab.parkour.evidence import contact_gate

    rows = {
        "duck_floor": dict(
            samples=100, max_penetration_m=0.001, p99_penetration_m=0.0001
        )
    }
    assert not contact_gate(rows, [0])["passed"]
    assert contact_gate(rows, [0], require_hazard_contact=False)["passed"]
    rows["duck_floor"]["max_penetration_m"] = 0.0016
    assert not contact_gate(rows, [0], require_hazard_contact=False)["passed"]
    assert not contact_gate({}, [0], require_hazard_contact=False)["passed"]


def test_untriggered_crate_forecast_reserves_its_drop_column():
    from microduck_lab.parkour.prediction import HazardState, conflicts

    crate = HazardState(
        "crate",
        "crate",
        (1.0, 0.12, 0.65),
        (0.0, 0.0, 0.0),
        (0.065, 0.065, 0.065),
        held=True,
    )
    low, _ = crate.bounds(0.0)
    assert low[2] > 0.5
    low, _ = crate.bounds(1.0)
    assert low[2] <= 0.0
    path = [
        (0.0, [0.2, 0.10, 0.02], [0.3, 0.2, 0.25]),
        (1.0, [0.95, 0.10, 0.02], [1.05, 0.2, 0.25]),
    ]
    assert conflicts(path, [crate]) == ["crate"]
