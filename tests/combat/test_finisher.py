from dataclasses import replace
import math
from pathlib import Path
import pytest
from microduck_lab.combat.tactics_v4 import FinisherPilot
from microduck_lab.combat.world import CombatBank
from test_v2 import body, config, context


def pilot():
    p = FinisherPilot(
        "a", "RED", {"a": "RED", "b": "BLUE"}, replace(config(), controller_revision=4)
    )
    p.motor_context = context()
    p.target = "b"
    p.role = "PRESS"
    return p


def test_supported_touching_actor_can_continue_a_real_push():
    p = pilot()
    a = body("a", xy=(0.35, 0.0), touching_opponent=True, boundary_distance=0.4)
    b = body("b", xy=(0.52, 0.0), boundary_distance=0.26, yaw=math.pi)
    assert (
        p.special_choice(2.0, a, b, {"a": a, "b": b}, ["a", "b"])[0] == "finish_drive"
    )
    p.choose(2.0, {"a": a, "b": b}, ["a", "b"])
    assert p.skill == "finish_drive" and p.phase == "WINDUP"


def test_finish_requires_loaded_upright_own_body_and_safe_margin():
    p = pilot()
    a = body("a", xy=(0.35, 0.0))
    b = body("b", xy=(0.52, 0.0), boundary_distance=0.26)
    for change in [
        dict(foot_support=False),
        dict(upright=0.9),
        dict(body_ground=True),
        dict(boundary_distance=0.08),
    ]:
        assert p.special_choice(2.0, replace(a, **change), b, {}, ["a", "b"]) is None


def test_finishing_commitment_is_bounded_and_does_not_change_short_bumps():
    p = pilot()
    a, b = body(), body("b", xy=(0.25, 0.0))
    p.skill = "finish_drive"
    assert p.attack_limits(a, b) == (2.2, 2.0, 0.38)
    p.skill = "short_bump"
    assert p.attack_limits(a, b) == (0.42, 0.16, 0.24)


def test_live_tracking_turn_is_bounded_and_does_not_lock_the_old_aim():
    p = pilot()
    p.skill = "finish_drive"
    p.aim = -2.0
    assert p.drive_yaw(body(), body("b", xy=(0.2, 0.1))) == 0.5
    assert p.drive_yaw(body(), body("b", xy=(0.2, -0.1))) == -0.5


def test_late_evade_handoff_does_not_expire_a_new_turn_immediately():
    p = pilot()
    a = body("a")
    b = body("b", xy=(0.25, 0.20), yaw=math.pi)
    p.start(0.8, "sidestep", 0.54, a, b)
    p.choose(2.0, {"a": a, "b": b}, ["a", "b"])
    assert p.skill == "turn_counter" and p.action is not None
    assert p.since == 2.0


def test_outward_momentum_requests_zero_motor_before_reaching_the_edge():
    p = pilot()
    a = body("a", xy=(0.62, 0.0), velocity=(0.40, 0.0), boundary_distance=0.21)
    b = body("b", xy=(0.78, 0.0))
    p.start(1.0, "finish_drive", 2.4, a, b)
    command = p.choose(1.2, {"a": a, "b": b}, ["a", "b"])
    assert p.skill == "edge_brake" and p.active_attack is None
    assert command.policy == "dash_candidate" and command.twist == (0.0, 0.0, 0.0)
    assert a.velocity == (0.40, 0.0)


def test_inward_momentum_is_not_stopped_and_edge_brake_waits_for_support():
    p = pilot()
    b = body("b", xy=(0.78, 0.0))
    a = body("a", xy=(0.62, 0.0), velocity=(-0.40, 0.0), boundary_distance=0.21)
    assert p.control_override(1.2, {"a": a, "b": b}, ["a", "b"]) is None
    a = replace(a, velocity=(0.40, 0.0))
    p.control_override(1.2, {"a": a, "b": b}, ["a", "b"])
    slow = replace(a, velocity=(0.0, 0.0), foot_support=False)
    c = p.control_override(1.7, {"a": slow, "b": b}, ["a", "b"])
    assert p.skill == "edge_brake" and c.twist == (0.0, 0.0, 0.0)


def test_finishing_handoff_keeps_the_same_motor_for_its_zero_command_brake():
    p = pilot()
    p.c = replace(
        p.c, finish_motor_path="native_finish.onnx", finish_motor_sha256="0" * 64
    )
    a = body("a", xy=(0.62, 0.0), velocity=(0.40, 0.0), boundary_distance=0.21)
    b = body("b", xy=(0.78, 0.0))
    p.start(1.0, "finish_drive", 2.4, a, b)
    cmd = p.choose(1.2, {"a": a, "b": b}, ["a", "b"])
    assert cmd.policy == "finish_candidate" and cmd.twist == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "changes",
    [
        dict(finish_motor_path="unsealed.onnx"),
        dict(finish_motor_sha256="0" * 64),
        dict(
            finish_motor_path="sealed.onnx",
            finish_motor_sha256="0" * 64,
            controller_revision=3,
        ),
    ],
)
def test_unpinned_or_incompatible_finishing_weights_are_rejected(changes):
    with pytest.raises(ValueError, match="[Ff]inishing motor"):
        replace(pilot().c, **changes)


def test_ordinary_contact_export_cannot_masquerade_as_native_finisher():
    path = Path(__file__).parents[2] / "policies/candidates/duckfu_contact_v1_100.onnx"
    with pytest.raises(ValueError, match="trained skill scope"):
        CombatBank({"finish_candidate": str(path)}).get("finish_candidate")
