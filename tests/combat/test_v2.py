from dataclasses import replace
import math
from pathlib import Path

import pytest

from microduck_lab.combat.config import strict_load
from microduck_lab.combat.v2_config import V2Config
from microduck_lab.combat.observations import Body
from microduck_lab.combat.tactics_v2 import V2Pilot
from microduck_lab.combat.team_intents import TeamPlanner
from microduck_lab.combat.transition_contracts import MotorContext, evaluate_entry
from microduck_lab.combat.parameterized_actions import ManeuverRequest


def config():
    return strict_load(
        Path(__file__).parents[2] / "configs/combat/v2/baseline.yaml", V2Config
    )


def body(name="a", **changes):
    return replace(
        Body(name, (0.0, 0.0), 0.12, (0.0, 0.0), 0.0, 1.0, True, False, 0.6), **changes
    )


def context(**changes):
    return replace(MotorContext(0.2, 4.0, 1.0, 0.3, 7.0, True), **changes)


def test_old_tactical_weights_are_rejected_instead_of_padded():
    with pytest.raises(ValueError, match="incompatible"):
        replace(config(), tactics_path="old.npz", tactics_sha256="0" * 64)


@pytest.mark.parametrize(
    "change",
    [
        dict(yaw_rate=3.0),
        dict(joint_speed_max=30.0),
        dict(foot_normal_N=0.0),
        dict(finite=False),
        dict(last_action_max=float("nan")),
    ],
)
def test_dynamic_entry_rejects_unsafe_actual_motor_states(change):
    assert not evaluate_entry(body(), context(**change))["accepted"]


def test_transition_requires_motor_history_and_actual_support():
    assert not evaluate_entry(body(), None)["accepted"]
    assert not evaluate_entry(body(foot_support=False), context())["accepted"]
    assert evaluate_entry(body(velocity=(0.1, 0.2)), context())["accepted"]


def prepared_evade(dynamic=True, unsafe=False):
    cfg = replace(config(), dynamic_handoff=dynamic)
    p = V2Pilot("a", "RED", {"a": "RED", "b": "BLUE"}, cfg)
    p.target, p.action, p.phase, p.skill = "b", 4, "ACTIVE", "cut_counter"
    p.since, p.phase_since = 1.0, 1.0
    p.motor_context = context(joint_speed_max=30.0 if unsafe else 4.0)
    p.observed_drive, p.evade_min_distance = True, 0.20
    public = {
        "a": body(xy=(0.0, 0.08), velocity=(0.0, 0.2)),
        "b": body("b", xy=(0.25, 0.0), yaw=math.pi, velocity=(0.3, 0.0)),
    }
    return p, public


def test_valid_dynamic_evade_hands_off_without_full_stop():
    p, public = prepared_evade()
    p.choose(1.62, public, list(public))
    assert any(e["state"] == "DYNAMIC_HANDOFF" for e in p.events)
    assert any(e["state"] == "COUNTER_ATTEMPT" for e in p.events)
    assert p.action == 9 and p.phase == "WINDUP"


def test_same_evade_with_full_brake_or_unsafe_state_cannot_force_combo():
    for dynamic, unsafe in [(False, False), (True, True)]:
        p, public = prepared_evade(dynamic, unsafe)
        command = p.choose(1.62, public, list(public))
        assert not any(e["state"] == "DYNAMIC_HANDOFF" for e in p.events)
        assert p.phase in ("ACTIVE", "RECOVERY") and command.policy == "evade_candidate"
        if unsafe:
            assert p.phase == "RECOVERY"


def test_dynamic_pivot_translates_toward_public_target_while_turning():
    p, public = prepared_evade()
    public["b"] = replace(public["b"], xy=(0.40, 0.0))
    command = p.choose(1.62, public, list(public))
    assert p.phase == "PIVOT" and command.policy == "walk"
    assert command.twist[0] > 0 and command.twist[2] < 0


def team_public():
    return dict(
        a=body("a", xy=(-0.4, -0.15)),
        c=body("c", xy=(-0.4, 0.15)),
        b=body("b", xy=(0.0, -0.15)),
        d=body("d", xy=(0.6, 0.3)),
    )


def test_team_can_focus_one_target_with_two_opponents_alive():
    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    p = TeamPlanner("RED", teams)
    public = team_public()
    p.update(1.0, public, list(public))
    assert p.assignment("a").target == p.assignment("c").target == "b"
    assert {a.role for a in p.assignments.values()} == {"PRESS", "FLANK"}


def test_endangered_teammate_changes_real_target_and_commitment_prevents_chatter():
    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    p = TeamPlanner("RED", teams)
    public = team_public()
    p.update(1.0, public, list(public))
    saved = dict(p.assignments)
    public["b"] = body("b", xy=(0.5, -0.2))
    p.update(1.2, public, list(public))
    assert p.assignments == saved
    public["c"] = body("c", xy=(0.55, 0.3), boundary_distance=0.1)
    p.update(1.25, public, list(public))
    assert p.assignment("a").role == "PEEL" and p.assignment("a").target == "d"
    assert p.assignment("c").role == "ESCAPE"
    assert p.assignment("a").intent == "PEEL_COVER"


def test_parameterized_action_does_not_accept_arbitrary_force_or_unknown_family():
    with pytest.raises(ValueError):
        ManeuverRequest("a", "b", "teleport")
    with pytest.raises(TypeError):
        ManeuverRequest("a", "b", "probe_bump", force=50)


def test_same_public_walking_is_not_mistaken_for_an_incoming_drive():
    p = V2Pilot("a", "RED", {"a": "RED", "b": "BLUE"}, config())
    public = {
        "a": body(velocity=(0.2, 0.0)),
        "b": body("b", xy=(0.4, 0.0), yaw=math.pi, velocity=(-0.18, 0.0)),
    }
    p.choose(1.0, public, list(public))
    assert p.action == 0  # Relative closing is .38, but enemy alone is only walking.


def test_flanker_reaching_the_angle_can_stop_circling_and_attack():
    from microduck_lab.combat.team_intents import Assignment

    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    planner = TeamPlanner("RED", teams)
    planner.assignments = {"a": Assignment("b", "FLANK", "PRESS_FLANK", 1.0)}
    p = V2Pilot("a", "RED", teams, config(), planner)
    public = dict(
        a=body("a", xy=(0.24, 0.0), yaw=math.pi),
        b=body("b"),
        c=body("c", xy=(0.0, 0.3)),
        d=body("d", xy=(0.6, -0.3)),
    )
    command = p.choose(1.0, public, list(public))
    assert p.action == 9 and p.phase == "WINDUP" and command.head[0] < 0
