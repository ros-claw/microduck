"""Execution contracts for public-state tactical composition."""

from dataclasses import replace
import math
import numpy as np

from microduck_lab.combat.tactics_v3 import TacticalPilot, TacticalTeamPlanner
from microduck_lab.combat.team_intents import Assignment
from test_v2 import body, config, context


def setup():
    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    public = dict(
        a=body("a", xy=(-0.30, -0.20)),
        b=body("b", xy=(0.30, -0.20), yaw=math.pi),
        c=body("c", xy=(-0.30, 0.20)),
        d=body("d", xy=(0.30, 0.20), yaw=math.pi),
    )
    planner = TacticalTeamPlanner("RED", teams)
    pilot = TacticalPilot(
        "a", "RED", teams, replace(config(), controller_revision=3), planner
    )
    pilot.motor_context = context()
    return teams, public, planner, pilot


def test_tactic_is_selected_by_style_not_color_or_actor_identity():
    teams, public, red, _ = setup()
    red.style = "pressure"
    red.update(1.0, public, list(teams))
    assert {a.role for a in red.assignments.values()} == {"PRESS"}
    blue = TacticalTeamPlanner("BLUE", teams)
    blue.style = "counter"
    blue.update(1.0, public, list(teams))
    assert {a.role for a in blue.assignments.values()} == {"DRAW", "FLANK"}


def test_cover_changes_only_at_a_real_action_boundary():
    teams, public, planner, pilot = setup()
    pilot.target, pilot.role = "b", "PRESS"
    pilot.start(1.0, "drive", 1.0, public["a"], public["b"], phase="ACTIVE")
    planner.assignments["a"] = Assignment("d", "COVER", "COVER_WITHDRAWAL", 1.1, "c")
    pilot.choose(1.12, public, list(teams))
    assert pilot.target == "b" and pilot.role == "PRESS"
    pilot.finish(1.14)
    pilot.choose(1.16, public, list(teams))
    assert pilot.target == "d" and pilot.role == "COVER"


def test_public_head_pose_can_trigger_evasion_without_private_attack_flags():
    teams, public, planner, pilot = setup()
    public["b"] = replace(
        public["b"], xy=(0.03, -0.20), neck_pitch_delta=-0.3, head_pitch_delta=-0.15
    )
    planner.update(1.0, public, list(teams))
    command = pilot.choose(1.0, public, list(teams))
    assert pilot.skill == "sidestep"
    assert command.policy == "evade_candidate"
    assert command.twist[0] == command.twist[2] == 0.0


def test_turn_into_counter_requires_actual_loaded_motor_state():
    teams, public, planner, pilot = setup()
    pilot.target = "b"
    public["b"] = replace(public["b"], xy=(0.02, -0.2))
    pilot.start(1.0, "turn_counter", 0.8, public["a"], public["b"])
    pilot.motor_context = context(foot_normal_N=0.0)
    pilot.choose(1.1, public, list(teams))
    assert pilot.skill == "turn_counter" and pilot.active_attack is None
    pilot.motor_context = context()
    pilot.choose(1.12, public, list(teams))
    assert pilot.skill == "counter_drive" and pilot.phase == "WINDUP"


def test_facing_steering_is_continuous_and_stays_within_motor_command_bounds():
    _, public, _, pilot = setup()
    a = public["a"]
    for angle in np.linspace(-math.pi, math.pi, 81):
        point = np.array(a.xy) + 0.6 * np.array([math.cos(angle), math.sin(angle)])
        command = pilot.navigate(a, point)
        assert np.isfinite(command.twist).all()
        assert abs(command.twist[0]) <= 0.30 and abs(command.twist[1]) <= 0.30
        assert abs(command.twist[2]) <= 1.0


def test_elimination_invalidates_committed_team_target_and_cover():
    teams, public, planner, _ = setup()
    planner.update(1.0, public, list(teams))
    planner.update(1.02, public, ["a", "c", "d"])
    assert all(a.target == "d" for a in planner.assignments.values())
