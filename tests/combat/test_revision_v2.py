"""Regressions for execution/plan consistency and geometry-based support."""

from dataclasses import replace
import math

import numpy as np
import pytest

from microduck_lab.combat.tactics_v2 import V2Pilot
from microduck_lab.combat.team_intents import Assignment, TeamPlanner, escape_waypoint
from test_v2 import body, config, context, prepared_evade, team_public


def test_revision_is_explicit_and_old_configs_keep_first_round_behavior():
    assert config().controller_revision == 1
    for value in (True, 0, 4, "2"):
        with pytest.raises(ValueError, match="revision"):
            replace(config(), controller_revision=value)


def test_new_plan_cannot_relabel_an_old_committed_attack_as_a_rescue():
    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    planner = TeamPlanner("RED", teams, revision=2)
    old = Assignment("b", "PRESS", "PRESS_FLANK", 0.8)
    planner.assignments = {"a": Assignment("d", "PEEL", "PEEL_COVER", 1.2, "c")}
    pilot = V2Pilot(
        "a", "RED", teams, replace(config(), controller_revision=2), planner
    )
    pilot.target, pilot.role, pilot.executed_assignment = "b", "PRESS", old
    pilot.action, pilot.phase, pilot.since, pilot.phase_since = 3, "ACTIVE", 1.0, 1.0
    pilot.origin, pilot.aim = np.array((-0.4, -0.15)), 0.0
    pilot.choose(1.25, team_public(), list(teams))
    assert pilot.target == "b" and pilot.role == "PRESS"
    assert not any(e["state"] == "TEAM_ASSIGNMENT_ADOPTED" for e in pilot.events)
    # The request becomes executable at a real maneuver boundary.
    pilot.finish(1.3)
    pilot.choose(1.32, team_public(), list(teams))
    assert pilot.target == "d" and pilot.role == "PEEL"


def test_peel_persists_through_brief_apparent_recovery():
    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    planner = TeamPlanner("RED", teams, revision=2)
    public = team_public()
    public["c"] = body("c", xy=(0.50, 0.3), boundary_distance=0.12)
    planner.update(1.0, public, list(teams))
    assert planner.assignment("a").role == "PEEL"
    protected_target = planner.assignment("a").target
    public["c"] = body("c", xy=(-0.15, 0.0), boundary_distance=0.40)
    planner.update(2.7, public, list(teams))
    assert planner.assignment("a").role == "PEEL"
    assert planner.assignment("a").target == protected_target


def test_public_outward_velocity_requests_support_before_very_edge():
    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    public = team_public()
    public["c"] = body(
        "c", xy=(0.4, 0.3), boundary_distance=0.33, velocity=(0.20, 0.15)
    )
    p = TeamPlanner("RED", teams, revision=2)
    p.update(1.0, public, list(teams))
    assert p.assignment("a").role == "PEEL"


def test_eliminated_rescuer_does_not_leave_a_stale_cover_assignment():
    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    public = team_public()
    public["c"] = body("c", xy=(0.5, 0.3), boundary_distance=0.1)
    planner = TeamPlanner("RED", teams, revision=2)
    planner.update(1.0, public, list(teams))
    assert planner.assignment("a").role == "PEEL"
    planner.update(1.1, public, ["c", "b", "d"])
    assert set(planner.assignments) == {"c"}
    assert planner.assignment("c").intent == "SOLO"


def test_unchanged_plan_retains_its_original_commitment_time():
    teams = dict(a="RED", c="RED", b="BLUE", d="BLUE")
    public = team_public()
    planner = TeamPlanner("RED", teams, revision=2)
    planner.update(1.0, public, list(teams))
    first = planner.assignment("a")
    planner.update(2.7, public, list(teams))
    assert planner.assignment("a") == first


def test_escape_route_goes_around_a_pursuer_on_the_inward_line():
    me = body("a", xy=(0.55, 0.0), boundary_distance=0.18, yaw=math.pi)
    pursuer = body("b", xy=(0.32, 0.0))
    point = escape_waypoint(me, pursuer, dict(a=me, b=pursuer), 0.85)
    assert np.linalg.norm(point) < np.linalg.norm(me.xy)
    assert abs(point[1]) > 0.1


def test_counter_rechecks_actual_motor_state_after_turning():
    p, public = prepared_evade()
    p.c = replace(p.c, controller_revision=2)
    p.phase, p.phase_since, p.pending_counter = "PIVOT", 1.5, True
    # Entry was safe earlier, but a fast turn is still ongoing now.
    p.motor_context = context(yaw_rate=3.0)
    p.choose(1.62, public, list(public))
    assert p.phase == "PIVOT" and p.action == 4
    assert not any(e["state"] == "COUNTER_ATTEMPT" for e in p.events)
