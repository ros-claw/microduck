from dataclasses import replace
import math

import pytest

from microduck_lab.combat.braking import SkillBrake
from microduck_lab.combat.geometry import boundary_distance
from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.observations import Body
from microduck_lab.combat.recovery import RecoveryGuard


def body(**changes):
    return replace(
        Body("a", (0.0, 0.0), 0.12, (0.0, 0.0), 0.0, 1.0, True, False, 0.8), **changes
    )


def test_zero_command_exit_accepts_settled_learned_equilibrium_but_rejects_swinging_head():
    brake = SkillBrake()
    for i in range(20):
        done = brake.ready(
            i * 0.02, body(head_pitch_delta=0.33 + (0.08 if i % 2 else 0.0))
        )
        assert not done
    for i in range(20, 40):
        done = brake.ready(i * 0.02, body(head_pitch_delta=0.33))
    assert done
    brake.reset()
    for i in range(30):
        assert not brake.ready(
            i * 0.02, body(head_pitch_delta=0.33, foot_support=False)
        )


def test_unqualified_roll_never_enters_production_recovery_even_on_retry():
    guard = RecoveryGuard("a")
    fallen = body(upright=-0.3, foot_support=False, body_ground=True)
    selected = [guard.choose(i * 0.02, fallen) for i in range(450)]
    assert "sitstand" in selected and "roulade" not in selected
    assert not any(e["state"] == "RECOVERED" for e in guard.events)


def test_octagon_boundary_uses_physical_faces_instead_of_circumcircle():
    config = MatchConfig(arena="fixed_octagon", radius_m=1.0)
    face = (math.cos(math.pi / 8), math.sin(math.pi / 8))
    assert boundary_distance((0.0, 0.0), config) == pytest.approx(math.cos(math.pi / 8))
    assert boundary_distance(tuple(0.95 * v for v in face), config) < 0
    assert (
        boundary_distance(
            tuple(0.95 * v for v in face), replace(config, arena="fixed_disc")
        )
        > 0
    )
    assert boundary_distance((1.0, 0.0), config) == pytest.approx(0.0)
