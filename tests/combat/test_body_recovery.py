from dataclasses import replace
from pathlib import Path

import pytest

from microduck_lab.combat.body_recovery import BodyRecoveryGuard
from microduck_lab.combat.config import strict_load
from microduck_lab.combat.observations import Body
from microduck_lab.combat.v2_config import V2Config


def body(**changes):
    return replace(Body("a", (0, 0), 0.117, (0, 0), 0, 1, True, False, 0.6), **changes)


def test_learned_motor_requires_measured_recent_contact_and_a_real_stumble():
    guard = BodyRecoveryGuard("a", learned=True)
    assert guard.choose(0, body(touching_opponent=True)) is None
    assert guard.choose(0.02, body(upright=0.7)) == "recovery_candidate"
    assert guard.choose(0.14, body(upright=0.7)) == "stand"
    assert guard.choose(0.16, body(upright=0.98, touching_opponent=True)) == "stand"
    assert (
        guard.choose(0.18, body(upright=0.5, z=-0.02, touching_opponent=True))
        == "stand"
    )


def test_stand_ablation_uses_the_same_measured_guard_without_the_new_motor():
    guard = BodyRecoveryGuard("a")
    assert guard.choose(0, body(upright=0.3, touching_opponent=True)) == "stand"
    assert guard.choose(0.5, body(upright=0.3, touching_opponent=True)) == "stand"


def test_stabilization_specialist_hands_off_to_stand_when_upright():
    guard = BodyRecoveryGuard("a", learned=True, contact_only=False)
    assert guard.choose(0, body(upright=0.7)) == "recovery_candidate"
    assert guard.choose(0.1, body(upright=0.95)) == "stand"
    assert guard.active
    guard.choose(0.4, body())
    assert not guard.active


@pytest.mark.parametrize(
    "change",
    [
        dict(z=0.08),
        dict(body_ground=True),
        dict(foot_support=False),
        dict(velocity=(0.3, 0)),
    ],
)
def test_recovery_does_not_finish_in_low_poses_or_without_stable_support(change):
    guard = BodyRecoveryGuard("a", learned=True)
    guard.choose(0, body(upright=0.7))
    for i in range(1, 40):
        guard.choose(i * 0.02, body(**change))
    assert guard.active
    assert not any(e["state"] == "RECOVERED" for e in guard.events)


def test_completion_has_a_continuous_support_dwell_and_settle_handoff():
    guard = BodyRecoveryGuard("a", learned=True)
    guard.choose(0, body(upright=0.7))
    guard.choose(0.1, body())
    guard.choose(0.38, body())
    assert guard.active
    guard.choose(0.4, body())
    assert not guard.active
    assert guard.choose(0.68, body()) == "stand"
    assert guard.choose(0.72, body()) is None


def test_learned_attempt_is_bounded_and_legacy_getup_remains_available():
    guard = BodyRecoveryGuard("a", learned=True)
    for i in range(170):
        guard.choose(i * 0.02, body(upright=0.2, touching_opponent=True))
    assert guard.learned_seconds <= 2.0 + 1e-9
    assert guard.last_policy != "recovery_candidate"
    assert any(e["state"] == "RECOVERY_LEGACY_FALLBACK" for e in guard.events)
    assert guard.choose(4.0, body(upright=0.2, touching_opponent=True)) == "sitstand"


def test_model_scope_is_opt_in_and_requires_a_real_digest():
    cfg = strict_load(
        Path(__file__).parents[2] / "configs/combat/v2/team_open_coordinated.yaml",
        V2Config,
    )
    assert cfg.recovery_mode == "legacy"
    with pytest.raises(ValueError, match="exact motor"):
        replace(cfg, recovery_mode="contact_learned")
    with pytest.raises(ValueError, match="explicit"):
        replace(cfg, recovery_motor_path="anything", recovery_motor_sha256="0" * 64)
