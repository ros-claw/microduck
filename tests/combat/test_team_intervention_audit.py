import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def qualifies(monkeypatch):
    scripts = Path(__file__).parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "team_audit_tested", scripts / "eval_combat_team_intents.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.qualifies_intervention


def test_near_instant_incidental_contact_is_not_a_planned_intervention(qualifies):
    assert not qualifies(0.004, 0.0, "PEEL", "enemy", "enemy", True)


def test_abandoned_peel_role_cannot_claim_later_collision(qualifies):
    assert not qualifies(1.0, 0.1, "PRESS", "enemy", "enemy", True)


def test_plan_target_mismatch_is_not_team_intervention(qualifies):
    assert not qualifies(1.0, 0.1, "PEEL", "other", "enemy", True)
    assert not qualifies(1.0, 0.1, "PEEL", "enemy", "enemy", False)


def test_retained_intent_with_actual_motion_and_later_contact_can_qualify(qualifies):
    assert qualifies(1.0, 0.1, "PEEL", "enemy", "enemy", True)
