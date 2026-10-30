"""Interrupted maneuvers cannot become misses or inflate counter statistics."""

import pytest
from microduck_lab.combat.events import attack_outcomes, counter_sequences


@pytest.mark.parametrize(
    "reason", ["RECOVERY_PREEMPTION", "ELIMINATED", "EPISODE_END", "TARGET_ELIMINATED"]
)
def test_interrupted_attack_is_not_a_miss_or_a_counter_opportunity(reason):
    events = [
        dict(
            state="ATTACK_LAUNCHED", time_s=1.0, actor="b", target="a", attack_id="b:1"
        ),
        dict(state="MANEUVER", time_s=1.05, actor="a", skill="evade_step"),
        dict(
            state="ATTACK_WINDOW_ENDED",
            time_s=1.7,
            actor="b",
            target="a",
            attack_id="b:1",
            reason=reason,
        ),
        dict(state="VALID_STRIKE", time_s=1.8, actor="a", target="b", attack_id="a:1"),
    ]
    outcomes = attack_outcomes(events, [], {"a": "RED", "b": "BLUE"})
    assert outcomes[0]["state"] == "ATTACK_INTERRUPTED"
    assert not outcomes[0]["completed"] and not outcomes[0]["contacted"]
    assert counter_sequences(events + outcomes) == []
    events[2]["reason"] = "COMPLETE"
    assert (
        len(
            counter_sequences(
                events + attack_outcomes(events, [], {"a": "RED", "b": "BLUE"})
            )
        )
        == 1
    )


def test_interruption_retains_real_contact_without_falsely_completing_the_attack():
    events = [
        dict(
            state="ATTACK_WINDOW_ENDED",
            time_s=2.0,
            actor="a",
            attack_id="a:1",
            reason="ELIMINATED",
        )
    ]
    contacts = [dict(event_id=3, attacks=["a:1"], actor_pair=["a", "b"])]
    outcome = attack_outcomes(events, contacts, {"a": "RED", "b": "BLUE"})[0]
    assert outcome["state"] == "ATTACK_INTERRUPTED" and outcome["contacted"]
    assert outcome["contact_episode_ids"] == [3]


def test_legacy_completed_lab_windows_keep_their_explicit_contact_outcome():
    event = dict(state="ATTACK_WINDOW_ENDED", time_s=2.0, actor="a", attack_id="a:1")
    outcome = attack_outcomes([event], [])[0]
    assert outcome["state"] == "MISS" and outcome["completed"]
