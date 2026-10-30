from dataclasses import replace

import numpy as np
import pytest

from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.observations import Body
from microduck_lab.combat.tactics import Pilot, features, FEATURES
from microduck_lab.combat.recovery import RecoveryGuard
from microduck_lab.combat.referee import Referee
from microduck_lab.combat.strikes import StrikeAudit
from microduck_lab.combat.learning import semi_mdp_gae
from microduck_lab.combat.arena import build_match, match_inventory
from microduck_lab.combat.config import Physics


def body(name="a", **kw):
    return replace(
        Body(name, (0.0, 0.0), 0.12, (0.0, 0.0), 0.0, 1.0, True, False, 0.75), **kw
    )


def test_competition_cannot_silently_enable_hardware_or_uncapped_commands():
    for changes in (
        dict(mode="hardware"),
        dict(players=3),
        dict(dash_speed=2.0),
        dict(tactics_path="missing.npz"),
        dict(ruleset="damage"),
    ):
        with pytest.raises(ValueError):
            MatchConfig(**changes)


def test_public_tactical_features_are_identity_invariant_and_include_affiliation():
    me, target = body(), body("b", xy=(0.35, 0.0))
    friend = body("friend", xy=(0.1, 0.2))
    a = features(me, target, [(friend, True)], 0.75, 1.0)
    b = features(
        replace(me, name="renamed"),
        replace(target, name="other"),
        [(friend, True)],
        0.75,
        1.0,
    )
    assert a.shape == (FEATURES,) and np.array_equal(a, b)
    enemy = features(me, target, [(friend, False)], 0.75, 1.0)
    assert not np.array_equal(a, enemy)
    assert np.isfinite(a).all()


def test_held_attack_does_not_accept_a_new_tactical_choice_midflight():
    class Counter:
        calls = 0

        def select(self, obs, mask, context):
            self.calls += 1
            return 3, dict(log_probability=0.0, value=0.0)

    selector = Counter()
    pilot = Pilot(
        "a", "RED", {"a": "RED", "b": "BLUE"}, MatchConfig(dash_windup_s=0.12), selector
    )
    public = {"a": body(), "b": body("b", xy=(0.3, 0.0))}
    pilot.choose(1.0, public, list(public))
    pilot.choose(1.14, public, list(public))
    public["b"] = body("b", xy=(0.3, 0.4))
    command = pilot.choose(1.16, public, list(public))
    assert selector.calls == 1 and pilot.active_attack == "a:1"
    assert command.twist[2] == 0.0  # locked launch heading, not perfect pursuit
    public["a"] = body(upright=0.4, foot_support=False, body_ground=True)
    command = pilot.choose(1.18, public, list(public))
    assert command.owner == "recovery" and pilot.active_attack is None
    assert pilot.trace[-1]["duration_s"] == pytest.approx(0.18)


def test_reverse_exit_does_not_use_forward_only_candidate():
    pilot = Pilot("a", "RED", {"a": "RED", "b": "BLUE"}, MatchConfig())
    pilot.action, pilot.phase, pilot.skill = 3, "RECOVERY", "dash_bump"
    pilot.phase_since, pilot.since, pilot.target = 1.0, 0.0, "b"
    cmd = pilot.choose(1.2, {"a": body(), "b": body("b", xy=(0.25, 0.0))}, ["a", "b"])
    assert cmd.policy == "walk" and cmd.twist[0] < 0


def test_recovery_needs_supported_dwell_and_logs_actual_burst_policy():
    guard = RecoveryGuard("a")
    fallen = body(upright=0.0, foot_support=False, body_ground=True)
    assert guard.choose(0.0, fallen) == "stand"
    assert guard.choose(0.8, fallen) == "sitstand"
    assert guard.events[-1]["policy"] == "sitstand"
    # Upright but airborne is not completion.
    for t in np.arange(1.6, 2.0, 0.02):
        guard.choose(float(t), body(foot_support=False))
    assert guard.active
    for t in np.arange(2.0, 2.34, 0.02):
        guard.choose(float(t), body())
    assert not guard.active and guard.events[-1]["state"] == "RECOVERED"


@pytest.mark.parametrize("names", [("a", "b", "c", "d"), ("d", "c", "b", "a")])
def test_team_win_waits_for_simultaneous_eliminations(names):
    teams = {"a": "RED", "c": "RED", "b": "BLUE", "d": "BLUE"}
    referee = Referee(names, 10, teams)
    state = {n: body(n) for n in names}
    for n in ("b", "d"):
        state[n] = body(n, z=-0.3, foot_support=False)
    referee.update(0.0, state)
    referee.update(0.14, state)
    assert referee.result["winner"] == "RED" and set(referee.eliminated) == {"b", "d"}
    both = Referee(names, 10, teams)
    state = {n: body(n, z=-0.3, foot_support=False) for n in names}
    both.update(0.0, state)
    both.update(0.14, state)
    assert both.result["kind"] == "DRAW"


def test_air_swing_and_friendly_collision_never_receive_strike_reward():
    audit = StrikeAudit({"a": "RED", "b": "RED"})
    before = {"a": body(), "b": body("b", xy=(0.3, 0.0))}
    after = {**before, "b": body("b", xy=(0.5, 0.0))}
    assert audit.observe(1.0, [], {}, before, after) == []
    collision = dict(
        actor_pair=["a", "b"],
        event_id=0,
        attacks=["a:1"],
        normal_impulse_on_second_Ns=[0.1, 0, 0],
    )
    assert audit.observe(1.0, [collision], {}, before, after) == []


def test_semi_mdp_gae_discounts_seconds_and_stops_at_terminal():
    trajectory = [
        dict(duration_s=0.2, reward=0.0, value=0.0, terminal=False),
        dict(duration_s=2.0, reward=1.0, value=0.0, terminal=True),
    ]
    _, returns = semi_mdp_gae(trajectory, gamma_per_second=0.5, lambda_per_second=1.0)
    assert returns[0] == pytest.approx(0.5**0.2)
    trajectory[0]["duration_s"] = 2.0
    _, slower = semi_mdp_gae(trajectory, gamma_per_second=0.5, lambda_per_second=1.0)
    assert slower[0] == pytest.approx(0.25)
    trajectory[0]["terminal"] = True
    _, terminal = semi_mdp_gae(trajectory)
    assert terminal[0] == 0.0


def test_four_ducks_share_one_world_and_friendly_contacts_are_not_exempt():
    model, data, ducks, spawns, teams = build_match(
        MatchConfig(players=4, radius_m=0.85), Physics(), 10201
    )
    assert model.nu == 56 and model.neq == 0 and len(ducks) == 4
    inventory = match_inventory(model, ducks)
    assert list(inventory["mass_by_duck"].values()) == pytest.approx([0.73724318] * 4)
    assert len({id(d.last_action) for d in ducks.values()}) == 4
    first = {}
    for name, duck in ducks.items():
        assert len(duck.act_ids) == 14 and duck.get_obs().shape == (61,)
        shell = [
            g
            for g in range(model.ngeom)
            if model.body(int(model.geom_bodyid[g])).name.startswith(name + "/")
            and "_combat_shell" in model.body(int(model.geom_bodyid[g])).name
        ]
        assert len(shell) == 70
        first[name] = shell[0]
    for a in ducks:
        for b in ducks:
            if a != b:
                assert (
                    model.geom_contype[first[a]] & model.geom_conaffinity[first[b]]
                ) != 0
    assert teams["lavender"] == teams["coral"]
    assert np.all(data.xfrc_applied == 0) and np.all(data.qfrc_applied == 0)
