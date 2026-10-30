from dataclasses import replace
import math

import numpy as np
import pytest

from microduck_lab.combat.match_config import MatchConfig
from microduck_lab.combat.observations import Body
from microduck_lab.combat.tactics import Pilot, RuleSelector, features, FEATURES
from microduck_lab.combat.recovery import RecoveryGuard
from microduck_lab.combat.referee import Referee
from microduck_lab.combat.strikes import StrikeAudit


def body(name="a", **kw):
    return replace(Body(name, (0.,0.), .12, (0.,0.), 0., 1., True, False, .75), **kw)


def test_competition_cannot_silently_enable_hardware_or_uncapped_commands():
    for changes in (dict(mode="hardware"), dict(players=3), dict(dash_speed=2.), dict(tactics_path="missing.npz"), dict(ruleset="damage")):
        with pytest.raises(ValueError):
            MatchConfig(**changes)


def test_public_tactical_features_are_identity_invariant_and_include_affiliation():
    me, target = body(), body("b", xy=(.35,0.))
    friend = body("friend", xy=(.1,.2))
    a = features(me,target,[(friend,True)],.75,1.)
    b = features(replace(me,name="renamed"),replace(target,name="other"),[(friend,True)],.75,1.)
    assert a.shape == (FEATURES,) and np.array_equal(a,b)
    enemy = features(me,target,[(friend,False)],.75,1.)
    assert not np.array_equal(a,enemy)
    assert np.isfinite(a).all()


def test_held_attack_does_not_accept_a_new_tactical_choice_midflight():
    class Counter:
        calls = 0
        def select(self, obs, mask, context):
            self.calls += 1
            return 3, dict(log_probability=0., value=0.)
    selector = Counter()
    pilot = Pilot("a","RED",{"a":"RED","b":"BLUE"},MatchConfig(),selector)
    public = {"a":body(), "b":body("b",xy=(.3,0.))}
    pilot.choose(1.,public,list(public))
    pilot.choose(1.14,public,list(public))
    public["b"] = body("b",xy=(.3,.4))
    command = pilot.choose(1.16,public,list(public))
    assert selector.calls == 1 and pilot.active_attack == "a:1"
    assert command.twist[2] == 0.  # locked launch heading, not perfect pursuit
    public["a"] = body(upright=.4,foot_support=False,body_ground=True)
    command = pilot.choose(1.18,public,list(public))
    assert command.owner == "recovery" and pilot.active_attack is None
    assert pilot.trace[-1]["duration_s"] == pytest.approx(.18)


def test_reverse_exit_does_not_use_forward_only_candidate():
    pilot = Pilot("a","RED",{"a":"RED","b":"BLUE"},MatchConfig())
    pilot.action, pilot.phase, pilot.skill = 3,"RECOVERY","dash_bump"
    pilot.phase_since, pilot.since, pilot.target = 1.,0.,"b"
    cmd = pilot.choose(1.2,{"a":body(),"b":body("b",xy=(.25,0.))},["a","b"])
    assert cmd.policy == "walk" and cmd.twist[0] < 0


def test_recovery_needs_supported_dwell_and_logs_actual_burst_policy():
    guard = RecoveryGuard("a")
    fallen = body(upright=0.,foot_support=False,body_ground=True)
    assert guard.choose(0.,fallen) == "stand"
    assert guard.choose(.8,fallen) == "sitstand"
    assert guard.events[-1]["policy"] == "sitstand"
    # Upright but airborne is not completion.
    for t in np.arange(1.6,2.,.02):
        guard.choose(float(t),body(foot_support=False))
    assert guard.active
    for t in np.arange(2.,2.34,.02):
        guard.choose(float(t),body())
    assert not guard.active and guard.events[-1]["state"] == "RECOVERED"


@pytest.mark.parametrize("names",[("a","b","c","d"),("d","c","b","a")])
def test_team_win_waits_for_simultaneous_eliminations(names):
    teams = {"a":"RED","c":"RED","b":"BLUE","d":"BLUE"}
    referee = Referee(names,10,teams)
    state = {n:body(n) for n in names}
    for n in ("b","d"):
        state[n] = body(n,z=-.3,foot_support=False)
    referee.update(0.,state); referee.update(.14,state)
    assert referee.result["winner"] == "RED" and set(referee.eliminated) == {"b","d"}
    both = Referee(names,10,teams)
    state = {n:body(n,z=-.3,foot_support=False) for n in names}
    both.update(0.,state); both.update(.14,state)
    assert both.result["kind"] == "DRAW"


def test_air_swing_and_friendly_collision_never_receive_strike_reward():
    audit = StrikeAudit({"a":"RED","b":"RED"})
    before={"a":body(),"b":body("b",xy=(.3,0.))}
    after={**before,"b":body("b",xy=(.5,0.))}
    assert audit.observe(1.,[],{},before,after) == []
    collision = dict(actor_pair=["a","b"], event_id=0, attacks=["a:1"], normal_impulse_on_second_Ns=[.1,0,0])
    assert audit.observe(1.,[collision],{},before,after) == []
