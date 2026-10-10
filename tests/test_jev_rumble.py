import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_lab.arena.jev_tactics import JevMotor, RecordedTactics, candidates, make_request, public_route, validate
from microduck_lab.arena.progressive_collapse import ProgressiveCollapse, connected
from microduck_lab.arena.survival_rumble import SurvivalConfig
from microduck_lab.arena.tiles import Tile


def observation():
    c = SurvivalConfig()
    return dict(sim_time_s=1., island=14, robot=dict(body_id="lavender", x=0., y=0., z=.15,
                yaw=0., current_tile_id=12, speed_m_s=0., up_cos=1.), nearby_ducks=[],
                visible_tiles=[dict(id=i, state="LOCKED", damage=0., warning_remaining_s=None,
                                    centre_xy=c.centre(i).tolist()) for i in c.ids])


def scheduler():
    c = SurvivalConfig()
    tiles = {i: Tile(i, i, 0, 0) for i in c.ids}
    d = SimpleNamespace(eq_active=np.ones(25, dtype=bool))
    graph = SimpleNamespace(tiles={}, feet={})
    return ProgressiveCollapse(c, graph, 4.), tiles, d


def test_progressive_releases_are_early_serial_and_connected():
    rule, tiles, d = scheduler()
    events = []
    for now in np.arange(0., 90., .02):
        events.extend(rule.advance(d, tiles, [], float(now), {}, set()))
        assert sum(t.state == "WARNING" for t in tiles.values()) <= 1
        assert connected({i for i in tiles if d.eq_active[i]})
    releases = [e for e in events if e["state"] == "RELEASED"]
    assert len(releases) == 24
    assert releases[0]["time"] == pytest.approx(9.2)
    assert all(b["time"]-a["time"] >= 3.2-1e-9 for a, b in zip(releases, releases[1:]))
    assert all(e["warning_s"] >= 3.2-1e-9 for e in releases)
    assert d.eq_active[12]


def test_grace_is_bounded_and_independent_of_identity():
    outcomes = []
    for name in ("lavender", "cream", "sky", "graphite"):
        rule, tiles, d = scheduler()
        rule.advance(d, tiles, [], 6., {}, set())
        i = rule.pending
        c = rule.config.centre(i)
        fallen = {name: dict(x=c[0], y=c[1], z=.1, upright=False)}
        e = rule.advance(d, tiles, [], 9.2, fallen, set())
        assert e[0]["state"] == "RECOVERY_GRACE"
        assert d.eq_active[i]
        e = rule.advance(d, tiles, [], 10.4, fallen, set())
        assert e[0]["state"] == "RELEASED"
        outcomes.append((i, e[0]["time"]))
    assert len(set(outcomes)) == 1


def test_disconnected_and_expiring_destinations_are_not_offered():
    o = observation()
    for t in o["visible_tiles"]:
        if t["id"] in (13, 9, 19):
            t.update(state="RELEASED")
    assert public_route(o, 14) is None
    assert "CAPTURE" not in candidates(o)
    o = observation()
    o["visible_tiles"][14].update(state="WARNING", warning_remaining_s=.5)
    assert "CAPTURE" not in candidates(o)


def test_recovery_preempts_jev_until_stable_actual_support():
    o = observation()
    motor = JevMotor("rusher")
    plan = dict(action="CAPTURE", tile=14, expires_s=10.)
    o["robot"]["up_cos"] = .1
    assert motor.choose(o, plan) == ("stand", (0, 0, 0), "LOCAL_RECOVERY")
    o["robot"].update(up_cos=1., current_tile_id=None)
    o["sim_time_s"] = 2.
    assert motor.choose(o, plan)[2] == "LOCAL_RECOVERY"
    o["robot"]["current_tile_id"] = 12
    o["sim_time_s"] = 2.1
    assert motor.choose(o, plan)[2] == "LOCAL_RECOVERY"
    o["sim_time_s"] = 2.3
    assert motor.choose(o, plan)[2] == "JEV_CAPTURE"
    assert motor.events[-1]["state"] == "RECOVERY_COMPLETED"


def test_untrusted_response_rejects_nan_missing_and_illegal_choice():
    opts = {"lavender": {"CAPTURE": dict(tile=12)}}
    raw = dict(model="jev-test", answers={"lavender": dict(type="choice", choice="CAPTURE", confidence=.7, probabilities={"CAPTURE": 1.})})
    assert validate(raw, opts)["lavender"]["tile"] == 12
    for change in (dict(choice="TELEPORT"), dict(confidence=float("nan")), dict(probabilities={"OTHER": 1.})):
        bad = copy.deepcopy(raw)
        bad["answers"]["lavender"].update(change)
        with pytest.raises(ValueError):
            validate(bad, opts)


def test_jev_goal_changes_motor_command_without_changing_beacon():
    o = observation()
    capture = JevMotor("rusher").choose(o, dict(action="CAPTURE", tile=14, expires_s=5.))
    evade = JevMotor("rusher").choose(o, dict(action="EVADE", tile=10, expires_s=5.))
    assert capture[1][2] > 0 and evade[1][2] < 0
    assert o["island"] == 14
    assert JevMotor("rusher").choose(o, None)[2] == "AWAIT_JEV"
    assert JevMotor("rusher").choose(o, dict(action="CAPTURE", tile=14, expires_s=.9))[2] == "AWAIT_JEV"


def test_replay_rejects_tampered_request_state(tmp_path):
    import json
    obs = {"lavender": observation()}
    opts = {n: candidates(o) for n, o in obs.items()}
    row = dict(id=0, submitted_s=1., request=make_request(obs, opts))
    tape = tmp_path/"tape.json"
    tape.write_text(json.dumps([row]))
    replay = RecordedTactics(tape, tmp_path/"copy.json")
    replay.update(1., obs)
    replay.close()
    replay = RecordedTactics(tape, tmp_path/"copy.json")
    obs["lavender"]["robot"]["x"] = .01
    with pytest.raises(ValueError, match="public request differs"):
        replay.update(1., obs)


def test_warned_floor_evacuates_while_jev_holds_or_waits():
    from microduck_lab.arena.jev_fairness import FairMotor
    for plan in (None, dict(action="HOLD", tile=12, expires_s=5.)):
        o = observation()
        o["visible_tiles"][12].update(state="WARNING", warning_remaining_s=4.)
        motor = FairMotor("survivor")
        result = motor.choose(o, plan)
        assert result[2] == "LOCAL_EVACUATE"
        assert motor.target in (7, 11, 13, 17)


def test_recovery_skill_watchdog_is_one_bounded_motor_burst():
    from microduck_lab.arena.jev_recovery import RecoveryMotor
    o = observation()
    o["robot"].update(up_cos=-.15, speed_m_s=.01, z=.08)
    motor = RecoveryMotor("survivor")
    assert motor.choose(o, None)[0] == "stand"
    o["sim_time_s"] = 1.82
    assert motor.choose(o, None) == ("sitstand", (0, 0, 0), "LOCAL_RECOVERY_SKILL")
    o["sim_time_s"] = 2.64
    assert motor.choose(o, None)[0] == "stand"
    o["sim_time_s"] = 8.
    assert motor.choose(o, None)[0] == "stand"
    assert sum(e["state"] == "RECOVERY_SKILL_STARTED" for e in motor.events) == 1


def test_sole_survivor_wins_on_real_remaining_floor_without_centre_trip():
    from microduck_lab.arena.grounded_victory import GroundedVictory
    c = SurvivalConfig()
    rule = GroundedVictory(c)
    o = dict(cream=dict(upright=True, z=.12, supporting_tiles=[7]))
    result = None
    for step in range(3001):
        result = rule.update(step*c.dt, o, set())
    assert result["winner"] == "cream" and result["supporting_tiles_at_win"] == [7]
    rule = GroundedVictory(c)
    o["cream"]["supporting_tiles"] = []
    assert rule.update(10., o, set()) is None


def test_moving_fall_gets_time_to_finish_standing_before_skill_switch():
    from microduck_lab.arena.jev_recovery import RecoveryMotor
    o = observation()
    o["robot"]["up_cos"] = .80
    motor = RecoveryMotor("survivor")
    assert motor.choose(o, None)[0] == "stand"
    o["robot"]["up_cos"] = .10
    o["sim_time_s"] = 1.82
    assert motor.choose(o, None)[0] == "stand"
    o["sim_time_s"] = 3.02
    assert motor.choose(o, None)[0] == "sitstand"


def test_match_budget_covers_serial_floor_countdowns():
    from microduck_lab.arena.jev_config import JevGameConfig
    c = JevGameConfig()
    assert c.duration >= 6.+24*4.+15.
    with pytest.raises(ValueError):
        JevGameConfig(duration=121.)


def test_valid_low_confidence_answer_is_not_silently_replaced(tmp_path):
    import json
    obs = {"lavender": observation()}
    opts = {n: candidates(o) for n, o in obs.items()}
    choices = opts["lavender"]
    raw = dict(model="test", answers=dict(lavender=dict(type="choice", choice="CAPTURE", confidence=.1,
               probabilities={k: 1./len(choices) for k in choices})))
    row = dict(id=0, submitted_s=1., delivered_s=1.1, request=make_request(obs, opts), raw=raw,
               validated_plans=validate(raw, opts), accepted=dict(lavender=True))
    tape = tmp_path/"tape.json"
    tape.write_text(json.dumps([row]))
    replay = RecordedTactics(tape, tmp_path/"copy.json")
    replay.update(1., obs)
    plans = replay.update(1.1, obs)
    assert plans["lavender"]["action"] == "CAPTURE"
    replay.close()


def test_archived_actor_sources_unchanged():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    # Frozen source hashes from 14d6e63; works in a shallow checkout or ZIP.
    expected = {'survival_rumble.py': 'c1c1f89e674c920c4fcc8b1b1cfcc4520f4d4d4b54a52a09246c68306359b893', 'survival_world.py': '2b13d1dcc217842ef89c2098c3ef6ae649b6d2b62e1b89a50a0d79162b610645', 'relay_rumble.py': 'f6c11d9f587290804cabeda9688e9ecf20c4d510ea13f5c28e94fa280d001357', 'multiplayer.py': '0379bff83a24cb7b727b5a981cb75aed5545cab1fe4126f63901109d2791cad6'}
    for name, digest in expected.items():
        assert hashlib.sha256((root/"src/microduck_lab/arena"/name).read_bytes()).hexdigest() == digest
