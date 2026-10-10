"""Survival may never award points victory or silently choose a tie-breaker."""

from types import SimpleNamespace
import numpy as np
import pytest
from microduck_lab.arena.survival_rumble import (
    SurvivalConfig,
    LastDuckVictory,
    SurvivalFinale,
    RoamingBeacons,
)
from microduck_lab.arena.tiles import Tile


def bodies():
    return {
        n: dict(x=0.0, y=0.0, z=0.12, upright=True, supporting_tiles=[12])
        for n in ("a", "b", "c", "d")
    }


def test_three_eliminations_and_stable_real_support_are_required():
    cfg = SurvivalConfig()
    v = LastDuckVictory(cfg)
    o = bodies()
    for k in range(4000):
        assert v.update(k * cfg.dt, o, {"b": {}, "c": {}}) is None
    result = None
    for k in range(4000):
        result = v.update(k * cfg.dt, o, {"b": {}, "c": {}, "d": {}})
        if result:
            break
    assert result["alive"] == ["a"] and result["winner"] == "a"
    assert result["stable_supported_s"] >= 0.75
    assert result["loaded_fraction"] >= 0.9


@pytest.mark.parametrize(
    "invalid", ("airborne", "toppled", "below_arena", "wrong_tile")
)
def test_sole_body_cannot_win_without_valid_core_support(invalid):
    cfg = SurvivalConfig()
    v = LastDuckVictory(cfg)
    o = bodies()
    if invalid == "airborne":
        o["a"]["supporting_tiles"] = []
    if invalid == "wrong_tile":
        o["a"]["supporting_tiles"] = [2]
    if invalid == "toppled":
        o["a"]["upright"] = False
    if invalid == "below_arena":
        o["a"]["z"] = -0.2
    for k in range(4000):
        assert v.update(k * cfg.dt, o, {"b": {}, "c": {}, "d": {}}) is None


def test_support_loss_resets_stability_and_all_out_is_draw():
    cfg = SurvivalConfig()
    v = LastDuckVictory(cfg)
    o = bodies()
    gone = {"b": {}, "c": {}, "d": {}}
    for k in range(2000):
        assert v.update(k * cfg.dt, o, gone) is None
    o["a"]["supporting_tiles"] = []
    assert v.update(0.54, o, gone) is None
    assert v.candidate is None
    o["a"]["supporting_tiles"] = [12]
    assert v.update(0.55, o, gone) is None
    assert v.update(0.8, o, gone) is None
    assert v.update(0.81, o, dict.fromkeys(o))["status"] == "DRAW"


def test_many_beacon_points_never_end_a_survival_match():
    cfg = SurvivalConfig(final_at_s=0)
    race = RoamingBeacons(cfg, ("a", "b"))
    now = 0
    for _ in range(6):
        x, y = cfg.centre(race.goal)
        o = {
            "a": dict(x=x, y=y, z=0.12, upright=True, supporting_tiles=[race.goal]),
            "b": dict(x=10, y=10, z=0.12, upright=True, supporting_tiles=[]),
        }
        for _ in range(3001):
            now += cfg.dt
            assert race.update(now, o, {}) is None
    assert race.scores["a"] >= 5
    assert not any(e["state"] == "TERMINAL" for e in race.events)


def test_finale_announces_two_rings_and_releases_only_real_constraints():
    cfg = SurvivalConfig()
    f = SurvivalFinale(cfg)
    d = SimpleNamespace(eq_active=np.ones(25, dtype=bool))
    tiles = {i: Tile(i, i, 0, 0) for i in cfg.ids}
    events = f.advance(d, tiles, 20, 5, 4)
    assert events[0]["state"] == "FINALE_WARNING"
    assert d.eq_active.all() and tiles[12].state == "LOCKED"
    f.advance(d, tiles, 20 + cfg.return_warning_s - 0.001, 5, 4)
    assert d.eq_active.all()
    f.advance(d, tiles, 20 + cfg.return_warning_s, 5, 4)
    assert sum(d.eq_active) == 9 and d.eq_active[12]
    f.advance(d, tiles, 20 + cfg.return_warning_s + cfg.inner_delay_s, 5, 4)
    assert sum(d.eq_active) == 1 and d.eq_active[12]
    assert any(abs(v) > 0 for v in f.targets(45, (0, 0), 2))
    angles = (0.04, -0.02)
    assert f.targets(46, angles, 1) == angles
    assert f.targets(47, (0.1, 0.1), 1) == angles


def test_gimbal_is_actuated_geometry_with_grounded_base():
    from microduck_lab.arena.survival_world import build_survival_world
    from microduck_lab.arena.world import ASSETS

    if not (ASSETS / "microduck/policies/alpha_stand.onnx").exists():
        pytest.skip("native assets")
    m, d, ducks, tiles, _ = build_survival_world(SurvivalConfig(), 61001)
    assert len(ducks) == 4 and len(tiles) == 25
    assert d.eq_active[m.equality("tile_12_support").id]
    assert m.body("tile_12").parentid == m.body("arena_roll_body").id
    for axis in ("roll", "pitch"):
        a = m.actuator("arena_" + axis).id
        assert np.allclose(m.actuator_forcerange[a], [-4, 4])
        assert m.actuator_trnid[a, 0] == m.joint("arena_" + axis + "_joint").id
