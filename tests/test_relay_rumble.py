"""Moving-goal rules must remain grounded in support and public geometry."""

from types import SimpleNamespace
import numpy as np
import pytest
from microduck_lab.arena.relay_rumble import (
    RelayConfig,
    RelayRace,
    ContactTriggeredCollapse,
    CHARACTERS,
)
from microduck_lab.arena.tiles import Tile


def observations(config, goal=12):
    x, y = config.centre(goal)
    return {
        "a": dict(x=x, y=y, z=0.12, upright=True, supporting_tiles=[goal]),
        "b": dict(x=10.0, y=10.0, z=0.12, upright=True, supporting_tiles=[]),
    }


def test_beacon_moves_after_actual_capture_and_opponents_can_take_next_point():
    cfg = RelayConfig(final_at_s=0)
    race = RelayRace(cfg, ("a", "b"))
    o = observations(cfg)
    for i in range(3000):
        assert race.update((i + 1) * cfg.dt, o, {}) is None
    assert race.scores == {"a": 1, "b": 0}
    assert race.goal == 2
    assert race.progress == {"a": 0.0, "b": 0.0}
    assert [e["state"] for e in race.events] == ["BEACON_CAPTURE", "BEACON_MOVED"]
    # Staying on the previous island earns nothing after it moves.
    race.update(1.0, o, {})
    assert race.progress == {"a": 0.0, "b": 0.0}
    o = observations(cfg, 2)
    o["b"], o["a"] = o["a"], o["b"]
    for i in range(3000):
        assert race.update(1.0 + i * cfg.dt, o, {}) is None
    assert race.scores == {"a": 1, "b": 1} and race.goal == 14


@pytest.mark.parametrize("invalid", ("airborne", "toppled", "contested", "eliminated"))
def test_no_score_from_proximity_only_or_invalid_contact(invalid):
    cfg = RelayConfig(final_at_s=0)
    race = RelayRace(cfg, ("a", "b"))
    o = observations(cfg)
    eliminated = {}
    if invalid == "airborne":
        o["a"]["supporting_tiles"] = []
    if invalid == "toppled":
        o["a"]["upright"] = False
    if invalid == "contested":
        o["b"].update(x=o["a"]["x"], y=o["a"]["y"], supporting_tiles=[])
    if invalid == "eliminated":
        eliminated = {"a": {}}
    for i in range(4000):
        assert race.update((i + 1) * cfg.dt, o, eliminated) is None
    assert race.scores == {"a": 0, "b": 0} and race.progress == {"a": 0.0, "b": 0.0}


def test_permanent_stations_and_load_consumed_bridges():
    cfg = RelayConfig(lifetime_s=3)
    d = SimpleNamespace(eq_active=np.ones(25, dtype=bool))
    tiles = {i: Tile(i, i, 0, 0) for i in cfg.ids}
    damage = ContactTriggeredCollapse(cfg, None, 7.23)
    for k in range(14000):
        damage.integrate(
            d,
            tiles,
            {i: 7.23 if i in (2, 7) else 0 for i in cfg.ids},
            {i: {"a"} for i in cfg.ids},
            (k + 1) * cfg.dt,
        )
    assert not d.eq_active[7] and d.eq_active[2]
    assert damage.damage[2] == 0 and damage.damage[7] >= 1
    assert damage.damage[8] == 0


def test_names_map_to_fixed_distinct_roles_and_larger_arena():
    assert {v["name"] for v in CHARACTERS.values()} == {"张三", "二呆", "老六", "卷王"}
    assert len({v["role"] for v in CHARACTERS.values()}) == 4
    cfg = RelayConfig()
    assert len(cfg.ids) == 25
    assert cfg.grid * cfg.pitch > 2.0
    with pytest.raises(ValueError):
        RelayConfig(grid=3)
