"""Rule invariants, load causality and adversarial claim transitions."""

from types import SimpleNamespace
import numpy as np
import pytest
from microduck_lab.arena.island_rumble import (
    RumbleConfig,
    ContactTriggeredCollapse,
    IslandClaim,
)
from microduck_lab.arena.tiles import Tile


def release_time(loads, impulse=False):
    cfg = RumbleConfig()
    d = SimpleNamespace(eq_active=np.ones(9, dtype=bool))
    tiles = {i: Tile(i, i, 0, 0) for i in cfg.ids}
    scheduler = ContactTriggeredCollapse(cfg, None, 7.23)
    for step in range(35000):
        t = (step + 1) * cfg.dt
        force = loads * 7.23 if not impulse or step == 0 else 0
        events = scheduler.integrate(
            d,
            tiles,
            {i: force if i in (0, 4) else 0 for i in cfg.ids},
            {i: {"a"} for i in cfg.ids},
            t,
        )
        if any(e["state"] == "RELEASED" for e in events):
            assert d.eq_active[4]  # even a heavily loaded gold island stays supported
            return t
    return None


def test_real_load_changes_release_time_and_impulse_does_not_delete_floor():
    single = release_time(1)
    double = release_time(2)
    assert 3.2 < single < 3.4
    assert 1.6 < double < 1.8
    assert double < single * 0.56
    assert release_time(100000, impulse=True) is None
    assert release_time(0) is None


def test_claim_cannot_accumulate_through_contest_or_toppled_occupant():
    cfg = RumbleConfig(final_at_s=0, claim_s=1.5)
    rule = IslandClaim(cfg)
    alone = {
        "a": dict(supporting_tiles=[4], upright=True),
        "b": dict(supporting_tiles=[], upright=True),
    }
    assert rule.update(0, alone, {}) is None
    # Rival carries load on gold even when toppled; must reset exclusivity.
    both = {**alone, "b": dict(supporting_tiles=[4], upright=False)}
    assert rule.update(1.4, both, {}) is None
    assert rule.update(1.5, alone, {}) is None
    assert rule.update(2.99, alone, {}) is None
    result = rule.update(3, alone, {})
    assert result["winner"] == "a" and result["rule"] == "IslandClaim"
    assert (
        rule.update(3.1, {"a": dict(supporting_tiles=[4], upright=False)}, {}) is None
    )


def test_only_loaded_upward_foot_contacts_damage_tiles():
    cfg = RumbleConfig()
    d = SimpleNamespace(eq_active=np.ones(9, dtype=bool))
    tiles = {i: Tile(i, i, 0, 0) for i in cfg.ids}
    graph = SimpleNamespace(tiles={10: 0}, feet={20: "a"})
    rule = ContactTriggeredCollapse(cfg, graph, 7.23)
    # Contact layout: two IDs, dist, 6 forces, position, normal.
    floor_foot = [10, 20, 0, 7.23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    rule.advance(d, tiles, [floor_foot], 0.00025)
    assert rule.damage[0] > 0
    before = rule.filtered[0]
    # Downward, torso and deactivated support cannot inject a new load.
    wrong = [10, 21, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    downward = floor_foot.copy()
    downward[-1] = -1
    rule.advance(d, tiles, [wrong, downward], 0.0005)
    assert rule.filtered[0] < before
    before = rule.damage[0]
    d.eq_active[0] = False
    rule.advance(d, tiles, [floor_foot], 0.00075)
    assert rule.damage[0] == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid": 5},
        {"dt": 0.001},
        {"gap": 0.03},
        {"tile_size": 0.2},
        {"claim_s": 0},
        {"lifetime_s": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RumbleConfig(**kwargs)


def test_crown_claim_is_a_distinct_public_rule_with_real_support():
    cfg = RumbleConfig(final_at_s=0, claim_radius_m=0.08)
    rule = IslandClaim(cfg)
    observations = {
        "a": dict(x=0.02, y=0.01, upright=True, supporting_tiles=[4]),
        "b": dict(x=0.12, y=0.01, upright=True, supporting_tiles=[4]),
    }
    assert rule.update(0, observations, {}) is None
    assert rule.update(1.5, observations, {})["rule"] == "CrownClaim"
    observations["b"]["x"] = 0.06
    assert rule.update(1.51, observations, {}) is None
    observations["b"]["supporting_tiles"] = []
    observations["a"]["supporting_tiles"] = []
    assert rule.update(3.1, observations, {}) is None  # airborne/receiver cannot claim
