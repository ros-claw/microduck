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


def test_crown_controller_holds_inside_target_instead_of_shuffling_forever():
    from microduck_lab.arena.island_rumble import Tactician

    cfg = RumbleConfig(claim_radius_m=0.08)
    obs = dict(
        sim_time_s=4,
        robot=dict(x=0.07, y=0, yaw=3.14, current_tile_id=4),
        visible_tiles=[
            dict(id=i, centre_xy=cfg.centre(i).tolist(), state="LOCKED", damage=0)
            for i in cfg.ids
        ],
        grid=3,
        island=4,
        passive=False,
        final_active=True,
        crown_mode=True,
        claim_radius_m=0.08,
        nearby_ducks=[],
    )
    actor = Tactician()
    policy, command, intent = actor.choose(obs)
    assert policy == "stand" and command == (0, 0, 0) and intent == "CLAIM_ISLAND"
    obs["robot"]["x"] = 0.1
    assert actor.choose(obs)[0] == "walk"


def test_crown_microgap_filter_does_not_accept_jumps_or_contested_airborne_rivals():
    from microduck_lab.arena.island_rumble import CrownClaim

    cfg = RumbleConfig(final_at_s=0, claim_radius_m=0.08)
    rule = CrownClaim(cfg)
    observations = {"a": dict(x=0.01, y=0, z=0.12, upright=True, supporting_tiles=[4])}
    result = None
    for i in range(1501):
        # 1ms gaps every 100ms; actual loaded duty cycle ~99%.
        observations["a"]["supporting_tiles"] = [] if i % 100 == 99 else [4]
        result = rule.update(i * 0.001, observations, {})
    assert result["winner"] == "a" and result["measured_loaded_fraction"] > 0.98
    for i in range(1502, 1515):
        observations["a"]["supporting_tiles"] = []
        assert rule.update(i * 0.001, observations, {}) is None
    assert rule.candidate is None  # >10ms unsupported resets the whole hold
    observations["a"]["supporting_tiles"] = [4]
    rule.update(2, observations, {})
    observations["b"] = dict(x=0.05, y=0, z=0.15, upright=True, supporting_tiles=[])
    assert rule.update(3.5, observations, {}) is None
    assert rule.candidate is None


def test_crown_race_scores_measured_support_only_and_preserves_contested_points():
    from microduck_lab.arena.island_rumble import CrownRace

    cfg = RumbleConfig(
        final_at_s=0, claim_s=0.5, claim_radius_m=0.08, score_mode="cumulative"
    )
    rule = CrownRace(cfg, ["a", "b"])
    obs = {
        "a": dict(x=0.01, y=0, z=0.12, upright=True, supporting_tiles=[4]),
        "b": dict(x=0.12, y=0, z=0.12, upright=True, supporting_tiles=[4]),
    }
    for i in range(1000):
        assert rule.update((i + 1) * cfg.dt, obs, {}) is None
    assert rule.scores["a"] == pytest.approx(0.25)
    obs["b"]["x"] = 0.03
    for i in range(1000):
        assert rule.update(0.25 + (i + 1) * cfg.dt, obs, {}) is None
    assert rule.scores["a"] == pytest.approx(0.25) and rule.scores["b"] == 0
    obs["b"]["x"] = 0.12
    obs["a"]["supporting_tiles"] = []
    for i in range(1000):
        assert rule.update(0.5 + (i + 1) * cfg.dt, obs, {}) is None
    assert rule.scores["a"] == pytest.approx(0.25)
    obs["a"]["supporting_tiles"] = [4]
    result = None
    for i in range(1000):
        result = rule.update(0.75 + (i + 1) * cfg.dt, obs, {})
    assert result["winner"] == "a" and result["rule"] == "CrownRace"
    assert result["loaded_crown_scores_s"]["a"] == pytest.approx(0.5)


def test_sources_changed_after_import_are_rejected_before_start(tmp_path, monkeypatch):
    from microduck_lab.arena import island_rumble

    monkeypatch.setattr(
        island_rumble, "_actor_source_hashes", lambda: {"tampered": True}
    )
    with pytest.raises(RuntimeError, match="restart the worker"):
        island_rumble.run_rumble(tmp_path / "should-not-exist")
    assert not (tmp_path / "should-not-exist").exists()
