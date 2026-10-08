import json
from types import SimpleNamespace
import numpy as np
import pytest
from microduck_lab.arena.tiles import Tile, TileScheduler
from microduck_lab.arena.referee import Referee
from microduck_lab.arena.controller import WarningResponder, centre


def robot(z=0.12, support=()):
    return dict(z=z, vz=0.0, supporting_tiles=list(support), contact_step=0)


def test_transient_airborne_and_low_supported_robot_are_not_eliminated():
    r = Referee()
    for t in np.arange(0, 1, 0.02):
        r.update(float(t), {"cream": robot(z=0.3)})
    assert r.outcome()["status"] == "RUNNING"
    r.update(1.0, {"cream": robot(z=-0.2)})
    r.update(1.2, {"cream": robot(z=0.12, support=(4,))})
    r.update(1.4, {"cream": robot(z=-0.2)})
    assert not r.update(1.65, {"cream": robot(z=-0.2)})
    assert r.update(1.7, {"cream": robot(z=-0.2)})[0]["state"] == "ELIMINATED"
    assert r.outcome()["winner"] is None
    assert r.outcome()["status"] == "ELIMINATED"


def test_simultaneous_loss_is_draw_and_single_survivor_wins():
    r = Referee(bodies=("a", "b"))
    for t in (1.0, 1.3):
        r.update(t, {"a": robot(z=-0.2), "b": robot(z=-0.2)})
    assert r.outcome()["status"] == "DRAW"
    r = Referee(bodies=("a", "b"))
    for t in (1.0, 1.3):
        r.update(t, {"a": robot(z=-0.2), "b": robot(support=(4,))})
    assert r.outcome()["winner"] == "b"


def test_lifecycle_requires_warning_and_observed_fall():
    tile = Tile(0, 0, 0, 0)
    data = SimpleNamespace(
        eq_active=np.array([True]), qpos=np.array([0.0, 0.0, -0.025]), qvel=np.zeros(3)
    )
    with pytest.raises(ValueError):
        tile.release(data, 1.0)
    tile.warn(1.0)
    before = data.qpos.copy()
    tile.release(data, 2.0)
    np.testing.assert_array_equal(data.qpos, before)
    assert not tile.update(data, 2.0)
    data.qpos[2] = -0.03
    data.qvel[2] = -0.2
    assert tile.update(data, 2.1)[0]["state"] == "FALLING"
    data.qpos[2] = -0.16
    assert tile.update(data, 2.2)[0]["state"] == "LOST"
    with pytest.raises(ValueError):
        tile.warn(3.0)


def test_seeded_schedule_is_unique_reproducible_and_rejects_zero_warning():
    assert TileScheduler(42).plan == TileScheduler(42).plan
    assert TileScheduler(42).plan != TileScheduler(43).plan
    assert len({x["tile"] for x in TileScheduler(42).plan}) == 9
    with pytest.raises(ValueError):
        TileScheduler(42, warning_s=0)
    with pytest.raises(ValueError):
        TileScheduler(42, warning_s=6, interval_s=5)
    public = Tile(0, 0, 0, 0).public(1.0)
    assert not any(
        x in json.dumps(public) for x in ("seed", "release_at", "future", "ttl")
    )


def test_responder_selects_only_safe_adjacent_tile_without_oracle():
    tiles = [dict(id=i, state="LOCKED") for i in range(9)]
    tiles[1]["state"] = "WARNING"
    pos = centre(1)
    obs = dict(
        robot=dict(x=pos[0], y=pos[1], yaw=0.0, current_tile_id=1), visible_tiles=tiles
    )
    c = WarningResponder()
    policy, command, intent = c.choose(obs)
    assert (
        c.target == 4 and policy == "walk" and command[0] > 0 and intent == "GO_TO_TILE"
    )
    tiles[4]["state"] = "RELEASED"
    c.choose(obs)
    assert c.target in (0, 2)
    tiles[0]["state"] = "WARNING"
    tiles[2]["state"] = "LOST"
    assert c.choose(obs)[2] == "NO_SAFE_OPTION"


def test_real_drop_has_contact_based_elimination_and_strict_semantic_replay(tmp_path):
    from microduck_lab.arena.survival import run
    from microduck_lab.arena.verify import verify

    path = tmp_path / "drop"
    audit = run(path, 11, 7.0, "drop_test", "hold")
    assert audit["outcome"]["status"] == "ELIMINATED"
    elimination = [e for e in audit["events"] if e["state"] == "ELIMINATED"][0]
    assert elimination["time"] > 6.3 and elimination["z"] < -0.12
    assert elimination["supporting_tile_ids"] == []
    assert verify(path)["passed"]
    audit["outcome"]["status"] = "WINNER"
    (path / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Outcome differs"):
        verify(path)
    del audit["hashes"]["contacts.jsonl.gz"]
    (path / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Incomplete replay manifest"):
        verify(path)


def test_last_body_still_falling_is_not_prematurely_declared_winner():
    r = Referee(bodies=("a", "b"))
    r.update(1.0, {"a": robot(z=-0.2), "b": robot(support=(4,))})
    r.update(1.1, {"a": robot(z=-0.2), "b": robot(z=-0.2)})
    r.update(1.3, {"a": robot(z=-0.2), "b": robot(z=-0.2)})
    assert r.outcome()["status"] == "RUNNING"
    r.update(1.4, {"a": robot(z=-0.2), "b": robot(z=-0.2)})
    assert r.outcome()["status"] == "DRAW"
