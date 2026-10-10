"""Real shared-model checks; no mocked physics or prerecorded success."""

import numpy as np
import mujoco
import math
from microduck_lab.arena.multiplayer import (
    GameConfig,
    build_game,
    ContactGraph,
    GameSchedule,
    Survivor,
    observe,
    GameReferee,
)


def test_independent_namespaces_and_public_actor_contract():
    c = GameConfig()
    m, d, ducks, tiles, spawns = build_game(c, 101)
    assert len(ducks) == 4 and m.nu == 56 and m.nsensor == 24
    assert len(set(a for duck in ducks.values() for a in duck.act_ids)) == 56
    assert len(set(q for duck in ducks.values() for q in duck.joint_qpos_idx)) == 56
    assert len({id(duck.last_action) for duck in ducks.values()}) == 4
    graph = ContactGraph(m, ducks, tiles)
    _, support, _ = graph.read(m, d, tiles)
    obs = observe(next(iter(ducks)), ducks, tiles, support, c, 0.0, {})
    assert set(obs) == {
        "schema",
        "sim_time_s",
        "robot",
        "visible_tiles",
        "nearby_ducks",
        "grid",
        "objective",
    }
    assert "seed" not in str(obs) and "release_at" not in str(obs)
    assert len({x["tile"] for x in spawns.values()}) == 4
    for name in ducks:
        assert any(
            owner == name and (m.geom_contype[g] or m.geom_conaffinity[g])
            for g, owner in graph.owners.items()
        )
        body_ids = [i for i in range(m.nbody) if m.body(i).name.startswith(name + "/")]
        assert abs(sum(m.body_mass[body_ids]) - 0.73724318) < 1e-10
        for part in ("jaw_soft", "trunk_base"):
            g = m.geom(name + "/" + part + "_external_envelope").id
            assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            assert m.geom_contype[g] and m.geom_conaffinity[g]
    assert not any(int(x // 65536) == 0 for x in m.exclude_signature), (
        "Receiver contact may not be excluded"
    )


def test_real_robot_robot_collision_without_state_assistance():
    c = GameConfig(players=2, grid=4, dt=0.0005)
    m, d, ducks, tiles, _ = build_game(c, 401)
    graph = ContactGraph(m, ducks, tiles)
    peak = 0.0
    for step in range(round(10.0 / c.dt)):
        if step % 40 == 0:
            for duck in ducks.values():
                duck.active_policy = "walk"
                delta = -duck.trunk_pos()[:2]
                error = math.atan2(delta[1], delta[0]) - duck.trunk_yaw()
                error = math.atan2(math.sin(error), math.cos(error))
                duck.set_command(
                    twist=(
                        0.45 if abs(error) < 0.45 else 0.0,
                        0.0,
                        float(np.clip(3 * error, -1.6, 1.6)),
                    )
                )
                duck.step()
        mujoco.mj_step(m, d)
        _, _, encounters = graph.read(m, d, tiles)
        peak = max(peak, max((e["force_N"] for e in encounters), default=0.0))
        if peak > 0.1:
            break
    assert peak > 0.1, "Two native robots must physically collide in one model"
    assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
    assert not any(w.number for w in d.warning)


def test_warning_response_and_only_release_input_changes_support():
    c = GameConfig()
    m, d, ducks, tiles, _ = build_game(c, 101)
    schedule = GameSchedule(c, 101)
    tile_id = schedule.plan[0]["tile"]
    before = d.qpos.copy()
    schedule.advance(d, tiles, 2.0)
    assert tiles[tile_id].state == "WARNING" and d.eq_active[tiles[tile_id].eq_id]
    assert np.array_equal(before, d.qpos)
    schedule.advance(d, tiles, 7.0)
    assert not d.eq_active[tiles[tile_id].eq_id] and np.array_equal(before, d.qpos)
    # Public neighbouring warning leaves ample time for a two-step escape.
    obs = dict(
        robot=dict(x=0.0, y=0.0, yaw=0.0, current_tile_id=12),
        grid=5,
        nearby_ducks=[],
        visible_tiles=[
            dict(id=12, state="WARNING", warning_remaining_s=0.2, centre_xy=[0.0, 0.0]),
            dict(
                id=17, state="WARNING", warning_remaining_s=4.0, centre_xy=[0.492, 0.0]
            ),
        ],
    )
    actor = Survivor()
    policy, command, intent = actor.choose(obs)
    assert (
        actor.target == 17
        and policy == "walk"
        and command[0] > 0
        and intent == "GO_TO_TILE"
    )


def test_toppled_last_duck_is_not_crowned():
    referee = GameReferee(("a", "b"))
    supported = dict(z=0.1, vz=0.0, supporting_tiles=[1], contact_step=0, upright=False)
    fallen = dict(z=-0.4, vz=-1.0, supporting_tiles=[], contact_step=0, upright=False)
    for t in (0.0, 0.31, 0.7):
        referee.update(t, {"a": supported, "b": fallen})
    assert referee.outcome()["status"] == "RUNNING"
    supported["upright"] = True
    referee.update(0.71, {"a": supported, "b": fallen})
    referee.update(0.95, {"a": supported, "b": fallen})
    assert referee.outcome()["winner"] == "a"
