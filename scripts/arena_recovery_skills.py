"""Calibration only: initialize fallen poses in the actual arena collision world."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from microduck_lab.arena.multiplayer import ContactGraph
from microduck_lab.arena.survival_rumble import SurvivalConfig
from microduck_lab.arena.relay_world import build_relay_world
from microduck_lab.parkour.world import geom_vertices


def trial(kind, seed, recovery=True, variant="stand"):
    from microduck_lab.arena.world import ASSETS
    config = SurvivalConfig()
    m, d, ducks, tiles, _ = build_relay_world(config, seed)
    r = ducks["lavender"]
    r.bank.paths.update(roulade=str(ASSETS/"microduck/policies/roulade.onnx"), sitstand=str(ASSETS/"microduck/policies/alpha_sitstand.onnx"))
    graph = ContactGraph(m, ducks, tiles)
    axis = np.array([1., 0, 0]) if kind.startswith("side") else np.array([0., 1, 0])
    angle = {"face_up": -1, "face_down": 1, "side_left": 1, "side_right": -1}[kind]
    angle = angle * math.pi / 2 + np.random.default_rng(seed).uniform(-.05, .05)
    q = r.trunk_qpos_adr
    d.qpos[q:q+3] = [0, 0, .2]
    d.qpos[q+3:q+7] = [math.cos(angle/2), *(axis*math.sin(angle/2))]
    mujoco.mj_forward(m, d)
    points = np.concatenate([
        geom_vertices(m, d, g) for g in range(m.ngeom)
        if m.body(m.geom_bodyid[g]).name.startswith("lavender/")
        and (m.geom_contype[g] or m.geom_conaffinity[g])
    ])
    d.qpos[q+2] += .005 - points[:, 2].min()
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, 1200)
    initial = float(d.xmat[r.trunk_body_id, 8])
    stable = 0
    complete = None
    maxdepth = maxforce = 0.
    start = r.trunk_pos().copy()
    for step in range(160):
        for name, duck in ducks.items():
            duck.active_policy = "stand" if recovery or name != "lavender" else "walk"
            duck.set_command(twist=(0, 0, 0) if recovery or name != "lavender" else (.35, 0, 1.2))
            if name == "lavender" and .8 <= step*.02 < 1.6:
                if variant in ("roulade", "sitstand"):
                    duck.active_policy = variant
                elif variant == "head_pulse":
                    duck.set_command(head=(.4, -.4, 0, 0))
            duck.step()
        for _ in range(80):
            mujoco.mj_step(m, d)
            maxdepth = max(maxdepth, max((-c.dist for c in d.contact), default=0))
            maxforce = max(maxforce, float(np.max(np.abs(d.actuator_force))))
        _, support, _ = graph.read(m, d, tiles)
        good = (d.xmat[r.trunk_body_id, 8] > .95 and bool(support["lavender"])
                and np.linalg.norm(r.trunk_linvel()) < .12)
        stable = stable + .02 if good else 0
        if complete is None and stable >= .15:
            complete = round((step+1)*.02, 3)
    return dict(kind=kind, seed=seed, recovery=recovery, variant=variant, initial_up=initial,
                completion_s=complete, success_2s=initial < .5 and complete is not None and complete <= 2,
                displacement_m=float(np.linalg.norm(r.trunk_pos()[:2]-start[:2])),
                penetration_max_m=maxdepth, motor_force_max_Nm=maxforce,
                warnings=d.warning.number.tolist(), initialized_pose_only=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for seed in range(3):
        for kind in ("face_up", "face_down", "side_left", "side_right"):
            for variant in ("roulade", "sitstand", "head_pulse"):
                row = trial(kind, seed, variant=variant)
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2)+"\n")
